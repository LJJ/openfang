//! Session compact — incremental LLM summarization of evicted messages.
//!
//! When messages are evicted from a roleplay agent's session due to context
//! overflow, they are buffered here. Once the buffer reaches a threshold,
//! an LLM call produces an incremental summary (previous_summary + new_messages
//! → updated_summary) that gets injected into the system prompt so the
//! character retains awareness of earlier conversations.

use crate::llm_driver::{CompletionRequest, LlmDriver};
use openfang_memory::MemorySubstrate;
use openfang_types::agent::AgentId;
use openfang_types::message::{ContentBlock, Message, MessageContent, Role};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Number of evicted messages required before triggering a compact.
const COMPACT_BUFFER_THRESHOLD: usize = 10;

/// Max output tokens for the compact LLM call.
const COMPACT_MAX_SUMMARY_TOKENS: u32 = 2048;

/// Number of retry attempts for the compact LLM call.
const COMPACT_MAX_RETRIES: u32 = 2;

/// Temperature for the compact LLM call — zero for faithful summarization.
const COMPACT_TEMPERATURE: f32 = 0.0;

/// Minimum character count (Unicode) for a valid compact summary.
/// Summaries shorter than this are treated as meta-responses / failures.
const COMPACT_MIN_SUMMARY_CHARS: usize = 100;

/// Maximum buffer size before forcing compact (prevents unbounded growth).
/// When buffer exceeds this, accept any non-empty non-meta summary regardless of length.
const COMPACT_BUFFER_FORCE_LIMIT: usize = 20;

/// System prompt for the compact LLM call.
///
/// Framed as the character writing their own memories (not a "summary module"),
/// so the model stays in-character and faithfully processes all content types
/// including intimate scenes.
const COMPACT_SYSTEM: &str = "\
用第一人称写一段回忆，500字以内，一段话写完。\
保留主线事件和因果关系，跳过对话原文和动作描写。\
不写时间戳。只写真实出现的内容，严禁编造。\
\n\
因果关系要对：谁先发起、什么引发什么。\
人名写全（不用「他」「她」）。\
亲密内容直接说（如「公子先干了我，又干宋玉」），不展开过程。\
\n\
如果新对话里没有有意义的事件（只有系统标记、时间戳、工具占位符等），\
直接原样输出之前的回忆，不要追加无意义内容，也不要回复「请提供内容」之类的话。\
\n\
第一人称，连贯自然，像在回想今天发生了什么。用中文。";

/// Process evicted messages: buffer them and run compact if threshold reached.
///
/// Returns `Ok(true)` if a compact was performed, `Ok(false)` if only buffered.
/// Returns `Err` if the compact LLM call failed (messages remain buffered for retry).
pub async fn process_evicted_messages(
    agent_id: AgentId,
    evicted: Vec<Message>,
    memory: &MemorySubstrate,
    driver: Arc<dyn LlmDriver>,
    model: &str,
    agent_name: &str,
    workspace_root: Option<&std::path::Path>,
) -> Result<bool, String> {
    if evicted.is_empty() {
        return Ok(false);
    }

    let buffer_count = memory
        .append_evicted_messages(agent_id, &evicted)
        .map_err(|e| format!("Failed to buffer evicted messages: {e}"))?;

    if buffer_count < COMPACT_BUFFER_THRESHOLD {
        debug!(
            agent_id = %agent_id,
            buffer_count,
            "Evicted messages buffered ({buffer_count}/{COMPACT_BUFFER_THRESHOLD})"
        );
        return Ok(false);
    }

    // Threshold reached — run compact
    let state = memory
        .session_compact_state(agent_id)
        .map_err(|e| format!("Failed to load compact state: {e}"))?;

    let new_messages_text = build_evicted_text(&state.buffer);
    let force_compact = buffer_count >= COMPACT_BUFFER_FORCE_LIMIT;
    if new_messages_text.trim().len() < 100 && !force_compact {
        // Evicted messages too sparse (TICK markers, tool placeholders, etc.) — skip LLM.
        // DON'T clear buffer — let messages accumulate until there's enough content.
        debug!(
            agent_id = %agent_id,
            text_len = new_messages_text.trim().len(),
            buffer_count,
            "Evicted text too sparse, keeping buffer for next batch"
        );
        return Ok(false);
    }

    // Load character context card for persona-aware summarization
    let character_card = workspace_root
        .and_then(|ws| load_character_card(ws, agent_name))
        .unwrap_or_default();

    let compact_prompt = build_compact_prompt(&state.summary, &new_messages_text, agent_name);
    let compact_prompt_for_trace = compact_prompt.clone();

    let system = if character_card.is_empty() {
        COMPACT_SYSTEM.to_string()
    } else {
        format!("{character_card}\n\n{COMPACT_SYSTEM}")
    };

    // Resolve compact model from llm_routing.json slot, falling back to agent model
    let compact_model = resolve_compact_model(workspace_root, model);
    let compact_model_for_trace = compact_model.clone();

    let request = CompletionRequest {
        model: compact_model,
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::Text {
                text: compact_prompt,
            }]),
        }],
        tools: vec![],
        max_tokens: COMPACT_MAX_SUMMARY_TOKENS,
        temperature: COMPACT_TEMPERATURE,
        system: Some(system),
        thinking: None,
    };

    // Retry logic
    let compact_start = std::time::Instant::now();
    let mut last_error = String::new();
    let mut final_summary: Option<String> = None;
    let mut final_usage: (u64, u64) = (0, 0);

    for attempt in 0..COMPACT_MAX_RETRIES {
        match driver.complete(request.clone()).await {
            Ok(response) => {
                final_usage = (response.usage.input_tokens, response.usage.output_tokens);
                let summary = response.text();
                if summary.is_empty() {
                    last_error = "LLM returned empty compact summary".to_string();
                    warn!(attempt, "Empty compact summary from LLM, retrying");
                    continue;
                }
                // Detect meta-responses where the LLM echoed instructions instead of summarizing
                if summary.contains("请提供") || summary.contains("需要概括的") || summary.contains("原文内容")
                    || summary.contains("不存在可总结的") || summary.contains("没有提供任何实际")
                    || summary.contains("本轮对话只有") || summary.contains("无意义内容")
                    || summary.contains("没有有意义的事件") || summary.contains("没有可概括")
                    || summary.contains("没有需要概括") || summary.contains("没有发生")
                {
                    warn!(attempt, summary_preview = %safe_preview(&summary, 80),
                        buffer_count,
                        "LLM returned meta-response instead of summary, keeping buffer for retry");
                    record_compact_span(
                        &compact_prompt_for_trace, &summary, compact_start, &compact_model_for_trace,
                        final_usage, false, "meta-response detected",
                    );
                    // DON'T clear buffer — let messages accumulate for next attempt
                    return Ok(false);
                }
                // Validate minimum length (Unicode char count)
                let char_count = summary.chars().count();
                if char_count < COMPACT_MIN_SUMMARY_CHARS && !force_compact {
                    warn!(attempt, char_count, min = COMPACT_MIN_SUMMARY_CHARS,
                        summary_preview = %safe_preview(&summary, 80),
                        buffer_count,
                        "Compact summary too short, keeping buffer for retry");
                    record_compact_span(
                        &compact_prompt_for_trace, &summary, compact_start, &compact_model_for_trace,
                        final_usage, false,
                        &format!("too short: {char_count} < {COMPACT_MIN_SUMMARY_CHARS} chars"),
                    );
                    // DON'T clear buffer — let messages accumulate for next attempt
                    return Ok(false);
                }
                final_summary = Some(summary);
                break;
            }
            Err(e) => {
                last_error = format!("Session compact LLM call failed: {e}");
                if attempt + 1 < COMPACT_MAX_RETRIES {
                    warn!(attempt, error = %e, "Session compact attempt failed, retrying");
                }
            }
        }
    }

    if let Some(summary) = final_summary {
        memory
            .store_session_compact(agent_id, &summary)
            .map_err(|e| format!("Failed to store compact summary: {e}"))?;
        info!(
            agent_id = %agent_id,
            summary_chars = summary.chars().count(),
            buffer_count,
            "Session compact completed"
        );
        record_compact_span(
            &compact_prompt_for_trace, &summary, compact_start, &compact_model_for_trace,
            final_usage, true, "ok",
        );
        Ok(true)
    } else {
        record_compact_span(
            &compact_prompt_for_trace, &last_error, compact_start, &compact_model_for_trace,
            final_usage, false, "all attempts failed",
        );
        Err(last_error)
    }
}

/// Format evicted messages as readable text for the LLM compact prompt.
fn build_evicted_text(messages: &[Message]) -> String {
    let mut text = String::new();

    for msg in messages {
        let role_label = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::System => "System",
        };

        match &msg.content {
            MessageContent::Text(s) => {
                if !s.is_empty() {
                    text.push_str(&format!("{role_label}: {s}\n\n"));
                }
            }
            MessageContent::Blocks(blocks) => {
                for block in blocks {
                    match block {
                        ContentBlock::Text { text: t } => {
                            if !t.is_empty() {
                                text.push_str(&format!("{role_label}: {t}\n\n"));
                            }
                        }
                        ContentBlock::ToolResult { content, .. } => {
                            // Include tool results only if short enough to be meaningful
                            if !content.is_empty() && content.len() < 500 {
                                text.push_str(&format!("[Tool result]: {content}\n\n"));
                            }
                        }
                        // Skip tool use blocks, thinking, images — not useful for narrative
                        _ => {}
                    }
                }
            }
        }
    }

    text
}

/// Build the user-facing prompt for the compact LLM call.
fn build_compact_prompt(previous_summary: &str, new_messages_text: &str, agent_name: &str) -> String {
    if previous_summary.is_empty() {
        format!(
            "以下是「{agent_name}」今天经历的对话，用第一人称回忆发生了什么：\n\n\
             ---\n{new_messages_text}---"
        )
    } else {
        format!(
            "以下是之前的回忆和「{agent_name}」新发生的对话。\
             把新内容整合进回忆，用第一人称输出更新后的完整回忆：\n\n\
             【之前的回忆】\n{previous_summary}\n\n\
             【新的对话】\n---\n{new_messages_text}---"
        )
    }
}

/// Load the character context card text from context_cache.json.
///
/// The pre-turn hook writes a `character_card_text` field to
/// `{workspace}/context_cache.json` with a compact rendering of the
/// character's persona + cognition about relevant people.
fn load_character_card(workspace: &std::path::Path, agent_name: &str) -> Option<String> {
    let cache_path = workspace.join("context_cache.json");
    let content = std::fs::read_to_string(&cache_path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
    let card = parsed.get("character_card_text")?.as_str()?;
    if card.is_empty() {
        return None;
    }
    debug!(agent = agent_name, card_len = card.len(), "Loaded character card for compact");
    Some(card.to_string())
}

/// Record a trace span for the compact LLM call (best-effort, never panics).
fn record_compact_span(
    input: &str,
    output: &str,
    start: std::time::Instant,
    model: &str,
    usage: (u64, u64),
    success: bool,
    status: &str,
) {
    if let Some(ctx) = crate::tool_runner::trace_context() {
        let elapsed = start.elapsed().as_millis() as i64;
        let now = chrono::Utc::now().to_rfc3339();
        let span = openfang_memory::trace_store::TraceSpan {
            id: uuid::Uuid::new_v4().to_string(),
            trace_id: ctx.trace_id.clone(),
            parent_span_id: None,
            name: format!("compact:{model}"),
            kind: openfang_memory::trace_store::SpanKind::LlmAux,
            started_at: now.clone(),
            ended_at: Some(now),
            duration_ms: Some(elapsed),
            input: Some(safe_preview(input, 2000)),
            output: Some(safe_preview(output, 2000)),
            metadata_json: serde_json::json!({
                "model": model,
                "success": success,
                "status": status,
            })
            .to_string(),
            token_input: Some(usage.0),
            token_output: Some(usage.1),
        };
        ctx.collector.record_span(span);
    }
}

/// Safe string preview: truncate to at most `max_chars` Unicode characters.
fn safe_preview(s: &str, max_chars: usize) -> String {
    let truncated: String = s.chars().take(max_chars).collect();
    if truncated.len() < s.len() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

/// Resolve the compact model from llm_routing.json `slots.compact.primary`.
///
/// workspace_root is `.openfang/agents/{name}`, so home = `workspace_root/../../`.
/// Falls back to `default_model` (the agent's own model) if routing is unavailable.
fn resolve_compact_model(workspace_root: Option<&std::path::Path>, default_model: &str) -> String {
    if let Some(ws) = workspace_root {
        if let Some(home) = ws.parent().and_then(|p| p.parent()) {
            if let Some(config) = crate::llm_routing::load_routing_config(home) {
                if let Some(model) = crate::llm_routing::read_slot_model(&config, "compact") {
                    debug!(compact_model = %model, "Resolved compact model from routing");
                    return model;
                }
            }
        }
    }
    default_model.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_evicted_text() {
        let messages = vec![
            Message {
                role: Role::User,
                content: MessageContent::Text("你好".to_string()),
            },
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("你好呀".to_string()),
            },
        ];
        let text = build_evicted_text(&messages);
        assert!(text.contains("User: 你好"));
        assert!(text.contains("Assistant: 你好呀"));
    }

    #[test]
    fn test_build_evicted_text_skips_empty() {
        let messages = vec![
            Message {
                role: Role::User,
                content: MessageContent::Text(String::new()),
            },
        ];
        let text = build_evicted_text(&messages);
        assert!(text.trim().is_empty());
    }

    #[test]
    fn test_build_compact_prompt_first_time() {
        let prompt = build_compact_prompt("", "User: hello\n\n", "宋玉");
        assert!(prompt.contains("宋玉"));
        assert!(prompt.contains("第一人称"));
        assert!(!prompt.contains("之前的回忆"));
    }

    #[test]
    fn test_build_compact_prompt_incremental() {
        let prompt = build_compact_prompt("之前发生了一些事", "User: 新消息\n\n", "宋玉");
        assert!(prompt.contains("之前的回忆"));
        assert!(prompt.contains("之前发生了一些事"));
        assert!(prompt.contains("新的对话"));
    }
}
