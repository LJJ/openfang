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

/// Compact summary token budget——硬 cap 而非随输入伸缩，防止 rolling summary 滚雪球。
/// Rolling summary 是"今天为止的完整回忆"，一天之内它应当稳定在可控范围而不是线性增长。
/// 目标长度：1000-1500 chars（≈ 800-1200 tokens output），通过 prompt 里的 aging 规则实现——
/// 越老的事越压缩，越近的事越详细，总长维持在 cap 以下。
const COMPACT_MIN_TOKENS: u32 = 600;
const COMPACT_MAX_TOKENS_CAP: u32 = 1600;

/// 输出 token 预算——和输入规模解耦，统一硬 cap。
/// 输入小的场景（第一次 compact、短对话）可以少点，其余一律到 cap。
fn compute_summary_token_budget(previous_summary: &str, new_messages_text: &str) -> u32 {
    let input_chars = previous_summary.chars().count() + new_messages_text.chars().count();
    // 首次压缩且内容很少时不必到 cap；其他情况一律 cap，由 prompt 控制内部分配
    if previous_summary.is_empty() && input_chars < 1500 {
        return COMPACT_MIN_TOKENS;
    }
    COMPACT_MAX_TOKENS_CAP
}

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

/// System prompt template for the compact LLM call.
///
/// Framed as the character writing their own memories (not a "summary module"),
/// so the model stays in-character and faithfully processes all content types
/// including intimate scenes. `{AGENT}` is replaced with the current agent_name
/// so the example doesn't accidentally anchor the LLM onto another character.
fn compact_instructions(agent_name: &str) -> String {
    format!(
        "你是 {agent}。用 {agent} 的第一人称写一段自然的中文回忆——像真人在心里默默回想。\n\
         \n\
         - 「我」=={agent}，永远不要把别的角色当成「我」\n\
         - 输入里对 {agent} 的一切描述（如「{agent}走过来」「她把杯子放下」）都要改写成「我走过来」「我把杯子放下」\n\
         - 输入里其他角色的名字和动作照旧用第三人称保留（「宋玉说」「紫灵看着我」这种不变）\n\
         - **严格只写输入里实际发生过的事。** 输入里没有的情节、对话、地点、菜名、人物、动作、时间段，一律不得出现。宁可回忆写得短、跳过大段时间，也绝对不要靠常识或合理想象去补全一段看起来该有的剧情\n\
         - **事实细节必须照搬**：输入里出现的地点名（比如「村上一屋」「合生汇」）、菜名（「荞麦面」「茶碗蒸」）、人名、原话，一字不改地用，不要替换成近似的东西\n\
         - **输入里覆盖到的事件都要落到字面上**：如果输入横跨几个场景（出门→到店→点菜→吃饭→回家），每个出现过的环节都要写进去，不要只展开开头一个画面就结束。但反过来，输入没覆盖的时间段就直接跳过，不要强行串连\n\
         - 氛围和心情服务于事件，不要让场景描写抢掉事件本身\n\
         - 关键话语可以摘录原话或大意\n\
         - 一段话写完，不用小标题、不列清单、不写时间戳\n\
         - 如果新内容没什么实质事件（比如只有 TICK 标记、没有对话和动作），就原样输出之前的回忆，不要硬写",
        agent = agent_name
    )
}

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

    // display_name 用作 prompt 里的人称标签——优先中文（"紫灵"），fallback 到 agent_name（"ziling"）
    let display_name = workspace_root
        .and_then(load_display_name)
        .unwrap_or_else(|| agent_name.to_string());

    let compact_prompt = build_compact_prompt(&state.summary, &new_messages_text, &display_name);
    let compact_prompt_for_trace = compact_prompt.clone();

    // 只在"一天第一次 compact"（state.summary 为空）时注入昨天摘要
    let yesterday = if state.summary.is_empty() {
        workspace_root
            .and_then(load_yesterday_summary)
            .unwrap_or_default()
    } else {
        String::new()
    };

    let system = build_compact_system(&character_card, &yesterday, &display_name);

    // Resolve compact model from llm_routing.json slot, falling back to agent model
    let compact_model = resolve_compact_model(workspace_root, model);
    let compact_model_for_trace = compact_model.clone();

    let max_tokens = compute_summary_token_budget(&state.summary, &new_messages_text);
    let request = CompletionRequest {
        model: compact_model,
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::Text {
                text: compact_prompt,
            }]),
        }],
        tools: vec![],
        max_tokens,
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
                if summary.contains("请提供")
                    || summary.contains("需要概括的")
                    || summary.contains("原文内容")
                    || summary.contains("不存在可总结的")
                    || summary.contains("没有提供任何实际")
                    || summary.contains("本轮对话只有")
                    || summary.contains("无意义内容")
                    || summary.contains("没有有意义的事件")
                    || summary.contains("没有可概括")
                    || summary.contains("没有需要概括")
                    || summary.contains("没有发生")
                {
                    warn!(attempt, summary_preview = %safe_preview(&summary, 80),
                        buffer_count,
                        "LLM returned meta-response instead of summary, keeping buffer for retry");
                    record_compact_span(
                        &compact_prompt_for_trace,
                        &summary,
                        compact_start,
                        &compact_model_for_trace,
                        final_usage,
                        false,
                        "meta-response detected",
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
                        &compact_prompt_for_trace,
                        &summary,
                        compact_start,
                        &compact_model_for_trace,
                        final_usage,
                        false,
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
            &compact_prompt_for_trace,
            &summary,
            compact_start,
            &compact_model_for_trace,
            final_usage,
            true,
            "ok",
        );
        Ok(true)
    } else {
        record_compact_span(
            &compact_prompt_for_trace,
            &last_error,
            compact_start,
            &compact_model_for_trace,
            final_usage,
            false,
            "all attempts failed",
        );
        Err(last_error)
    }
}

/// Force a full compact: merge previous summary + pending buffer + current session
/// messages into one new rolling summary, no threshold or min-length gates.
///
/// Caller (typically the kernel on manual `/compact`) is responsible for
/// persisting the returned summary via `store_session_compact` and clearing
/// `session.messages` afterwards.
///
/// Returns `Ok(new_summary)` on success. Returns `Err` if the LLM call failed
/// or returned a meta-response on every retry. Returns `Ok(previous_summary)`
/// verbatim if there are no buffered messages and no session messages (nothing
/// new to fold in).
pub async fn force_compact_session(
    previous_summary: &str,
    buffer_messages: &[Message],
    session_messages: &[Message],
    driver: Arc<dyn LlmDriver>,
    model: &str,
    agent_name: &str,
    workspace_root: Option<&std::path::Path>,
) -> Result<String, String> {
    let mut all_messages = Vec::with_capacity(buffer_messages.len() + session_messages.len());
    all_messages.extend_from_slice(buffer_messages);
    all_messages.extend_from_slice(session_messages);
    let new_messages_text = build_evicted_text(&all_messages);

    if new_messages_text.trim().is_empty() {
        // Nothing new to fold in — keep the old summary untouched.
        return Ok(previous_summary.to_string());
    }

    let character_card = workspace_root
        .and_then(|ws| load_character_card(ws, agent_name))
        .unwrap_or_default();

    let display_name = workspace_root
        .and_then(load_display_name)
        .unwrap_or_else(|| agent_name.to_string());

    let compact_prompt = build_compact_prompt(previous_summary, &new_messages_text, &display_name);
    let compact_prompt_for_trace = compact_prompt.clone();

    // 只在"一天第一次 compact"（previous_summary 为空）时注入昨天摘要
    let yesterday = if previous_summary.is_empty() {
        workspace_root
            .and_then(load_yesterday_summary)
            .unwrap_or_default()
    } else {
        String::new()
    };

    let system = build_compact_system(&character_card, &yesterday, &display_name);

    let compact_model = resolve_compact_model(workspace_root, model);
    let compact_model_for_trace = compact_model.clone();

    let max_tokens = compute_summary_token_budget(previous_summary, &new_messages_text);
    let request = CompletionRequest {
        model: compact_model,
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::Text {
                text: compact_prompt,
            }]),
        }],
        tools: vec![],
        max_tokens,
        temperature: COMPACT_TEMPERATURE,
        system: Some(system),
        thinking: None,
    };

    let compact_start = std::time::Instant::now();
    let mut last_error = String::new();
    let mut final_usage: (u64, u64) = (0, 0);

    for attempt in 0..COMPACT_MAX_RETRIES {
        match driver.complete(request.clone()).await {
            Ok(response) => {
                final_usage = (response.usage.input_tokens, response.usage.output_tokens);
                let summary = response.text();
                if summary.is_empty() {
                    last_error = "LLM returned empty summary".to_string();
                    warn!(attempt, "Empty force-compact summary, retrying");
                    continue;
                }
                if summary.contains("请提供")
                    || summary.contains("需要概括的")
                    || summary.contains("原文内容")
                    || summary.contains("不存在可总结的")
                    || summary.contains("没有提供任何实际")
                    || summary.contains("本轮对话只有")
                    || summary.contains("无意义内容")
                    || summary.contains("没有有意义的事件")
                    || summary.contains("没有可概括")
                    || summary.contains("没有需要概括")
                    || summary.contains("没有发生")
                {
                    last_error = "LLM returned meta-response".to_string();
                    warn!(
                        attempt,
                        summary_preview = %safe_preview(&summary, 80),
                        "Force compact got meta-response, retrying"
                    );
                    continue;
                }
                record_compact_span(
                    &compact_prompt_for_trace,
                    &summary,
                    compact_start,
                    &compact_model_for_trace,
                    final_usage,
                    true,
                    "force ok",
                );
                info!(
                    summary_chars = summary.chars().count(),
                    buffer_count = buffer_messages.len(),
                    session_count = session_messages.len(),
                    "Force compact completed"
                );
                return Ok(summary);
            }
            Err(e) => {
                last_error = format!("Force compact LLM call failed: {e}");
                if attempt + 1 < COMPACT_MAX_RETRIES {
                    warn!(attempt, error = %e, "Force compact attempt failed, retrying");
                }
            }
        }
    }

    record_compact_span(
        &compact_prompt_for_trace,
        &last_error,
        compact_start,
        &compact_model_for_trace,
        final_usage,
        false,
        "all attempts failed",
    );
    Err(last_error)
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
fn build_compact_prompt(
    previous_summary: &str,
    new_messages_text: &str,
    agent_name: &str,
) -> String {
    if previous_summary.is_empty() {
        format!(
            "我是 {agent_name}。用我（{agent_name}）的第一人称，把下面这段内容写成一段自然的回忆。\n\
             **只写输入里实际出现的事**——没写进来的情节、地点、菜名、动作绝对不要补。\n\
             输入里有出现的环节（比如去了哪、吃了什么、说了什么）都要落到字面上，名字原话一字不改。\n\n\
             【对话】\n---\n{new_messages_text}---"
        )
    } else {
        format!(
            "我是 {agent_name}。用我（{agent_name}）的第一人称，把【之前的回忆】和【新的对话】合并成一段更新后的回忆。\n\
             **只写两段里实际出现的事**——任何一段没写过的情节、地点、菜名、动作都不要补。\n\
             名字和原话原样保留，输入里出现的环节都落到字面上。\n\n\
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
/// 拼装 compact 的 system prompt。
///   character_card（持久身份 + 认知 + 此刻 + 日程）—— 每次都注入
///   yesterday（昨天摘要）—— 仅 previous_summary 为空时传入，给一天的首次 compact 提供跨天锚点
///   compact_instructions(agent_name)（指令）—— 结尾，示例里的"我"=agent_name
fn build_compact_system(character_card: &str, yesterday: &str, agent_name: &str) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(3);
    if !character_card.is_empty() {
        parts.push(character_card.to_string());
    }
    if !yesterday.is_empty() {
        parts.push(format!("## {agent_name}昨天的回忆\n\n{yesterday}"));
    }
    parts.push(compact_instructions(agent_name));
    parts.join("\n\n")
}

fn load_character_card(workspace: &std::path::Path, agent_name: &str) -> Option<String> {
    let cache_path = workspace.join("context_cache.json");
    let content = std::fs::read_to_string(&cache_path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
    let card = parsed.get("character_card_text")?.as_str()?;
    if card.is_empty() {
        return None;
    }
    debug!(
        agent = agent_name,
        card_len = card.len(),
        "Loaded character card for compact"
    );
    Some(card.to_string())
}

/// 加载昨天的摘要（由 pre-turn hook 写入 context_cache.yesterday_summary_for_compact）。
/// 只在一天第一次 compact（previous_summary 为空）时使用——提供跨天的叙事连贯性。
/// 今天已经有 summary 后，它承载了今天的连续性，昨天不再需要重复注入。
fn load_yesterday_summary(workspace: &std::path::Path) -> Option<String> {
    let cache_path = workspace.join("context_cache.json");
    let content = std::fs::read_to_string(&cache_path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
    let ys = parsed.get("yesterday_summary_for_compact")?.as_str()?;
    if ys.is_empty() {
        None
    } else {
        Some(ys.to_string())
    }
}

/// 加载中文显示名（由 pre-turn hook 写入 context_cache.display_name）。
/// 用于 compact prompt 里的人称锚定——优先用中文名（"紫灵"），避免拼音 agent_name
/// （"ziling"）和中文 character_card 混排导致的语感割裂。
pub(crate) fn load_display_name(workspace: &std::path::Path) -> Option<String> {
    let cache_path = workspace.join("context_cache.json");
    let content = std::fs::read_to_string(&cache_path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
    let name = parsed.get("display_name")?.as_str()?;
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
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
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text(String::new()),
        }];
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
        assert!(prompt.contains("宋玉"));
    }

    #[test]
    fn test_compact_instructions_anchors_agent_name() {
        let ziling = compact_instructions("紫灵");
        assert!(ziling.contains("你是 紫灵"));
        assert!(ziling.contains("紫灵走过来"));
        assert!(!ziling.contains("宋玉走过来"));
    }
}
