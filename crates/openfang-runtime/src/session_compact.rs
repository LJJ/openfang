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
use tracing::{debug, warn};

/// Number of evicted messages required before triggering a compact.
const COMPACT_BUFFER_THRESHOLD: usize = 5;

/// Max output tokens for the compact LLM call.
const COMPACT_MAX_SUMMARY_TOKENS: u32 = 2048;

/// Number of retry attempts for the compact LLM call.
const COMPACT_MAX_RETRIES: u32 = 2;

/// Temperature for the compact LLM call (low for deterministic summarization).
const COMPACT_TEMPERATURE: f32 = 0.3;

/// System prompt for the compact LLM call.
const COMPACT_SYSTEM: &str = "\
你是角色记忆回顾模块。你需要以指定角色的第一人称视角，回顾之前发生的事情。\
写成流水账式的回忆——今天做了什么、跟谁说了什么话、去了哪里、吃了什么、发生了什么事。\
重点保留：具体的人名（必须写全名，禁止用「他」「她」「大家」「有人」代替）、\
具体的地点、具体做了什么事、说了什么关键的话、情绪变化、未解决的话题、重要的承诺或决定。\
语气自然，像在写日记一样朴素地记录，不要写空泛的情绪感悟。\
不要加标题或解释，只输出回忆本身。用中文。控制在800字以内。";

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
    if new_messages_text.trim().is_empty() {
        // All evicted messages were empty (tool-only, etc.) — clear buffer, skip LLM
        let _ = memory.store_session_compact(agent_id, &state.summary);
        return Ok(false);
    }

    // Load character context card from context_cache.json (written by pre-turn hook)
    let character_card = workspace_root
        .and_then(|ws| load_character_card(ws, agent_name))
        .unwrap_or_default();

    let compact_prompt = build_compact_prompt(&state.summary, &new_messages_text, agent_name);

    // Build system prompt: character card + generic compact instructions
    let system = if character_card.is_empty() {
        COMPACT_SYSTEM.to_string()
    } else {
        format!("{character_card}\n\n{COMPACT_SYSTEM}")
    };

    let request = CompletionRequest {
        model: model.to_string(),
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
    let mut last_error = String::new();
    for attempt in 0..COMPACT_MAX_RETRIES {
        match driver.complete(request.clone()).await {
            Ok(response) => {
                let summary = response.text();
                if summary.is_empty() {
                    last_error = "LLM returned empty compact summary".to_string();
                    warn!(attempt, "Empty compact summary from LLM, retrying");
                    continue;
                }
                memory
                    .store_session_compact(agent_id, &summary)
                    .map_err(|e| format!("Failed to store compact summary: {e}"))?;
                debug!(
                    agent_id = %agent_id,
                    summary_len = summary.len(),
                    buffer_count,
                    "Session compact completed"
                );
                return Ok(true);
            }
            Err(e) => {
                last_error = format!("Session compact LLM call failed: {e}");
                if attempt + 1 < COMPACT_MAX_RETRIES {
                    warn!(attempt, error = %e, "Session compact attempt failed, retrying");
                }
            }
        }
    }

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
fn build_compact_prompt(previous_summary: &str, new_messages_text: &str, agent_name: &str) -> String {
    if previous_summary.is_empty() {
        format!(
            "以下是今天「{agent_name}」经历的对话。\
             请以「{agent_name}」的第一人称视角回顾这段经历：\n\n\
             ---\n{new_messages_text}---"
        )
    } else {
        format!(
            "以下是「{agent_name}」之前的回忆和新发生的对话。\
             请以「{agent_name}」的第一人称视角，把新内容整合进回忆，输出更新后的完整回忆：\n\n\
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
