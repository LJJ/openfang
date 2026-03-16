//! Session projections separate dialogue memory from execution trace.
//!
//! A single runtime turn may need rich tool-use / tool-result history to
//! complete a complex task, but that same raw execution trace should not be
//! carried into future turns as durable dialogue context.
//!
//! Trace filter rules are split into two layers:
//! - **Code defaults**: generic patterns shared by all agents (below).
//! - **Config extras**: per-agent patterns from `metadata.session_projection.extra_trace_filters`.

use openfang_types::message::{ContentBlock, Message, MessageContent, Role};

/// High-level message lanes used by the projection layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageLane {
    /// Natural-language dialogue that should persist across turns.
    Dialogue,
    /// Execution-only trace such as tool use and tool results.
    Execution,
    /// Mixed content that contains both dialogue and execution blocks.
    Mixed,
    /// Non-dialogue rich media such as images.
    RichMedia,
    /// Internal reasoning blocks.
    Reasoning,
    /// Empty / unknown payload after projection.
    Empty,
}

/// Generic exact-match tokens for assistant messages that are internal trace.
const DEFAULT_EXACT_TOKENS: &[&str] = &["no_reply", "[no reply needed]", "ok. no_reply"];

/// Generic prefix patterns for assistant messages that are internal trace.
const DEFAULT_ASSISTANT_PREFIXES: &[&str] = &[
    "[task completed",
    "[the model returned an empty response",
    "[no response]",
];

/// Generic prefix patterns for user messages that are internal trace.
const DEFAULT_USER_PREFIXES: &[&str] = &["[system] please output your turn script directly"];

fn is_internal_trace_text(role: Role, text: &str, extra_filters: &[String]) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    let lower = trimmed.to_lowercase();

    // Exact match tokens (assistant only)
    if role == Role::Assistant && DEFAULT_EXACT_TOKENS.iter().any(|token| lower == *token) {
        return true;
    }

    match role {
        Role::Assistant => {
            // Generic prefixes
            if DEFAULT_ASSISTANT_PREFIXES
                .iter()
                .any(|prefix| lower.starts_with(prefix))
            {
                return true;
            }
            // Per-agent extra filters (prefix match)
            if extra_filters
                .iter()
                .any(|filter| lower.starts_with(&filter.to_lowercase()))
            {
                return true;
            }
            false
        }
        Role::User => DEFAULT_USER_PREFIXES
            .iter()
            .any(|prefix| lower.starts_with(prefix)),
        Role::System => true,
    }
}

/// Classify a message by the kinds of blocks it contains.
pub fn classify_message_lane(message: &Message) -> MessageLane {
    match &message.content {
        MessageContent::Text(text) => {
            if text.trim().is_empty() {
                MessageLane::Empty
            } else {
                MessageLane::Dialogue
            }
        }
        MessageContent::Blocks(blocks) => {
            let mut has_text = false;
            let mut has_media = false;
            let mut has_execution = false;
            let mut has_reasoning = false;

            for block in blocks {
                match block {
                    ContentBlock::Text { text } if !text.trim().is_empty() => has_text = true,
                    ContentBlock::Image { .. } => has_media = true,
                    ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => {
                        has_execution = true
                    }
                    ContentBlock::Thinking { .. } => has_reasoning = true,
                    ContentBlock::Unknown | ContentBlock::Text { .. } => {}
                }
            }

            match (has_text, has_media, has_execution, has_reasoning) {
                (false, false, false, false) => MessageLane::Empty,
                (true, false, false, false) => MessageLane::Dialogue,
                (false, true, false, false) => MessageLane::RichMedia,
                (false, false, true, false) => MessageLane::Execution,
                (false, false, false, true) => MessageLane::Reasoning,
                _ => MessageLane::Mixed,
            }
        }
    }
}

fn strip_execution_blocks(
    role: Role,
    content: &MessageContent,
    extra_filters: &[String],
) -> Option<MessageContent> {
    match content {
        MessageContent::Text(text) => {
            if is_internal_trace_text(role, text, extra_filters) {
                None
            } else {
                Some(MessageContent::Text(text.clone()))
            }
        }
        MessageContent::Blocks(blocks) => {
            let kept: Vec<ContentBlock> = blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => {
                        if is_internal_trace_text(role, text, extra_filters) {
                            None
                        } else {
                            Some(ContentBlock::Text { text: text.clone() })
                        }
                    }
                    ContentBlock::Unknown => Some(ContentBlock::Unknown),
                    ContentBlock::Image { .. } => None,
                    ContentBlock::ToolUse { .. }
                    | ContentBlock::ToolResult { .. }
                    | ContentBlock::Thinking { .. } => None,
                })
                .collect();

            if kept.is_empty() {
                None
            } else {
                Some(MessageContent::Blocks(kept))
            }
        }
    }
}

/// Keep only durable dialogue content for future turns.
///
/// `extra_trace_filters` are per-agent prefix patterns from config
/// (`metadata.session_projection.extra_trace_filters`). Pass an empty
/// slice when no per-agent config is available.
pub fn project_for_persistent_dialogue(
    messages: &[Message],
    extra_trace_filters: &[String],
) -> Vec<Message> {
    messages
        .iter()
        .filter_map(|msg| {
            strip_execution_blocks(msg.role, &msg.content, extra_trace_filters).map(|content| {
                Message {
                    role: msg.role,
                    content,
                }
            })
        })
        .collect()
}

/// Read extra trace filters from agent manifest metadata.
///
/// Looks for `metadata.session_projection.extra_trace_filters` (string array).
/// Returns an empty vec if not configured.
pub fn extra_trace_filters_from_metadata(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Vec<String> {
    metadata
        .get("session_projection")
        .and_then(|v| v.get("extra_trace_filters"))
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str().map(ToOwned::to_owned))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use openfang_types::message::Role;

    #[test]
    fn classify_dialogue_message() {
        let msg = Message::assistant("hello");
        assert_eq!(classify_message_lane(&msg), MessageLane::Dialogue);
    }

    #[test]
    fn classify_execution_message() {
        let msg = Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "fake_tool".to_string(),
                input: serde_json::json!({"x": 1}),
            }]),
        };
        assert_eq!(classify_message_lane(&msg), MessageLane::Execution);
    }

    #[test]
    fn project_drops_tool_blocks_keeps_text() {
        let messages = vec![
            Message::user("hello"),
            Message {
                role: Role::Assistant,
                content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "some_tool".to_string(),
                    input: serde_json::json!({}),
                }]),
            },
            Message::assistant("world"),
        ];

        let projected = project_for_persistent_dialogue(&messages, &[]);
        assert_eq!(projected.len(), 2);
        assert_eq!(projected[0].content.text_content(), "hello");
        assert_eq!(projected[1].content.text_content(), "world");
    }

    #[test]
    fn project_drops_default_trace_patterns() {
        let messages = vec![
            Message::assistant("OK. NO_REPLY"),
            Message::assistant("[task completed successfully]"),
            Message::assistant("real dialogue"),
        ];

        let projected = project_for_persistent_dialogue(&messages, &[]);
        assert_eq!(projected.len(), 1);
        assert_eq!(projected[0].content.text_content(), "real dialogue");
    }

    #[test]
    fn project_uses_extra_trace_filters() {
        let messages = vec![
            Message::assistant("给公子回了消息"),
            Message::assistant("real dialogue"),
        ];

        let extra = vec!["给公子回了消息".to_string()];
        let projected = project_for_persistent_dialogue(&messages, &extra);
        assert_eq!(projected.len(), 1);
        assert_eq!(projected[0].content.text_content(), "real dialogue");
    }

    #[test]
    fn project_without_extra_filters_keeps_business_text() {
        let messages = vec![Message::assistant("给公子回了消息")];

        let projected = project_for_persistent_dialogue(&messages, &[]);
        assert_eq!(projected.len(), 1); // kept because no extra filter
    }
}
