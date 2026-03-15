//! Session projections separate dialogue memory from execution trace.
//!
//! A single runtime turn may need rich tool-use / tool-result history to
//! complete a complex task, but that same raw execution trace should not be
//! carried into future turns as durable dialogue context.

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

fn is_internal_trace_text(role: Role, text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    let lower = trimmed.to_lowercase();
    let exact_tokens = ["no_reply", "[no reply needed]", "ok. no_reply"];
    if role == Role::Assistant && exact_tokens.iter().any(|token| lower == *token) {
        return true;
    }

    let assistant_prefixes = [
        "[task completed",
        "[the model returned an empty response",
        "[no response]",
        "看起来 turn script 文件路径不对",
        "让我尝试读取环境变量来获取正确的路径",
        "既然 `pending.json` 为空",
        "根据职责 a 第 8 条",
        "因此，我将输出 no_reply",
        "给公子回了消息",
        "给他回了消息",
        "给公子发了消息",
    ];

    let user_prefixes = ["[system] please output your turn script directly"];

    match role {
        Role::Assistant => {
            assistant_prefixes
                .iter()
                .any(|prefix| lower.starts_with(prefix))
                || (lower.starts_with("拍了张") && (lower.contains("发给公子") || lower.contains("发给他")))
        }
        Role::User => user_prefixes.iter().any(|prefix| lower.starts_with(prefix)),
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

fn strip_execution_blocks(role: Role, content: &MessageContent) -> Option<MessageContent> {
    match content {
        MessageContent::Text(text) => {
            if is_internal_trace_text(role, text) {
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
                        if is_internal_trace_text(role, text) {
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
pub fn project_for_persistent_dialogue(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .filter_map(|msg| {
            strip_execution_blocks(msg.role, &msg.content).map(|content| Message {
                role: msg.role,
                content,
            })
        })
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
    fn project_for_persistent_dialogue_drops_tool_blocks_but_keeps_text() {
        let messages = vec![
            Message::user("你已经放进衣橱了吗？"),
            Message {
                role: Role::Assistant,
                content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "mcp_toolbox_add_to_wardrobe".to_string(),
                    input: serde_json::json!({"path": "/tmp/demo.png"}),
                }]),
            },
            Message {
                role: Role::User,
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: "{\"ok\":true}".to_string(),
                    is_error: false,
                }]),
            },
            Message::assistant("已经放进去了。"),
        ];

        let projected = project_for_persistent_dialogue(&messages);
        assert_eq!(projected.len(), 2);
        assert_eq!(projected[0].content.text_content(), "你已经放进衣橱了吗？");
        assert_eq!(projected[1].content.text_content(), "已经放进去了。");
    }

    #[test]
    fn project_for_persistent_dialogue_drops_images() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::Image {
                media_type: "image/jpeg".to_string(),
                data: "abc".to_string(),
            }]),
        }];

        let projected = project_for_persistent_dialogue(&messages);
        assert!(projected.is_empty());
    }

    #[test]
    fn project_for_persistent_dialogue_drops_internal_trace_text() {
        let messages = vec![
            Message::assistant("OK. NO_REPLY"),
            Message::assistant("给公子回了消息，问他怎么了。"),
            Message::user("拍了张照片给你看"),
            Message::assistant("正常回复"),
        ];

        let projected = project_for_persistent_dialogue(&messages);
        assert_eq!(projected.len(), 2);
        assert_eq!(projected[0].content.text_content(), "拍了张照片给你看");
        assert_eq!(projected[1].content.text_content(), "正常回复");
    }
}
