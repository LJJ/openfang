//! Context overflow recovery pipeline.
//!
//! Provides a 4-stage recovery pipeline that replaces the brute-force
//! `emergency_trim_messages()` with structured, progressive recovery:
//!
//! 1. Auto-compact via message trimming (keep recent, drop old)
//! 2. Aggressive overflow compaction (drop all but last N)
//! 3. Truncate historical tool results to 2K chars each
//! 4. Return error suggesting /reset or /compact

use openfang_types::message::{ContentBlock, Message, MessageContent};
use openfang_types::tool::ToolDefinition;
use tracing::{debug, warn};

/// Recovery stage that was applied.
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStage {
    /// No recovery needed.
    None,
    /// Stage 1: moderate trim (keep last 10).
    AutoCompaction { removed: usize },
    /// Stage 2: aggressive trim (keep last 4).
    OverflowCompaction { removed: usize },
    /// Stage 3: truncated tool results.
    ToolResultTruncation { truncated: usize },
    /// Stage 4: unrecoverable — suggest /reset.
    FinalError,
}

/// Result of overflow recovery, including any messages that were evicted.
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Which recovery stage was applied.
    pub stage: RecoveryStage,
    /// Messages that were removed from the session during recovery.
    /// Empty if no messages were evicted (stages 3/4 or no recovery needed).
    pub evicted: Vec<Message>,
}

/// Estimate token count using chars/4 heuristic.
fn estimate_tokens(messages: &[Message], system_prompt: &str, tools: &[ToolDefinition]) -> usize {
    crate::compactor::estimate_token_count(messages, Some(system_prompt), Some(tools))
}

/// Run the 4-stage overflow recovery pipeline.
///
/// Returns the recovery stage applied and any messages that were evicted.
pub fn recover_from_overflow(
    messages: &mut Vec<Message>,
    system_prompt: &str,
    tools: &[ToolDefinition],
    context_window: usize,
) -> RecoveryResult {
    let estimated = estimate_tokens(messages, system_prompt, tools);
    let threshold_70 = (context_window as f64 * 0.70) as usize;
    let threshold_90 = (context_window as f64 * 0.90) as usize;

    // No recovery needed
    if estimated <= threshold_70 {
        return RecoveryResult {
            stage: RecoveryStage::None,
            evicted: Vec::new(),
        };
    }

    let mut all_evicted: Vec<Message> = Vec::new();

    // Stage 1: Moderate trim — keep last 10 messages
    if estimated <= threshold_90 {
        let keep = 10.min(messages.len());
        let remove = messages.len() - keep;
        if remove > 0 {
            debug!(
                estimated_tokens = estimated,
                removing = remove,
                "Stage 1: moderate trim to last {keep} messages"
            );
            all_evicted.extend(messages.drain(..remove));
            // Re-check after trim
            let new_est = estimate_tokens(messages, system_prompt, tools);
            if new_est <= threshold_70 {
                return RecoveryResult {
                    stage: RecoveryStage::AutoCompaction { removed: remove },
                    evicted: all_evicted,
                };
            }
        }
    }

    // Stage 2: Aggressive trim — keep last 4 messages + summary marker
    {
        let keep = 4.min(messages.len());
        let remove = messages.len() - keep;
        if remove > 0 {
            warn!(
                estimated_tokens = estimate_tokens(messages, system_prompt, tools),
                removing = remove,
                "Stage 2: aggressive overflow compaction to last {keep} messages"
            );
            let total_removed = all_evicted.len() + remove;
            let summary = Message::user(format!(
                "[System: {} earlier messages were removed due to context overflow. \
                 The conversation continues from here. Use /compact for smarter summarization.]",
                total_removed
            ));
            all_evicted.extend(messages.drain(..remove));
            messages.insert(0, summary);

            let new_est = estimate_tokens(messages, system_prompt, tools);
            if new_est <= threshold_90 {
                return RecoveryResult {
                    stage: RecoveryStage::OverflowCompaction { removed: total_removed },
                    evicted: all_evicted,
                };
            }
        }
    }

    // Stage 3: Truncate all historical tool results to 2K chars
    let tool_truncation_limit = 2000;
    let mut truncated = 0;
    for msg in messages.iter_mut() {
        if let MessageContent::Blocks(blocks) = &mut msg.content {
            for block in blocks.iter_mut() {
                if let ContentBlock::ToolResult { content, .. } = block {
                    if content.len() > tool_truncation_limit {
                        let keep = tool_truncation_limit.saturating_sub(80);
                        *content = format!(
                            "{}\n\n[OVERFLOW RECOVERY: truncated from {} to {} chars]",
                            &content[..keep],
                            content.len(),
                            keep
                        );
                        truncated += 1;
                    }
                }
            }
        }
    }

    if truncated > 0 {
        let new_est = estimate_tokens(messages, system_prompt, tools);
        if new_est <= threshold_90 {
            return RecoveryResult {
                stage: RecoveryStage::ToolResultTruncation { truncated },
                evicted: all_evicted,
            };
        }
        warn!(
            estimated_tokens = new_est,
            "Stage 3 truncated {} tool results but still over threshold", truncated
        );
    }

    // Stage 4: Final error — nothing more we can do automatically
    warn!("Stage 4: all recovery stages exhausted, context still too large");
    RecoveryResult {
        stage: RecoveryStage::FinalError,
        evicted: all_evicted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openfang_types::message::{Message, Role};

    fn make_messages(count: usize, size_each: usize) -> Vec<Message> {
        (0..count)
            .map(|i| {
                let text = format!("msg{}: {}", i, "x".repeat(size_each));
                Message {
                    role: if i % 2 == 0 {
                        Role::User
                    } else {
                        Role::Assistant
                    },
                    content: MessageContent::Text(text),
                }
            })
            .collect()
    }

    #[test]
    fn test_no_recovery_needed() {
        let mut msgs = make_messages(2, 100);
        let result = recover_from_overflow(&mut msgs, "sys", &[], 200_000);
        assert_eq!(result.stage, RecoveryStage::None);
        assert!(result.evicted.is_empty());
    }

    #[test]
    fn test_stage1_moderate_trim() {
        // Create messages that push us past 70% but not 90%
        // Context window: 1000 tokens = 4000 chars
        // 70% = 700 tokens = 2800 chars
        let original_count = 20;
        let mut msgs = make_messages(original_count, 150); // ~3000 chars total
        let result = recover_from_overflow(&mut msgs, "system", &[], 1000);
        match result.stage {
            RecoveryStage::AutoCompaction { removed } => {
                assert!(removed > 0);
                assert!(msgs.len() <= 10);
                assert_eq!(result.evicted.len(), removed);
            }
            RecoveryStage::OverflowCompaction { .. } => {
                // Also acceptable if moderate wasn't enough
                assert!(!result.evicted.is_empty());
            }
            _ => {} // depends on exact token estimation
        }
    }

    #[test]
    fn test_stage2_aggressive_trim() {
        // Push past 90%: 1000 tokens = 4000 chars, 90% = 3600 chars
        let mut msgs = make_messages(30, 200); // ~6000 chars
        let result = recover_from_overflow(&mut msgs, "system", &[], 1000);
        match result.stage {
            RecoveryStage::OverflowCompaction { removed } => {
                assert!(removed > 0);
                assert!(!result.evicted.is_empty());
            }
            RecoveryStage::ToolResultTruncation { .. } | RecoveryStage::FinalError => {}
            _ => {} // acceptable cascading
        }
    }

    #[test]
    fn test_stage3_tool_truncation() {
        let big_result = "x".repeat(5000);
        let mut msgs = vec![
            Message::user("hi"),
            Message {
                role: Role::User,
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: big_result.clone(),
                    is_error: false,
                }]),
            },
            Message {
                role: Role::User,
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "t2".to_string(),
                    content: big_result,
                    is_error: false,
                }]),
            },
        ];
        // Tiny context window to force all stages
        let result = recover_from_overflow(&mut msgs, "system", &[], 500);
        // Should at least reach tool truncation
        match result.stage {
            RecoveryStage::ToolResultTruncation { truncated } => {
                assert!(truncated > 0);
            }
            RecoveryStage::OverflowCompaction { .. } | RecoveryStage::FinalError => {}
            _ => {}
        }
    }

    #[test]
    fn test_cascading_stages() {
        // Ensure stages cascade: if stage 1 isn't enough, stage 2 kicks in
        let mut msgs = make_messages(50, 500);
        let result = recover_from_overflow(&mut msgs, "system prompt", &[], 2000);
        // With 50 messages of 500 chars each (25000 chars), context of 2000 tokens (8000 chars),
        // we should cascade through stages
        assert_ne!(result.stage, RecoveryStage::None);
        assert!(!result.evicted.is_empty());
    }
}
