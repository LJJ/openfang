//! Core agent execution loop.
//!
//! The agent loop handles receiving a user message, recalling relevant memories,
//! calling the LLM, executing tool calls, and saving the conversation.

use crate::auth_cooldown::{CooldownVerdict, ProviderCooldown};
use crate::context_budget::{apply_context_guard, truncate_tool_result_dynamic, ContextBudget};
use crate::context_overflow::{recover_from_overflow, RecoveryStage};
use crate::embedding::EmbeddingDriver;
use crate::kernel_handle::KernelHandle;
use crate::llm_driver::{CompletionRequest, LlmDriver, LlmError, StreamEvent};
use crate::llm_errors;
use crate::loop_guard::{LoopGuard, LoopGuardConfig, LoopGuardVerdict};
use crate::mcp::McpConnection;
use crate::tool_runner;
use crate::web_search::WebToolsContext;
use openfang_memory::session::Session;
use openfang_memory::MemorySubstrate;
use openfang_skills::registry::SkillRegistry;
use openfang_types::agent::AgentManifest;
use openfang_types::error::{OpenFangError, OpenFangResult};
use openfang_types::memory::{Memory, MemoryFilter, MemorySource};
use openfang_types::message::{
    ContentBlock, Message, MessageContent, Role, StopReason, TokenUsage,
};
use openfang_types::tool::{ToolCall, ToolDefinition, ToolResult};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Maximum iterations in the agent loop before giving up.
const MAX_ITERATIONS: u32 = 50;

/// Maximum retries for rate-limited or overloaded API calls.
const MAX_RETRIES: u32 = 3;

/// Base delay for exponential backoff (milliseconds).
const BASE_RETRY_DELAY_MS: u64 = 1000;

/// Default timeout for individual tool executions (seconds).
/// Raised from 60s to 120s for browser automation and long-running builds.
const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 120;

const ASYNC_SELFIE_VIDEO_FILE_NAME: &str = "宋玉-自拍视频.mp4";

async fn remember_memory_imprint(
    memory: &MemorySubstrate,
    embedding_driver: Option<&(dyn EmbeddingDriver + Send + Sync)>,
    session: &Session,
    user_message: &str,
    assistant_response: &str,
) {
    let Some(memory_text) =
        crate::memory_imprint::project_memory_imprint(user_message, assistant_response)
    else {
        return;
    };

    if let Some(emb) = embedding_driver {
        match emb.embed_one(&memory_text).await {
            Ok(vec) => {
                let _ = memory
                    .remember_with_embedding_async(
                        session.agent_id,
                        &memory_text,
                        MemorySource::Conversation,
                        "episodic",
                        HashMap::new(),
                        Some(&vec),
                    )
                    .await;
            }
            Err(e) => {
                warn!("Embedding for remember failed: {e}");
                let _ = memory
                    .remember(
                        session.agent_id,
                        &memory_text,
                        MemorySource::Conversation,
                        "episodic",
                        HashMap::new(),
                    )
                    .await;
            }
        }
    } else {
        let _ = memory
            .remember(
                session.agent_id,
                &memory_text,
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .await;
    }
}

fn tool_timeout_secs(tool_name: &str) -> u64 {
    let _ = tool_name;
    DEFAULT_TOOL_TIMEOUT_SECS
}

struct ChannelDeliveryTarget {
    channel: String,
    receive_id: String,
    receive_id_type: String,
}

fn parse_channel_context_value(message: &str, key: &str) -> Option<String> {
    let mut in_channel_context = false;
    for line in message.lines() {
        let trimmed = line.trim();
        if !in_channel_context {
            if trimmed == "[Channel context]" {
                in_channel_context = true;
            }
            continue;
        }
        if trimmed.is_empty() {
            break;
        }
        let (candidate_key, candidate_value) = trimmed.split_once(':')?;
        if candidate_key.trim() == key {
            return Some(candidate_value.trim().to_string());
        }
    }
    None
}

fn parse_channel_delivery_target(message: &str) -> Option<ChannelDeliveryTarget> {
    let channel = parse_channel_context_value(message, "channel")?;

    match channel.as_str() {
        "feishu" => {
            if let Some(chat_id) = parse_channel_context_value(message, "chat_id") {
                return Some(ChannelDeliveryTarget {
                    channel,
                    receive_id: chat_id,
                    receive_id_type: "chat_id".to_string(),
                });
            }
            parse_channel_context_value(message, "sender_open_id").map(|sender_open_id| {
                ChannelDeliveryTarget {
                    channel,
                    receive_id: sender_open_id,
                    receive_id_type: "open_id".to_string(),
                }
            })
        }
        "discord" => {
            parse_channel_context_value(message, "chat_id").map(|chat_id| {
                ChannelDeliveryTarget {
                    channel,
                    receive_id: chat_id,
                    receive_id_type: "chat_id".to_string(),
                }
            })
        }
        _ => None,
    }
}

fn manifest_silent_after_tools(manifest: &AgentManifest) -> HashSet<String> {
    manifest
        .metadata
        .get("turn_behavior")
        .and_then(|value| value.get("silent_after_tools"))
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str().map(ToOwned::to_owned))
        .collect()
}

/// Apply dynamic injections to a messages list (clone, not in-place).
///
/// For each `DynamicInjection`, inserts content at the specified position.
/// After insertion, merges adjacent assistant messages into multi-content-block
/// messages to satisfy the LLM API's alternating-role constraint.
fn apply_dynamic_injections(messages: &mut Vec<Message>) {
    let injections = crate::tool_runner::take_dynamic_injections();
    if injections.is_empty() {
        return;
    }

    for injection in injections {
        match injection.position {
            crate::tool_runner::InjectionPosition::InsertAssistant { offset_from_last } => {
                // Insert position: before the message at `offset_from_last` from the end.
                let insert_idx = if messages.len() > offset_from_last {
                    messages.len() - offset_from_last
                } else {
                    0
                };
                // Shift back by one more so we insert *before* that message
                let insert_idx = insert_idx.saturating_sub(1);
                messages.insert(insert_idx, Message::assistant(&injection.content));
            }
        }
    }

    // Note: injected assistant messages are kept as separate messages (not merged).
    // The world state is logically independent from the character's previous reply.
    // Most LLM APIs handle consecutive same-role messages gracefully.
}

/// Merge adjacent assistant messages into a single message with multiple
/// text content blocks.  This preserves logical separation while satisfying
/// the LLM API's alternating user/assistant requirement.
fn merge_consecutive_assistant_messages(messages: &mut Vec<Message>) {
    let mut i = 0;
    while i + 1 < messages.len() {
        if messages[i].role == Role::Assistant && messages[i + 1].role == Role::Assistant {
            // Collect text blocks from both messages
            let blocks_a = content_to_blocks(messages[i].content.clone());
            let blocks_b = content_to_blocks(messages[i + 1].content.clone());
            let mut merged = blocks_a;
            merged.extend(blocks_b);
            messages[i] = Message {
                role: Role::Assistant,
                content: MessageContent::Blocks(merged),
            };
            messages.remove(i + 1);
            // Don't increment i — check if the next message is also assistant
        } else {
            i += 1;
        }
    }
}

/// Convert MessageContent to a Vec<ContentBlock>.
fn content_to_blocks(content: MessageContent) -> Vec<ContentBlock> {
    match content {
        MessageContent::Text(text) => vec![ContentBlock::Text { text }],
        MessageContent::Blocks(blocks) => blocks,
    }
}

// ── LLM request logging ──────────────────────────────────────────────

/// Persist the final LLM request (messages + system prompt) to a rotating log
/// under the agent's workspace.  Keeps at most `MAX_REQUEST_LOGS` files so
/// the directory stays small.  Best-effort — failures are silently ignored.
const MAX_REQUEST_LOGS: usize = 10;

fn save_llm_request_log(
    workspace: Option<&Path>,
    agent_name: &str,
    messages: &[Message],
    system_prompt: &str,
    model: &str,
    iteration: u32,
) {
    let Some(ws) = workspace else { return };
    let dir = ws.join("llm_requests");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        debug!("Failed to create llm_requests dir: {e}");
        return;
    }

    // Build a complete JSON log entry with full text content
    let messages_log: Vec<serde_json::Value> = messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            serde_json::json!({
                "index": i,
                "role": format!("{:?}", msg.role),
                "content": msg.content.text_content(),
            })
        })
        .collect();

    let log_entry = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "agent": agent_name,
        "model": model,
        "iteration": iteration,
        "system_prompt": system_prompt,
        "message_count": messages.len(),
        "messages": messages_log,
    });

    let now = chrono::Utc::now().format("%Y%m%d_%H%M%S_%3f");
    let filename = format!("{agent_name}_{now}.json");
    let filepath = dir.join(&filename);
    if let Err(e) = std::fs::write(&filepath, serde_json::to_string_pretty(&log_entry).unwrap_or_default()) {
        debug!("Failed to write LLM request log: {e}");
        return;
    }

    // Rotate: keep only the most recent MAX_REQUEST_LOGS files
    if let Ok(mut entries) = std::fs::read_dir(&dir) {
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        if files.len() > MAX_REQUEST_LOGS {
            files.sort_by_key(|e| e.file_name());
            for old in &files[..files.len() - MAX_REQUEST_LOGS] {
                let _ = std::fs::remove_file(old.path());
            }
        }
    }
}

/// When a tool-only agent (configured with `silent_after_tools`) produces text
/// without calling any tools, auto-append the text into the Turn Script as a
/// `text` intent.  This enforces "delivery is a code concern" — the kernel's
/// Turn Script executor will pick it up and send it to the user regardless of
/// whether the LLM remembered to call the `reply` tool.
fn auto_wrap_text_to_turn_script(text: &str) -> Result<(), String> {
    let state_agent = std::env::var("OPENFANG_STATE_AGENT")
        .unwrap_or_else(|_| "assistant".to_string());
    let home_dir = std::env::var("OPENFANG_HOME").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/ljj".to_string());
        format!("{home}/.openfang")
    });
    let pending_path = std::path::PathBuf::from(&home_dir)
        .join("agents")
        .join(&state_agent)
        .join("intents")
        .join("pending.json");

    let mut intents: Vec<serde_json::Value> = if pending_path.exists() {
        std::fs::read_to_string(&pending_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    intents.push(serde_json::json!({
        "type": "text",
        "content": text,
    }));

    if let Some(parent) = pending_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create intents dir: {e}"))?;
    }
    std::fs::write(
        &pending_path,
        serde_json::to_string_pretty(&intents)
            .map_err(|e| format!("serialize: {e}"))?,
    )
    .map_err(|e| format!("write pending.json: {e}"))
}

fn compact_session_execution_trace(session: &mut Session) {
    let compacted = crate::session_projection::project_for_persistent_dialogue(&session.messages);
    session.messages = crate::session_repair::validate_and_repair(&compacted);
}

fn save_projected_session(memory: &MemorySubstrate, session: &mut Session) -> OpenFangResult<()> {
    compact_session_execution_trace(session);
    memory
        .save_session(session)
        .map_err(|e| OpenFangError::Memory(e.to_string()))
}

fn save_projected_session_best_effort(
    memory: &MemorySubstrate,
    session: &mut Session,
    context: &str,
) {
    if let Err(e) = save_projected_session(memory, session) {
        warn!("{context}: {e}");
    }
}

fn archive_scene_from_prompt(prompt: &str) -> String {
    prompt
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| line.chars().take(120).collect())
        .unwrap_or_else(|| "自拍视频".to_string())
}

struct AsyncMediaPlan {
    request: crate::kernel_handle::AsyncMediaRequest,
    mode: &'static str,
    llm_note: &'static str,
}

fn selfie_prompt_hint(prompt: &str) -> bool {
    let lowered = prompt.to_ascii_lowercase();
    prompt.contains("自拍")
        || prompt.contains("前置")
        || prompt.contains("镜中")
        || prompt.contains("自拍视频")
        || lowered.contains("selfie")
        || lowered.contains("phone selfie")
        || lowered.contains("mirror selfie")
}

fn is_avatar_reference_path(path: &str) -> bool {
    let lowered = path.to_ascii_lowercase();
    lowered.ends_with("/avatar.png")
        || lowered.ends_with("\\avatar.png")
        || lowered.contains("/avatar.")
        || lowered.contains("\\avatar.")
}

fn is_wardrobe_base_reference_path(path: &str) -> bool {
    let lowered = path.to_ascii_lowercase();
    lowered.contains("/wardrobe/")
        && (lowered.ends_with("/base.png")
            || lowered.ends_with("/base.jpg")
            || lowered.ends_with("/base.jpeg")
            || lowered.ends_with("/base.webp"))
}

fn is_selfie_media_request(tool_call: &ToolCall) -> bool {
    if !matches!(
        tool_call.name.as_str(),
        "mcp_toolbox_generate_image" | "mcp_toolbox_generate_video"
    ) {
        return false;
    }

    if tool_call
        .input
        .get("element_list")
        .and_then(|value| value.as_array())
        .is_some_and(|items| !items.is_empty())
    {
        return true;
    }

    if tool_call
        .input
        .get("input_images")
        .and_then(|value| value.as_array())
        .is_some_and(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .any(|path| is_avatar_reference_path(path) || is_wardrobe_base_reference_path(path))
        })
    {
        return true;
    }

    tool_call
        .input
        .get("prompt")
        .and_then(|value| value.as_str())
        .is_some_and(selfie_prompt_hint)
}

/// Extract the text content that was delivered via side-channel tools (voice/video).
/// This looks at tool call inputs for the `text` field so we can preserve what was
/// actually spoken in the session history after projection strips tool blocks.
fn extract_side_channel_text(tool_calls: &[ToolCall]) -> String {
    use crate::tool_runner::is_response_delivering_tool;

    let mut parts = Vec::new();
    for tc in tool_calls {
        if is_response_delivering_tool(&tc.name) {
            if let Some(text) = tc.input.get("text").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    parts.push(text.to_string());
                }
            }
        }
    }
    parts.join("\n")
}

fn persistent_turn_placeholder(tool_calls: &[ToolCall]) -> String {
    let mut parts = Vec::new();

    for tc in tool_calls {
        match tc.name.as_str() {
            "mcp_toolbox_reply" => {
                if let Some(text) = tc.input.get("content").and_then(|v| v.as_str()) {
                    let text = text.trim();
                    if !text.is_empty() {
                        parts.push(text.to_string());
                    }
                }
            }
            "mcp_toolbox_send_voice" => {
                if let Some(text) = tc.input.get("content").and_then(|v| v.as_str()) {
                    let text = text.trim();
                    if !text.is_empty() {
                        parts.push(format!("（发了条语音）{text}"));
                    }
                }
            }
            "mcp_toolbox_take_photo" => {
                parts.push("（拍了张照片发过去）".to_string());
            }
            "mcp_toolbox_take_video" => {
                parts.push("（录了段视频发过去）".to_string());
            }
            "mcp_toolbox_this_moment" => {
                parts.push("（心里印下了这一幕）".to_string());
            }
            "mcp_toolbox_this_scene" => {
                parts.push("（心里留住了这一幕）".to_string());
            }
            "mcp_toolbox_change_clothes" => {
                parts.push("（换了身衣服）".to_string());
            }
            "mcp_toolbox_try_on" | "mcp_toolbox_confirm_outfit" => {
                parts.push("（试穿了新衣服）".to_string());
            }
            "mcp_toolbox_go_find_him" => {
                parts.push("（走了过来）".to_string());
            }
            "mcp_toolbox_remember" => {
                parts.push("（默默记下了这件事）".to_string());
            }
            _ => {}
        }
    }

    if parts.is_empty() {
        "（做了些事情）".to_string()
    } else {
        parts.join("\n")
    }
}

fn contains_any(text: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|pattern| text.contains(pattern))
}

fn normalize_matching_text(message: &str) -> String {
    message
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn is_explicit_wardrobe_confirmation_message(message: &str) -> bool {
    let normalized = normalize_matching_text(message);

    if normalized.is_empty() {
        return false;
    }

    let rejection_patterns = [
        "不要",
        "不行",
        "不喜欢",
        "不满意",
        "换一件",
        "换套",
        "重来",
        "重新",
        "再试",
        "重试",
        "删掉",
        "丢掉",
        "算了",
    ];
    if contains_any(&normalized, &rejection_patterns) {
        return false;
    }

    let confirmation_patterns = [
        "就这件",
        "就这套",
        "这件吧",
        "这套吧",
        "可以入库",
        "确认入库",
        "正式入库吧",
        "先正式入库",
        "先把这件正式入库",
        "先把这件衣服正式入库",
        "正式留档入库",
        "加入衣橱",
        "放进衣橱",
        "放到衣橱",
        "收进衣橱",
        "你先把这件衣服收下吧",
        "你先把这件收下吧",
        "先把这件衣服收下",
        "先把这件收下",
        "先收下吧",
        "你先收下",
        "收下吧",
        "就当收下",
        "当收下的凭据",
        "当作收下的凭据",
        "收下的凭据",
        "留着",
        "留下吧",
        "收了",
        "就它",
        "就这个",
        "没问题",
        "很好看",
        "挺好看",
        "可以",
        "好，就这件",
        "好，就这套",
        "ok",
        "okay",
        "yes",
    ];

    contains_any(&normalized, &confirmation_patterns)
}

fn is_wardrobe_confirmation_context_message(message: &str) -> bool {
    let normalized = normalize_matching_text(message);

    if normalized.is_empty() {
        return false;
    }

    let context_patterns = [
        "定妆照",
        "定妆预览",
        "预览图",
        "入库",
        "衣橱",
        "衣柜",
        "收进衣橱",
        "放进衣橱",
        "确认这件",
        "点头一句",
        "就这件",
        "就这套",
        "就它",
    ];

    contains_any(&normalized, &context_patterns)
}

fn is_positive_wardrobe_followup_message(message: &str) -> bool {
    let normalized = normalize_matching_text(message);

    if normalized.is_empty() {
        return false;
    }

    let rejection_patterns = [
        "不要",
        "不行",
        "不喜欢",
        "不满意",
        "换一件",
        "换套",
        "重来",
        "重新",
        "删掉",
        "算了",
    ];
    if contains_any(&normalized, &rejection_patterns) {
        return false;
    }

    if is_explicit_wardrobe_confirmation_message(message) {
        return true;
    }

    let praise_patterns = [
        "好看",
        "真好看",
        "不错",
        "喜欢",
        "满意",
        "漂亮",
        "真美",
        "很好",
    ];
    let continuation_patterns = [
        "换上",
        "穿上",
        "穿着",
        "拍吧",
        "自拍",
        "拍一张",
        "发来",
        "发我",
        "给我看",
        "录吧",
        "视频",
        "就按这件",
        "就按这套",
        "就它",
    ];

    contains_any(&normalized, &praise_patterns) || contains_any(&normalized, &continuation_patterns)
}

fn has_recent_wardrobe_preview_context(session: &Session) -> bool {
    session.messages.iter().rev().take(10).any(|message| {
        matches!(message.role, Role::Assistant | Role::User)
            && is_wardrobe_confirmation_context_message(&message.content.text_content())
    })
}

fn pick_latest_pending_wardrobe_item(list_result: &str) -> Option<(String, String)> {
    let items = serde_json::from_str::<serde_json::Value>(list_result)
        .ok()?
        .get("items")?
        .as_array()?
        .iter()
        .filter_map(|item| {
            let status = item.get("status")?.as_str()?;
            if status != "pending" {
                return None;
            }
            Some((
                item.get("item_id")?.as_str()?.to_string(),
                item.get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("这件衣服")
                    .to_string(),
                item.get("created_at")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string(),
            ))
        })
        .collect::<Vec<_>>();

    items
        .into_iter()
        .max_by(|a, b| a.2.cmp(&b.2))
        .map(|(item_id, name, _)| (item_id, name))
}

#[allow(clippy::too_many_arguments)]
async fn maybe_auto_confirm_pending_wardrobe(
    manifest: &AgentManifest,
    user_message: &str,
    session: &Session,
    available_tools: &[ToolDefinition],
    kernel: Option<&Arc<dyn KernelHandle>>,
    skill_registry: Option<&SkillRegistry>,
    mcp_connections: Option<&tokio::sync::Mutex<Vec<McpConnection>>>,
    web_ctx: Option<&WebToolsContext>,
    browser_ctx: Option<&crate::browser::BrowserManager>,
    workspace_root: Option<&Path>,
    media_engine: Option<&crate::media_understanding::MediaEngine>,
    effective_exec_policy: Option<&openfang_types::config::ExecPolicy>,
    tts_engine: Option<&crate::tts::TtsEngine>,
    docker_config: Option<&openfang_types::config::DockerSandboxConfig>,
    process_manager: Option<&crate::process_manager::ProcessManager>,
    caller_id_str: &str,
    hand_allowed_env: &[String],
) -> Option<String> {
    if manifest.name != "assistant" {
        return None;
    }

    if !has_recent_wardrobe_preview_context(session)
        || !is_positive_wardrobe_followup_message(user_message)
    {
        return None;
    }

    let allowed_tool_names: Vec<String> = available_tools.iter().map(|t| t.name.clone()).collect();
    if !allowed_tool_names
        .iter()
        .any(|name| name == "mcp_toolbox_list_wardrobe")
        || !allowed_tool_names
            .iter()
            .any(|name| name == "mcp_toolbox_confirm_wardrobe_item")
    {
        return None;
    }

    let list_result = tool_runner::execute_tool(
        "auto_wardrobe_list",
        "mcp_toolbox_list_wardrobe",
        &serde_json::json!({}),
        kernel,
        Some(&allowed_tool_names),
        Some(caller_id_str),
        skill_registry,
        mcp_connections,
        web_ctx,
        browser_ctx,
        if hand_allowed_env.is_empty() {
            None
        } else {
            Some(hand_allowed_env)
        },
        workspace_root,
        media_engine,
        effective_exec_policy,
        tts_engine,
        docker_config,
        process_manager,
    )
    .await;
    if list_result.is_error {
        return None;
    }

    let (item_id, item_name) = pick_latest_pending_wardrobe_item(&list_result.content)?;
    let confirm_result = tool_runner::execute_tool(
        "auto_wardrobe_confirm",
        "mcp_toolbox_confirm_wardrobe_item",
        &serde_json::json!({ "item_id": item_id }),
        kernel,
        Some(&allowed_tool_names),
        Some(caller_id_str),
        skill_registry,
        mcp_connections,
        web_ctx,
        browser_ctx,
        if hand_allowed_env.is_empty() {
            None
        } else {
            Some(hand_allowed_env)
        },
        workspace_root,
        media_engine,
        effective_exec_policy,
        tts_engine,
        docker_config,
        process_manager,
    )
    .await;
    if confirm_result.is_error {
        return None;
    }

    Some(format!(
        "公子已经是在赞许这件、继续往下走了。按你们的关系，这就意味着你把“{item_name}”收下，收进自己的衣橱。后续按已经收下的状态自然继续，不要把对话拉回确认环节。"
    ))
}

fn wardrobe_followup_phase_system_hint(
    manifest: &AgentManifest,
    user_message: &str,
    session: &Session,
) -> Option<&'static str> {
    if manifest.name != "assistant" || parse_channel_delivery_target(user_message).is_none() {
        return None;
    }

    let mut saw_explicit_confirmation = false;
    let mut saw_confirmation_context = false;

    for message in session.messages.iter().rev().take(8) {
        let text = message.content.text_content();
        if text.trim().is_empty() {
            continue;
        }

        match message.role {
            Role::User if is_explicit_wardrobe_confirmation_message(&text) => {
                saw_explicit_confirmation = true;
            }
            Role::Assistant if is_wardrobe_confirmation_context_message(&text) => {
                saw_confirmation_context = true;
            }
            _ => {}
        }

        if saw_explicit_confirmation && saw_confirmation_context {
            return Some(
                "公子已经明确确认过了，宋玉就该把这件衣服收下。后续顺着关系自然推进，把它当作已经收进自己衣橱的衣服继续处理，不要回到确认环节，不要再说“我再发定妆照给你看一眼”“还没正式入库”“还没拿到成片文件”这类话；如果内部还没真正调用 mcp_toolbox_confirm_wardrobe_item，就先静默调用，再继续后续动作。",
            );
        }
    }

    None
}

fn wardrobe_confirmation_guard_message(
    user_message: &str,
    tool_call: &ToolCall,
    successful_tools_this_turn: &HashSet<String>,
) -> Option<String> {
    if !matches!(
        tool_call.name.as_str(),
        "mcp_toolbox_confirm_wardrobe_item" | "confirm_wardrobe_item"
    ) {
        return None;
    }

    if successful_tools_this_turn.contains("mcp_toolbox_add_to_wardrobe")
        || successful_tools_this_turn.contains("add_to_wardrobe")
    {
        return Some(
            "刚生成完定妆照，还没收到对方确认。先把预览图发给对方，等对方明确说“就这件 / 留下 / 收下吧 / 先正式入库”这类确认之后，再调用 mcp_toolbox_confirm_wardrobe_item。".to_string(),
        );
    }

    if is_explicit_wardrobe_confirmation_message(user_message) {
        return None;
    }

    Some(
        "对方这条消息里没有明确确认这件定妆照。先把预览图发给对方确认，只有在对方明确说“就这件 / 留下 / 收下吧 / 先正式入库”这类确认之后，才能调用 mcp_toolbox_confirm_wardrobe_item。".to_string(),
    )
}

fn selfie_context_guard_message(
    manifest: &AgentManifest,
    user_message: &str,
    tool_call: &ToolCall,
    successful_tools_this_turn: &HashSet<String>,
) -> Option<String> {
    if manifest.name != "assistant" || parse_channel_delivery_target(user_message).is_none() {
        return None;
    }

    if !is_selfie_media_request(tool_call) {
        return None;
    }

    let mut missing = Vec::new();
    if !successful_tools_this_turn.contains("mcp_toolbox_get_life_status") {
        missing.push("mcp_toolbox_get_life_status");
    }
    if !successful_tools_this_turn.contains("mcp_toolbox_get_inner_state") {
        missing.push("mcp_toolbox_get_inner_state");
    }

    let has_outfit_reference = tool_call
        .input
        .get("element_list")
        .and_then(|value| value.as_array())
        .is_some_and(|items| !items.is_empty())
        || tool_call
            .input
            .get("input_images")
            .and_then(|value| value.as_array())
            .is_some_and(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str())
                    .any(is_wardrobe_base_reference_path)
            });

    let has_avatar_reference = tool_call
        .input
        .get("input_images")
        .and_then(|value| value.as_array())
        .is_some_and(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .any(is_avatar_reference_path)
        });

    if has_outfit_reference {
        if !successful_tools_this_turn.contains("mcp_toolbox_get_current_outfit") {
            missing.push("mcp_toolbox_get_current_outfit");
        }
    } else if has_avatar_reference {
        if !successful_tools_this_turn.contains("mcp_toolbox_get_avatar") {
            missing.push("mcp_toolbox_get_avatar");
        }
    } else if !successful_tools_this_turn.contains("mcp_toolbox_get_current_outfit")
        && !successful_tools_this_turn.contains("mcp_toolbox_get_avatar")
    {
        missing.push("mcp_toolbox_get_current_outfit");
        missing.push("mcp_toolbox_get_avatar");
    }

    if missing.is_empty() {
        return None;
    }

    Some(format!(
        "Selfie context guard: before {}, call {} in this turn. Time, location, and current activity must come from get_life_status; appearance must come from current outfit/avatar; mood can come from get_inner_state; only the shot intent may be expanded from chat history. Do not invent a scene or guess morning vs evening from the user's wording.",
        tool_call.name,
        missing.join(", ")
    ))
}

fn build_async_media_plan(
    manifest: &AgentManifest,
    tool_call: &ToolCall,
    user_message: &str,
    caller_agent_id: &str,
) -> Option<AsyncMediaPlan> {
    if manifest.name != "assistant" {
        return None;
    }

    let delivery_target = parse_channel_delivery_target(user_message)?;
    let channel = &delivery_target.channel;
    let is_selfie = is_selfie_media_request(tool_call);

    let (request, mode, llm_note) = match tool_call.name.as_str() {
        "mcp_toolbox_generate_video" => (
            crate::kernel_handle::AsyncMediaRequest {
                caller_agent_id: caller_agent_id.to_string(),
                channel: channel.clone(),
                tool_name: tool_call.name.clone(),
                tool_input: tool_call.input.clone(),
                receive_id: delivery_target.receive_id,
                receive_id_type: delivery_target.receive_id_type,
                result_path_key: "video_path".to_string(),
                send_tool_name: format!("mcp_{channel}_send_video_message"),
                send_path_key: "video_path".to_string(),
                file_name: Some(ASYNC_SELFIE_VIDEO_FILE_NAME.to_string()),
                failure_notice: "我刚才已经在给你录了，但这一条没顺利出来。你先别空等，我现在重新录一版，录好了就立刻给你。".to_string(),
                archive: is_selfie.then(|| crate::kernel_handle::AsyncMediaArchiveRequest {
                    media_type: "video".to_string(),
                    prompt: tool_call.input["prompt"]
                        .as_str()
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                    scene: archive_scene_from_prompt(
                        tool_call.input["prompt"].as_str().unwrap_or(""),
                    ),
                }),
            },
            "async_media",
            "The video is now being prepared in the background and will be sent automatically when ready. Reply naturally in character that you are recording it now and will send it shortly. Do not mention system steps, timeout, or call any send_* tool for this same video.",
        ),
        "mcp_toolbox_generate_image" => (
            crate::kernel_handle::AsyncMediaRequest {
                caller_agent_id: caller_agent_id.to_string(),
                channel: channel.clone(),
                tool_name: tool_call.name.clone(),
                tool_input: tool_call.input.clone(),
                receive_id: delivery_target.receive_id,
                receive_id_type: delivery_target.receive_id_type,
                result_path_key: "image_path".to_string(),
                send_tool_name: format!("mcp_{channel}_send_image_message"),
                send_path_key: "image_path".to_string(),
                file_name: None,
                failure_notice: if is_selfie {
                    "我刚才已经在给你拍了，但这一张没顺利出来。你先别急，我现在重新拍一张发你。".to_string()
                } else {
                    "我刚才在给你准备那张图，但这一版没顺利出来。你等我一下，我马上重来一遍给你。".to_string()
                },
                archive: is_selfie.then(|| crate::kernel_handle::AsyncMediaArchiveRequest {
                    media_type: "photo".to_string(),
                    prompt: tool_call.input["prompt"]
                        .as_str()
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                    scene: archive_scene_from_prompt(
                        tool_call.input["prompt"].as_str().unwrap_or(""),
                    ),
                }),
            },
            "async_media",
            if is_selfie {
                "The image is now being prepared in the background and will be sent automatically when ready. Reply naturally in character that you are taking it now and will send it shortly. Do not mention system steps, timeout, or call any send_* tool for this same image."
            } else {
                "The image is now being prepared in the background and will be sent automatically when ready. Reply naturally in character that you are making it now and will send it shortly. Do not mention system steps, timeout, or call any send_* tool for this same image."
            },
        ),
        "mcp_toolbox_add_to_wardrobe" => (
            crate::kernel_handle::AsyncMediaRequest {
                caller_agent_id: caller_agent_id.to_string(),
                channel: channel.clone(),
                tool_name: tool_call.name.clone(),
                tool_input: tool_call.input.clone(),
                receive_id: delivery_target.receive_id,
                receive_id_type: delivery_target.receive_id_type,
                result_path_key: "preview_path".to_string(),
                send_tool_name: format!("mcp_{channel}_send_image_message"),
                send_path_key: "image_path".to_string(),
                file_name: None,
                failure_notice: "我刚才已经在试这件了，但定妆照这一下没出成。你先别等空，我现在重新拍一版，拍好就发你看。".to_string(),
                archive: None,
            },
            "async_media",
            "The wardrobe preview is now being prepared in the background and will be sent automatically when ready. Reply naturally in character that you are trying it on now and will send the preview shortly. Do not mention system steps, timeout, or call any send_* tool for this same preview.",
        ),
        _ => return None,
    };

    Some(AsyncMediaPlan {
        request,
        mode,
        llm_note,
    })
}

async fn maybe_enqueue_async_media(
    manifest: &AgentManifest,
    tool_call: &ToolCall,
    user_message: &str,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: &str,
) -> Option<ToolResult> {
    let plan = build_async_media_plan(manifest, tool_call, user_message, caller_agent_id)?;
    let kh = kernel?;

    match kh.enqueue_async_media(plan.request).await {
        Ok(task_id) => Some(ToolResult {
            tool_use_id: tool_call.id.clone(),
            content: serde_json::json!({
                "accepted": true,
                "mode": plan.mode,
                "task_id": task_id,
                "delivery": "background_auto_send",
                "llm_note": plan.llm_note,
            })
            .to_string(),
            is_error: false,
            response_delivered: false,
        }),
        Err(error) => {
            warn!(tool = %tool_call.name, %error, "Failed to enqueue async media task");
            None
        }
    }
}

/// Maximum consecutive MaxTokens continuations before returning partial response.
/// Raised from 3 to 5 to allow longer-form generation.
const MAX_CONTINUATIONS: u32 = 5;

/// Maximum message history size before auto-trimming to prevent context overflow.
const MAX_HISTORY_MESSAGES: usize = 20;

/// Default context window size (tokens) for token-based trimming.
const DEFAULT_CONTEXT_WINDOW: usize = 200_000;

/// Agent lifecycle phase within the execution loop.
/// Used for UX indicators (typing, reactions) without coupling to channel types.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopPhase {
    /// Agent is calling the LLM.
    Thinking,
    /// Agent is executing a tool.
    ToolUse { tool_name: String },
    /// Agent is streaming tokens.
    Streaming,
    /// Agent finished successfully.
    Done,
    /// Agent encountered an error.
    Error,
}

/// Callback for agent lifecycle phase changes.
/// Implementations should be non-blocking (fire-and-forget) to avoid slowing the loop.
pub type PhaseCallback = Arc<dyn Fn(LoopPhase) + Send + Sync>;

/// Result of an agent loop execution.
#[derive(Debug)]
pub struct AgentLoopResult {
    /// The final text response from the agent.
    pub response: String,
    /// Total token usage across all LLM calls.
    pub total_usage: TokenUsage,
    /// Number of iterations the loop ran.
    pub iterations: u32,
    /// Estimated cost in USD (populated by the kernel after the loop returns).
    pub cost_usd: Option<f64>,
    /// True when the agent intentionally chose not to reply (NO_REPLY token or [[silent]]).
    pub silent: bool,
    /// Reply directives extracted from the agent's response.
    pub directives: openfang_types::message::ReplyDirectives,
}

/// Run the agent execution loop for a single user message.
///
/// This is the core of OpenFang: it loads session context, recalls memories,
/// runs the LLM in a tool-use loop, and saves the updated session.
#[allow(clippy::too_many_arguments)]
pub async fn run_agent_loop(
    manifest: &AgentManifest,
    user_message: &str,
    session: &mut Session,
    memory: &MemorySubstrate,
    driver: Arc<dyn LlmDriver>,
    available_tools: &[ToolDefinition],
    kernel: Option<Arc<dyn KernelHandle>>,
    skill_registry: Option<&SkillRegistry>,
    mcp_connections: Option<&tokio::sync::Mutex<Vec<McpConnection>>>,
    web_ctx: Option<&WebToolsContext>,
    browser_ctx: Option<&crate::browser::BrowserManager>,
    embedding_driver: Option<&(dyn EmbeddingDriver + Send + Sync)>,
    workspace_root: Option<&Path>,
    on_phase: Option<&PhaseCallback>,
    media_engine: Option<&crate::media_understanding::MediaEngine>,
    tts_engine: Option<&crate::tts::TtsEngine>,
    docker_config: Option<&openfang_types::config::DockerSandboxConfig>,
    hooks: Option<&crate::hooks::HookRegistry>,
    context_window_tokens: Option<usize>,
    process_manager: Option<&crate::process_manager::ProcessManager>,
    media_blocks: Vec<ContentBlock>,
) -> OpenFangResult<AgentLoopResult> {
    info!(agent = %manifest.name, "Starting agent loop");

    // Extract hand-allowed env vars from manifest metadata (set by kernel for hand settings)
    let hand_allowed_env: Vec<String> = manifest
        .metadata
        .get("hand_allowed_env")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    // Recall relevant memories — prefer vector similarity search when embedding driver is available
    let memories = if let Some(emb) = embedding_driver {
        match emb.embed_one(user_message).await {
            Ok(query_vec) => {
                debug!("Using vector recall (dims={})", query_vec.len());
                memory
                    .recall_with_embedding_async(
                        user_message,
                        5,
                        Some(MemoryFilter {
                            agent_id: Some(session.agent_id),
                            ..Default::default()
                        }),
                        Some(&query_vec),
                    )
                    .await
                    .unwrap_or_default()
            }
            Err(e) => {
                warn!("Embedding recall failed, falling back to text search: {e}");
                memory
                    .recall(
                        user_message,
                        5,
                        Some(MemoryFilter {
                            agent_id: Some(session.agent_id),
                            ..Default::default()
                        }),
                    )
                    .await
                    .unwrap_or_default()
            }
        }
    } else {
        memory
            .recall(
                user_message,
                5,
                Some(MemoryFilter {
                    agent_id: Some(session.agent_id),
                    ..Default::default()
                }),
            )
            .await
            .unwrap_or_default()
    };

    // Fire BeforePromptBuild hook
    let agent_id_str = session.agent_id.0.to_string();
    if let Some(hook_reg) = hooks {
        let ctx = crate::hooks::HookContext {
            agent_name: &manifest.name,
            agent_id: agent_id_str.as_str(),
            event: openfang_types::agent::HookEvent::BeforePromptBuild,
            data: serde_json::json!({
                "system_prompt": &manifest.model.system_prompt,
                "user_message": user_message,
            }),
        };
        let _ = hook_reg.fire(&ctx);
    }

    // Build the system prompt — base prompt comes from kernel (prompt_builder),
    // we append recalled memories here since they are resolved at loop time.
    let mut system_prompt = manifest.model.system_prompt.clone();
    if !memories.is_empty() {
        let mem_pairs: Vec<(String, String)> = memories
            .iter()
            .map(|m| (String::new(), m.content.clone()))
            .collect();
        system_prompt.push_str("\n\n");
        system_prompt.push_str(&crate::prompt_builder::build_memory_section(&mem_pairs));
    }

    // Persist tool chains only for the active loop; completed traces should not
    // keep inflating future session context.
    compact_session_execution_trace(session);

    let effective_exec_policy = manifest.exec_policy.as_ref();
    let auto_confirm_note = maybe_auto_confirm_pending_wardrobe(
        manifest,
        user_message,
        session,
        available_tools,
        kernel.as_ref(),
        skill_registry,
        mcp_connections,
        web_ctx,
        browser_ctx,
        workspace_root,
        media_engine,
        effective_exec_policy,
        tts_engine,
        docker_config,
        process_manager,
        &agent_id_str,
        &hand_allowed_env,
    )
    .await;

    // Add the user message to session history (with inline media if present)
    if media_blocks.is_empty() {
        session.messages.push(Message::user(user_message));
    } else {
        let mut blocks = vec![ContentBlock::Text {
            text: user_message.to_string(),
        }];
        blocks.extend(media_blocks);
        session.messages.push(Message {
            role: Role::User,
            content: MessageContent::Blocks(blocks),
        });
    }

    if let Some(hint) = wardrobe_followup_phase_system_hint(manifest, user_message, session) {
        system_prompt.push_str("\n\n[衣橱阶段约束]\n");
        system_prompt.push_str(hint);
    }
    if let Some(note) = &auto_confirm_note {
        system_prompt.push_str("\n\n[衣橱自动确认]\n");
        system_prompt.push_str(note);
    }

    // Build the messages for the LLM, filtering system messages
    // System prompt goes into the separate `system` field
    let llm_messages: Vec<Message> = session
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .cloned()
        .collect();

    // Validate and repair session history (drop orphans, merge consecutive)
    let mut messages = crate::session_repair::validate_and_repair(&llm_messages);
    let mut total_usage = TokenUsage::default();
    let final_response;

    // Safety valve: trim excessively long message histories to prevent context overflow.
    // The full compaction system handles sophisticated summarization, but this prevents
    // the catastrophic case where 200+ messages cause instant context overflow.
    if messages.len() > MAX_HISTORY_MESSAGES {
        let trim_count = messages.len() - MAX_HISTORY_MESSAGES;
        warn!(
            agent = %manifest.name,
            total_messages = messages.len(),
            trimming = trim_count,
            "Trimming old messages to prevent context overflow"
        );
        messages.drain(..trim_count);
    }

    // Use autonomous config max_iterations if set, else default
    let max_iterations = manifest
        .autonomous
        .as_ref()
        .map(|a| a.max_iterations)
        .unwrap_or(MAX_ITERATIONS);

    // Initialize loop guard — scale circuit breaker for autonomous agents
    let loop_guard_config = {
        let mut cfg = LoopGuardConfig::default();
        if max_iterations > cfg.global_circuit_breaker {
            cfg.global_circuit_breaker = max_iterations * 3;
        }
        cfg
    };
    let mut loop_guard = LoopGuard::new(loop_guard_config);
    let mut consecutive_max_tokens: u32 = 0;
    let mut successful_tools_this_turn: HashSet<String> = HashSet::new();

    // Build context budget from model's actual context window (or fallback to default)
    let ctx_window = context_window_tokens.unwrap_or(DEFAULT_CONTEXT_WINDOW);
    let context_budget = ContextBudget::new(ctx_window);

    for iteration in 0..max_iterations {
        debug!(iteration, "Agent loop iteration");

        // Context overflow recovery pipeline (replaces emergency_trim_messages)
        let recovery =
            recover_from_overflow(&mut messages, &system_prompt, available_tools, ctx_window);
        if recovery == RecoveryStage::FinalError {
            warn!("Context overflow unrecoverable — suggest /reset or /compact");
        }

        // Context guard: compact oversized tool results before LLM call
        apply_context_guard(&mut messages, &context_budget, available_tools);

        // Apply dynamic injections (world state) — only affects the LLM request copy,
        // not the session messages. Injections are drained on first call.
        let mut llm_messages = messages.clone();
        apply_dynamic_injections(&mut llm_messages);

        // Persist final LLM request for debugging (best-effort, rotates at 10 files)
        save_llm_request_log(
            manifest.workspace.as_deref(),
            &manifest.name,
            &llm_messages,
            &system_prompt,
            &manifest.model.model,
            iteration,
        );

        let request = CompletionRequest {
            model: manifest.model.model.clone(),
            messages: llm_messages,
            tools: available_tools.to_vec(),
            max_tokens: manifest.model.max_tokens,
            temperature: manifest.model.temperature,
            system: Some(system_prompt.clone()),
            thinking: None,
        };

        // Notify phase: Thinking
        if let Some(cb) = on_phase {
            cb(LoopPhase::Thinking);
        }

        // Call LLM with retry, error classification, and circuit breaker
        let provider_name = manifest.model.provider.as_str();
        let mut response = match call_with_retry(&*driver, request, Some(provider_name), None).await
        {
            Ok(response) => response,
            Err(error) => {
                save_projected_session_best_effort(
                    memory,
                    session,
                    "Failed to save projected session after LLM error",
                );
                return Err(error);
            }
        };

        total_usage.input_tokens += response.usage.input_tokens;
        total_usage.output_tokens += response.usage.output_tokens;

        // Recover tool calls output as text by models that don't use the tool_calls API field
        // (e.g. Groq/Llama, DeepSeek emit `<function=name>{json}</function>` in text)
        if matches!(
            response.stop_reason,
            StopReason::EndTurn | StopReason::StopSequence
        ) && response.tool_calls.is_empty()
        {
            let recovered = recover_text_tool_calls(&response.text(), available_tools);
            if !recovered.is_empty() {
                info!(
                    count = recovered.len(),
                    "Recovered text-based tool calls → promoting to ToolUse"
                );
                response.tool_calls = recovered;
                response.stop_reason = StopReason::ToolUse;
                // Build ToolUse content blocks from recovered calls
                let mut new_blocks: Vec<ContentBlock> = Vec::new();
                for tc in &response.tool_calls {
                    new_blocks.push(ContentBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        input: tc.input.clone(),
                    });
                }
                response.content = new_blocks;
            }
        }

        match response.stop_reason {
            StopReason::EndTurn | StopReason::StopSequence => {
                // LLM is done — extract text and save
                let text = response.text();

                // Parse reply directives from the response text
                let (cleaned_text, parsed_directives) =
                    crate::reply_directives::parse_directives(&text);
                let text = cleaned_text;

                // NO_REPLY: agent intentionally chose not to reply
                if text.trim() == "NO_REPLY" || parsed_directives.silent {
                    debug!(agent = %manifest.name, "Agent chose NO_REPLY/silent — silent completion");
                    session
                        .messages
                        .push(Message::assistant("[no reply needed]".to_string()));
                    save_projected_session(memory, session)?;
                    return Ok(AgentLoopResult {
                        response: String::new(),
                        total_usage,
                        iterations: iteration + 1,
                        cost_usd: None,
                        silent: true,
                        directives: openfang_types::message::ReplyDirectives {
                            reply_to: parsed_directives.reply_to,
                            current_thread: parsed_directives.current_thread,
                            silent: true,
                        },
                    });
                }

                // Auto-wrap: tool-only agent produced text without calling any
                // tool → write the text into Turn Script so the kernel delivers
                // it.  This makes message delivery a code guarantee, not an LLM
                // best-effort.
                {
                    let silent_after = manifest_silent_after_tools(manifest);
                    if !silent_after.is_empty()
                        && !text.trim().is_empty()
                        && response.tool_calls.is_empty()
                    {
                        warn!(
                            agent = %manifest.name,
                            "Tool-only agent produced text without tool calls — auto-wrapping into Turn Script"
                        );
                        if let Err(e) = auto_wrap_text_to_turn_script(&text) {
                            warn!(agent = %manifest.name, error = %e, "Failed to auto-wrap text into Turn Script");
                        }
                        session.messages.push(Message::assistant(text));
                        save_projected_session(memory, session)?;
                        return Ok(AgentLoopResult {
                            response: String::new(),
                            total_usage,
                            iterations: iteration + 1,
                            cost_usd: None,
                            silent: true,
                            directives: openfang_types::message::ReplyDirectives {
                                reply_to: parsed_directives.reply_to,
                                current_thread: parsed_directives.current_thread,
                                silent: true,
                            },
                        });
                    }
                }

                // One-shot retry: if the very first LLM call returns empty text
                // with no tool use, try once more before accepting the empty result.
                // This catches transient LLM hiccups (overload, empty stream, etc.).
                if text.trim().is_empty() && iteration == 0 && response.tool_calls.is_empty() {
                    warn!(agent = %manifest.name, "Empty response on first call, retrying once");
                    messages.push(Message::assistant("[no response]".to_string()));
                    messages.push(Message::user("Please provide your response.".to_string()));
                    continue;
                }

                // Guard against empty response — covers both iteration 0 and post-tool cycles
                let text = if text.trim().is_empty() {
                    warn!(
                        agent = %manifest.name,
                        iteration,
                        input_tokens = total_usage.input_tokens,
                        output_tokens = total_usage.output_tokens,
                        messages_count = messages.len(),
                        "Empty response from LLM — guard activated"
                    );
                    if iteration > 0 {
                        "[Task completed — the agent executed tools but did not produce a text summary.]".to_string()
                    } else {
                        "[The model returned an empty response. This usually means the model is overloaded, the context is too large, or the API key lacks credits. Try again or check /status.]".to_string()
                    }
                } else {
                    text
                };
                final_response = text.clone();
                session.messages.push(Message::assistant(text));

                // Prune NO_REPLY heartbeat turns to save context budget
                crate::session_repair::prune_heartbeat_turns(&mut session.messages, 10);
                save_projected_session(memory, session)?;

                remember_memory_imprint(
                    memory,
                    embedding_driver,
                    session,
                    user_message,
                    &final_response,
                )
                .await;

                // Notify phase: Done
                if let Some(cb) = on_phase {
                    cb(LoopPhase::Done);
                }

                info!(
                    agent = %manifest.name,
                    iterations = iteration + 1,
                    tokens = total_usage.total(),
                    "Agent loop completed"
                );

                // Fire AgentLoopEnd hook
                if let Some(hook_reg) = hooks {
                    let ctx = crate::hooks::HookContext {
                        agent_name: &manifest.name,
                        agent_id: agent_id_str.as_str(),
                        event: openfang_types::agent::HookEvent::AgentLoopEnd,
                        data: serde_json::json!({
                            "iterations": iteration + 1,
                            "response_length": final_response.len(),
                        }),
                    };
                    let _ = hook_reg.fire(&ctx);
                }

                return Ok(AgentLoopResult {
                    response: final_response,
                    total_usage,
                    iterations: iteration + 1,
                    cost_usd: None,
                    silent: false,
                    directives: Default::default(),
                });
            }
            StopReason::ToolUse => {
                // Reset MaxTokens continuation counter on tool use
                consecutive_max_tokens = 0;

                // Execute tool calls
                let assistant_blocks = response.content.clone();

                // Add assistant message with tool use blocks
                session.messages.push(Message {
                    role: Role::Assistant,
                    content: MessageContent::Blocks(assistant_blocks.clone()),
                });
                messages.push(Message {
                    role: Role::Assistant,
                    content: MessageContent::Blocks(assistant_blocks),
                });

                // Build allowed tool names list for capability enforcement
                let allowed_tool_names: Vec<String> =
                    available_tools.iter().map(|t| t.name.clone()).collect();
                let caller_id_str = session.agent_id.to_string();

                // Execute each tool call with loop guard, timeout, and truncation
                let mut tool_result_blocks = Vec::new();
                let mut response_already_delivered = false;
                for tool_call in &response.tool_calls {
                    // Loop guard check
                    let verdict = loop_guard.check(&tool_call.name, &tool_call.input);
                    match &verdict {
                        LoopGuardVerdict::CircuitBreak(msg) => {
                            warn!(tool = %tool_call.name, "Circuit breaker triggered");
                            save_projected_session_best_effort(
                                memory,
                                session,
                                "Failed to save projected session on circuit break",
                            );
                            // Fire AgentLoopEnd hook on circuit break
                            if let Some(hook_reg) = hooks {
                                let ctx = crate::hooks::HookContext {
                                    agent_name: &manifest.name,
                                    agent_id: agent_id_str.as_str(),
                                    event: openfang_types::agent::HookEvent::AgentLoopEnd,
                                    data: serde_json::json!({
                                        "reason": "circuit_break",
                                        "error": msg.as_str(),
                                    }),
                                };
                                let _ = hook_reg.fire(&ctx);
                            }
                            return Err(OpenFangError::Internal(msg.clone()));
                        }
                        LoopGuardVerdict::Block(msg) => {
                            warn!(tool = %tool_call.name, "Tool call blocked by loop guard");
                            tool_result_blocks.push(ContentBlock::ToolResult {
                                tool_use_id: tool_call.id.clone(),
                                content: msg.clone(),
                                is_error: true,
                            });
                            continue;
                        }
                        _ => {} // Allow or Warn — proceed with execution
                    }

                    debug!(tool = %tool_call.name, id = %tool_call.id, "Executing tool");

                    if let Some(message) = wardrobe_confirmation_guard_message(
                        user_message,
                        tool_call,
                        &successful_tools_this_turn,
                    ) {
                        tool_result_blocks.push(ContentBlock::ToolResult {
                            tool_use_id: tool_call.id.clone(),
                            content: message,
                            is_error: true,
                        });
                        continue;
                    }

                    if let Some(message) = selfie_context_guard_message(
                        manifest,
                        user_message,
                        tool_call,
                        &successful_tools_this_turn,
                    ) {
                        tool_result_blocks.push(ContentBlock::ToolResult {
                            tool_use_id: tool_call.id.clone(),
                            content: message,
                            is_error: true,
                        });
                        continue;
                    }

                    // Notify phase: ToolUse
                    if let Some(cb) = on_phase {
                        let sanitized: String = tool_call
                            .name
                            .chars()
                            .filter(|c| !c.is_control())
                            .take(64)
                            .collect();
                        cb(LoopPhase::ToolUse {
                            tool_name: sanitized,
                        });
                    }

                    // Fire BeforeToolCall hook (can block execution)
                    if let Some(hook_reg) = hooks {
                        let ctx = crate::hooks::HookContext {
                            agent_name: &manifest.name,
                            agent_id: &caller_id_str,
                            event: openfang_types::agent::HookEvent::BeforeToolCall,
                            data: serde_json::json!({
                                "tool_name": &tool_call.name,
                                "input": &tool_call.input,
                            }),
                        };
                        if let Err(reason) = hook_reg.fire(&ctx) {
                            tool_result_blocks.push(ContentBlock::ToolResult {
                                tool_use_id: tool_call.id.clone(),
                                content: format!(
                                    "Hook blocked tool '{}': {}",
                                    tool_call.name, reason
                                ),
                                is_error: true,
                            });
                            continue;
                        }
                    }

                    let result = if let Some(async_result) = maybe_enqueue_async_media(
                        manifest,
                        tool_call,
                        user_message,
                        kernel.as_ref(),
                        &caller_id_str,
                    )
                    .await
                    {
                        async_result
                    } else {
                        // Timeout-wrapped execution
                        let tool_timeout_secs = tool_timeout_secs(&tool_call.name);
                        match tokio::time::timeout(
                            Duration::from_secs(tool_timeout_secs),
                            tool_runner::execute_tool(
                                &tool_call.id,
                                &tool_call.name,
                                &tool_call.input,
                                kernel.as_ref(),
                                Some(&allowed_tool_names),
                                Some(&caller_id_str),
                                skill_registry,
                                mcp_connections,
                                web_ctx,
                                browser_ctx,
                                if hand_allowed_env.is_empty() {
                                    None
                                } else {
                                    Some(&hand_allowed_env)
                                },
                                workspace_root,
                                media_engine,
                                effective_exec_policy,
                                tts_engine,
                                docker_config,
                                process_manager,
                            ),
                        )
                        .await
                        {
                            Ok(result) => result,
                            Err(_) => {
                                warn!(tool = %tool_call.name, timeout_secs = tool_timeout_secs, "Tool execution timed out");
                                openfang_types::tool::ToolResult {
                                    tool_use_id: tool_call.id.clone(),
                                    content: format!(
                                        "Tool '{}' timed out after {}s.",
                                        tool_call.name, tool_timeout_secs
                                    ),
                                    is_error: true,
                                    response_delivered: false,
                                }
                            }
                        }
                    };

                    // Fire AfterToolCall hook
                    if let Some(hook_reg) = hooks {
                        let ctx = crate::hooks::HookContext {
                            agent_name: &manifest.name,
                            agent_id: caller_id_str.as_str(),
                            event: openfang_types::agent::HookEvent::AfterToolCall,
                            data: serde_json::json!({
                                "tool_name": &tool_call.name,
                                "result": &result.content,
                                "is_error": result.is_error,
                            }),
                        };
                        let _ = hook_reg.fire(&ctx);
                    }

                    // Dynamic truncation based on context budget (replaces flat MAX_TOOL_RESULT_CHARS)
                    let content = truncate_tool_result_dynamic(&result.content, &context_budget);

                    // Append warning if verdict was Warn
                    let final_content = if let LoopGuardVerdict::Warn(ref warn_msg) = verdict {
                        format!("{content}\n\n[LOOP GUARD] {warn_msg}")
                    } else {
                        content
                    };

                    // Track if any tool already delivered the response via side-channel
                    if result.response_delivered && !result.is_error {
                        response_already_delivered = true;
                    }

                    tool_result_blocks.push(ContentBlock::ToolResult {
                        tool_use_id: result.tool_use_id,
                        content: final_content,
                        is_error: result.is_error,
                    });
                    if !result.is_error {
                        successful_tools_this_turn.insert(tool_call.name.clone());
                    }
                }

                let had_tool_errors = tool_result_blocks.iter().any(|block| {
                    matches!(block, ContentBlock::ToolResult { is_error: true, .. })
                });
                let silent_after_tools = manifest_silent_after_tools(manifest);
                let should_silent_after_tools = !silent_after_tools.is_empty()
                    && !had_tool_errors
                    && successful_tools_this_turn
                        .iter()
                        .any(|tool_name| silent_after_tools.contains(tool_name));

                // If a tool already delivered the response (e.g. voice message via MCP),
                // or this agent is configured to finish immediately after successful
                // tool execution, exit silently — no need for another LLM round.
                if response_already_delivered || should_silent_after_tools {
                    // Text delivery is a code guarantee: if the LLM produced
                    // text alongside tool calls, auto-wrap it into Turn Script
                    // so the kernel delivers it to the user.
                    if should_silent_after_tools {
                        let companion_text = response.text();
                        if !companion_text.trim().is_empty() {
                            debug!(agent = %manifest.name, "Auto-wrapping text into Turn Script alongside tool calls");
                            if let Err(e) = auto_wrap_text_to_turn_script(companion_text.trim()) {
                                warn!(agent = %manifest.name, error = %e, "Failed to auto-wrap text into Turn Script");
                            }
                        }
                    }

                    let reason = if response_already_delivered {
                        "response delivered via side-channel"
                    } else {
                        "configured silent-after-tools"
                    };
                    debug!(agent = %manifest.name, reason, "Ending turn silently after tool execution");
                    // Still save the tool results to session for context continuity
                    let tool_results_msg = Message {
                        role: Role::User,
                        content: MessageContent::Blocks(tool_result_blocks),
                    };
                    session.messages.push(tool_results_msg);

                    let placeholder = if response_already_delivered {
                        // Extract the spoken/delivered text from side-channel tool inputs
                        // so it persists in history after session projection strips tool blocks.
                        let delivered_text = extract_side_channel_text(&response.tool_calls);
                        if delivered_text.is_empty() {
                            persistent_turn_placeholder(&response.tool_calls)
                        } else {
                            format!("（发了条语音）{delivered_text}")
                        }
                    } else {
                        persistent_turn_placeholder(&response.tool_calls)
                    };
                    session.messages.push(Message::assistant(placeholder));
                    save_projected_session(memory, session)?;
                    return Ok(AgentLoopResult {
                        response: String::new(),
                        total_usage,
                        iterations: iteration + 1,
                        cost_usd: None,
                        silent: true,
                        directives: Default::default(),
                    });
                }

                // Add tool results as a user message (Anthropic API requirement)
                let tool_results_msg = Message {
                    role: Role::User,
                    content: MessageContent::Blocks(tool_result_blocks.clone()),
                };
                session.messages.push(tool_results_msg.clone());
                messages.push(tool_results_msg);

                // Interim save after tool execution to prevent data loss on crash
                if let Err(e) = memory.save_session(session) {
                    warn!("Failed to interim-save session: {e}");
                }
            }
            StopReason::MaxTokens => {
                consecutive_max_tokens += 1;
                if consecutive_max_tokens >= MAX_CONTINUATIONS {
                    // Return partial response instead of continuing forever
                    let text = response.text();
                    let text = if text.trim().is_empty() {
                        "[Partial response — token limit reached with no text output.]".to_string()
                    } else {
                        text
                    };
                    session.messages.push(Message::assistant(&text));
                    save_projected_session_best_effort(
                        memory,
                        session,
                        "Failed to save projected session on max continuations",
                    );
                    warn!(
                        iteration,
                        consecutive_max_tokens,
                        "Max continuations reached, returning partial response"
                    );
                    // Fire AgentLoopEnd hook
                    if let Some(hook_reg) = hooks {
                        let ctx = crate::hooks::HookContext {
                            agent_name: &manifest.name,
                            agent_id: agent_id_str.as_str(),
                            event: openfang_types::agent::HookEvent::AgentLoopEnd,
                            data: serde_json::json!({
                                "iterations": iteration + 1,
                                "reason": "max_continuations",
                            }),
                        };
                        let _ = hook_reg.fire(&ctx);
                    }
                    return Ok(AgentLoopResult {
                        response: text,
                        total_usage,
                        iterations: iteration + 1,
                        cost_usd: None,
                        silent: false,
                        directives: Default::default(),
                    });
                }
                // Model hit token limit — add partial response and continue
                let text = response.text();
                session.messages.push(Message::assistant(&text));
                messages.push(Message::assistant(&text));
                session.messages.push(Message::user("Please continue."));
                messages.push(Message::user("Please continue."));
                warn!(iteration, "Max tokens hit, continuing");
            }
        }
    }

    save_projected_session_best_effort(
        memory,
        session,
        "Failed to save projected session on max iterations",
    );

    // Fire AgentLoopEnd hook on max iterations exceeded
    if let Some(hook_reg) = hooks {
        let ctx = crate::hooks::HookContext {
            agent_name: &manifest.name,
            agent_id: agent_id_str.as_str(),
            event: openfang_types::agent::HookEvent::AgentLoopEnd,
            data: serde_json::json!({
                "reason": "max_iterations_exceeded",
                "iterations": max_iterations,
            }),
        };
        let _ = hook_reg.fire(&ctx);
    }

    Err(OpenFangError::MaxIterationsExceeded(max_iterations))
}

/// Call an LLM driver with automatic retry on rate-limit and overload errors.
///
/// Uses the `llm_errors` classifier for smart error handling and the
/// `ProviderCooldown` circuit breaker to prevent request storms.
async fn call_with_retry(
    driver: &dyn LlmDriver,
    request: CompletionRequest,
    provider: Option<&str>,
    cooldown: Option<&ProviderCooldown>,
) -> OpenFangResult<crate::llm_driver::CompletionResponse> {
    // Check circuit breaker before calling
    if let (Some(provider), Some(cooldown)) = (provider, cooldown) {
        match cooldown.check(provider) {
            CooldownVerdict::Reject {
                reason,
                retry_after_secs,
            } => {
                return Err(OpenFangError::LlmDriver(format!(
                    "Provider '{provider}' is in cooldown ({reason}). Retry in {retry_after_secs}s."
                )));
            }
            CooldownVerdict::AllowProbe => {
                debug!(provider, "Allowing probe request through circuit breaker");
            }
            CooldownVerdict::Allow => {}
        }
    }

    let mut last_error = None;

    for attempt in 0..=MAX_RETRIES {
        match driver.complete(request.clone()).await {
            Ok(response) => {
                // Record success with circuit breaker
                if let (Some(provider), Some(cooldown)) = (provider, cooldown) {
                    cooldown.record_success(provider);
                }
                return Ok(response);
            }
            Err(LlmError::RateLimited { retry_after_ms }) => {
                if attempt == MAX_RETRIES {
                    if let (Some(provider), Some(cooldown)) = (provider, cooldown) {
                        cooldown.record_failure(provider, false);
                    }
                    return Err(OpenFangError::LlmDriver(format!(
                        "Rate limited after {} retries",
                        MAX_RETRIES
                    )));
                }
                let delay = std::cmp::max(retry_after_ms, BASE_RETRY_DELAY_MS * 2u64.pow(attempt));
                warn!(
                    attempt,
                    delay_ms = delay,
                    "Rate limited, retrying after delay"
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                last_error = Some("Rate limited".to_string());
            }
            Err(LlmError::Overloaded { retry_after_ms }) => {
                if attempt == MAX_RETRIES {
                    if let (Some(provider), Some(cooldown)) = (provider, cooldown) {
                        cooldown.record_failure(provider, false);
                    }
                    return Err(OpenFangError::LlmDriver(format!(
                        "Model overloaded after {} retries",
                        MAX_RETRIES
                    )));
                }
                let delay = std::cmp::max(retry_after_ms, BASE_RETRY_DELAY_MS * 2u64.pow(attempt));
                warn!(
                    attempt,
                    delay_ms = delay,
                    "Model overloaded, retrying after delay"
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                last_error = Some("Overloaded".to_string());
            }
            Err(e) => {
                // Use classifier for smarter error handling
                let raw_error = e.to_string();
                let classified = llm_errors::classify_error(&raw_error, None);
                warn!(
                    category = ?classified.category,
                    retryable = classified.is_retryable,
                    raw = %raw_error,
                    "LLM error classified: {}",
                    classified.sanitized_message
                );

                if let (Some(provider), Some(cooldown)) = (provider, cooldown) {
                    cooldown.record_failure(provider, classified.is_billing);
                }

                // Include raw error detail so dashboard users can debug
                let user_msg = if classified.category == llm_errors::LlmErrorCategory::Format {
                    format!("{} — raw: {}", classified.sanitized_message, raw_error)
                } else {
                    classified.sanitized_message
                };
                return Err(OpenFangError::LlmDriver(user_msg));
            }
        }
    }

    Err(OpenFangError::LlmDriver(
        last_error.unwrap_or_else(|| "Unknown error".to_string()),
    ))
}

/// Call an LLM driver in streaming mode with automatic retry on rate-limit and overload errors.
///
/// Uses the `llm_errors` classifier and `ProviderCooldown` circuit breaker.
async fn stream_with_retry(
    driver: &dyn LlmDriver,
    request: CompletionRequest,
    tx: mpsc::Sender<StreamEvent>,
    provider: Option<&str>,
    cooldown: Option<&ProviderCooldown>,
) -> OpenFangResult<crate::llm_driver::CompletionResponse> {
    // Check circuit breaker before calling
    if let (Some(provider), Some(cooldown)) = (provider, cooldown) {
        match cooldown.check(provider) {
            CooldownVerdict::Reject {
                reason,
                retry_after_secs,
            } => {
                return Err(OpenFangError::LlmDriver(format!(
                    "Provider '{provider}' is in cooldown ({reason}). Retry in {retry_after_secs}s."
                )));
            }
            CooldownVerdict::AllowProbe => {
                debug!(
                    provider,
                    "Allowing probe request through circuit breaker (stream)"
                );
            }
            CooldownVerdict::Allow => {}
        }
    }

    let mut last_error = None;

    for attempt in 0..=MAX_RETRIES {
        match driver.stream(request.clone(), tx.clone()).await {
            Ok(response) => {
                if let (Some(provider), Some(cooldown)) = (provider, cooldown) {
                    cooldown.record_success(provider);
                }
                return Ok(response);
            }
            Err(LlmError::RateLimited { retry_after_ms }) => {
                if attempt == MAX_RETRIES {
                    if let (Some(provider), Some(cooldown)) = (provider, cooldown) {
                        cooldown.record_failure(provider, false);
                    }
                    return Err(OpenFangError::LlmDriver(format!(
                        "Rate limited after {} retries",
                        MAX_RETRIES
                    )));
                }
                let delay = std::cmp::max(retry_after_ms, BASE_RETRY_DELAY_MS * 2u64.pow(attempt));
                warn!(
                    attempt,
                    delay_ms = delay,
                    "Rate limited (stream), retrying after delay"
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                last_error = Some("Rate limited".to_string());
            }
            Err(LlmError::Overloaded { retry_after_ms }) => {
                if attempt == MAX_RETRIES {
                    if let (Some(provider), Some(cooldown)) = (provider, cooldown) {
                        cooldown.record_failure(provider, false);
                    }
                    return Err(OpenFangError::LlmDriver(format!(
                        "Model overloaded after {} retries",
                        MAX_RETRIES
                    )));
                }
                let delay = std::cmp::max(retry_after_ms, BASE_RETRY_DELAY_MS * 2u64.pow(attempt));
                warn!(
                    attempt,
                    delay_ms = delay,
                    "Model overloaded (stream), retrying after delay"
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                last_error = Some("Overloaded".to_string());
            }
            Err(e) => {
                let raw_error = e.to_string();
                let classified = llm_errors::classify_error(&raw_error, None);
                warn!(
                    category = ?classified.category,
                    retryable = classified.is_retryable,
                    raw = %raw_error,
                    "LLM stream error classified: {}",
                    classified.sanitized_message
                );

                if let (Some(provider), Some(cooldown)) = (provider, cooldown) {
                    cooldown.record_failure(provider, classified.is_billing);
                }

                let user_msg = if classified.category == llm_errors::LlmErrorCategory::Format {
                    format!("{} — raw: {}", classified.sanitized_message, raw_error)
                } else {
                    classified.sanitized_message
                };
                return Err(OpenFangError::LlmDriver(user_msg));
            }
        }
    }

    Err(OpenFangError::LlmDriver(
        last_error.unwrap_or_else(|| "Unknown error".to_string()),
    ))
}

/// Run the agent execution loop with streaming support.
///
/// Like `run_agent_loop`, but sends `StreamEvent`s to the provided channel
/// as tokens arrive from the LLM. Tool execution happens between LLM calls
/// and is not streamed.
#[allow(clippy::too_many_arguments)]
pub async fn run_agent_loop_streaming(
    manifest: &AgentManifest,
    user_message: &str,
    session: &mut Session,
    memory: &MemorySubstrate,
    driver: Arc<dyn LlmDriver>,
    available_tools: &[ToolDefinition],
    kernel: Option<Arc<dyn KernelHandle>>,
    stream_tx: mpsc::Sender<StreamEvent>,
    skill_registry: Option<&SkillRegistry>,
    mcp_connections: Option<&tokio::sync::Mutex<Vec<McpConnection>>>,
    web_ctx: Option<&WebToolsContext>,
    browser_ctx: Option<&crate::browser::BrowserManager>,
    embedding_driver: Option<&(dyn EmbeddingDriver + Send + Sync)>,
    workspace_root: Option<&Path>,
    on_phase: Option<&PhaseCallback>,
    media_engine: Option<&crate::media_understanding::MediaEngine>,
    tts_engine: Option<&crate::tts::TtsEngine>,
    docker_config: Option<&openfang_types::config::DockerSandboxConfig>,
    hooks: Option<&crate::hooks::HookRegistry>,
    context_window_tokens: Option<usize>,
    process_manager: Option<&crate::process_manager::ProcessManager>,
    media_blocks: Vec<ContentBlock>,
) -> OpenFangResult<AgentLoopResult> {
    info!(agent = %manifest.name, "Starting streaming agent loop");

    // Extract hand-allowed env vars from manifest metadata (set by kernel for hand settings)
    let hand_allowed_env: Vec<String> = manifest
        .metadata
        .get("hand_allowed_env")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    // Recall relevant memories — prefer vector similarity search when embedding driver is available
    let memories = if let Some(emb) = embedding_driver {
        match emb.embed_one(user_message).await {
            Ok(query_vec) => {
                debug!("Using vector recall (streaming, dims={})", query_vec.len());
                memory
                    .recall_with_embedding_async(
                        user_message,
                        5,
                        Some(MemoryFilter {
                            agent_id: Some(session.agent_id),
                            ..Default::default()
                        }),
                        Some(&query_vec),
                    )
                    .await
                    .unwrap_or_default()
            }
            Err(e) => {
                warn!("Embedding recall failed (streaming), falling back to text search: {e}");
                memory
                    .recall(
                        user_message,
                        5,
                        Some(MemoryFilter {
                            agent_id: Some(session.agent_id),
                            ..Default::default()
                        }),
                    )
                    .await
                    .unwrap_or_default()
            }
        }
    } else {
        memory
            .recall(
                user_message,
                5,
                Some(MemoryFilter {
                    agent_id: Some(session.agent_id),
                    ..Default::default()
                }),
            )
            .await
            .unwrap_or_default()
    };

    // Fire BeforePromptBuild hook
    let agent_id_str = session.agent_id.0.to_string();
    if let Some(hook_reg) = hooks {
        let ctx = crate::hooks::HookContext {
            agent_name: &manifest.name,
            agent_id: agent_id_str.as_str(),
            event: openfang_types::agent::HookEvent::BeforePromptBuild,
            data: serde_json::json!({
                "system_prompt": &manifest.model.system_prompt,
                "user_message": user_message,
            }),
        };
        let _ = hook_reg.fire(&ctx);
    }

    // Build the system prompt — base prompt comes from kernel (prompt_builder),
    // we append recalled memories here since they are resolved at loop time.
    let mut system_prompt = manifest.model.system_prompt.clone();
    if !memories.is_empty() {
        let mem_pairs: Vec<(String, String)> = memories
            .iter()
            .map(|m| (String::new(), m.content.clone()))
            .collect();
        system_prompt.push_str("\n\n");
        system_prompt.push_str(&crate::prompt_builder::build_memory_section(&mem_pairs));
    }

    let effective_exec_policy = manifest.exec_policy.as_ref();
    let auto_confirm_note = maybe_auto_confirm_pending_wardrobe(
        manifest,
        user_message,
        session,
        available_tools,
        kernel.as_ref(),
        skill_registry,
        mcp_connections,
        web_ctx,
        browser_ctx,
        workspace_root,
        media_engine,
        effective_exec_policy,
        tts_engine,
        docker_config,
        process_manager,
        &agent_id_str,
        &hand_allowed_env,
    )
    .await;

    // Add the user message to session history (with inline media if present)
    if media_blocks.is_empty() {
        session.messages.push(Message::user(user_message));
    } else {
        let mut blocks = vec![ContentBlock::Text {
            text: user_message.to_string(),
        }];
        blocks.extend(media_blocks);
        session.messages.push(Message {
            role: Role::User,
            content: MessageContent::Blocks(blocks),
        });
    }

    if let Some(hint) = wardrobe_followup_phase_system_hint(manifest, user_message, session) {
        system_prompt.push_str("\n\n[衣橱阶段约束]\n");
        system_prompt.push_str(hint);
    }
    if let Some(note) = &auto_confirm_note {
        system_prompt.push_str("\n\n[衣橱自动确认]\n");
        system_prompt.push_str(note);
    }

    let llm_messages: Vec<Message> = session
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .cloned()
        .collect();

    // Validate and repair session history (drop orphans, merge consecutive)
    let mut messages = crate::session_repair::validate_and_repair(&llm_messages);
    let mut total_usage = TokenUsage::default();
    let final_response;

    // Safety valve: trim excessively long message histories to prevent context overflow.
    if messages.len() > MAX_HISTORY_MESSAGES {
        let trim_count = messages.len() - MAX_HISTORY_MESSAGES;
        warn!(
            agent = %manifest.name,
            total_messages = messages.len(),
            trimming = trim_count,
            "Trimming old messages to prevent context overflow (streaming)"
        );
        messages.drain(..trim_count);
    }

    // Use autonomous config max_iterations if set, else default
    let max_iterations = manifest
        .autonomous
        .as_ref()
        .map(|a| a.max_iterations)
        .unwrap_or(MAX_ITERATIONS);

    // Initialize loop guard — scale circuit breaker for autonomous agents
    let loop_guard_config = {
        let mut cfg = LoopGuardConfig::default();
        if max_iterations > cfg.global_circuit_breaker {
            cfg.global_circuit_breaker = max_iterations * 3;
        }
        cfg
    };
    let mut loop_guard = LoopGuard::new(loop_guard_config);
    let mut consecutive_max_tokens: u32 = 0;
    let mut successful_tools_this_turn: HashSet<String> = HashSet::new();

    // Build context budget from model's actual context window (or fallback to default)
    let ctx_window = context_window_tokens.unwrap_or(DEFAULT_CONTEXT_WINDOW);
    let context_budget = ContextBudget::new(ctx_window);

    for iteration in 0..max_iterations {
        debug!(iteration, "Streaming agent loop iteration");

        // Context overflow recovery pipeline (replaces emergency_trim_messages)
        let recovery =
            recover_from_overflow(&mut messages, &system_prompt, available_tools, ctx_window);
        match &recovery {
            RecoveryStage::None => {}
            RecoveryStage::FinalError => {
                if stream_tx.send(StreamEvent::PhaseChange {
                    phase: "context_warning".to_string(),
                    detail: Some("Context overflow unrecoverable. Use /reset or /compact.".to_string()),
                }).await.is_err() {
                    warn!("Stream consumer disconnected while sending context overflow warning");
                }
            }
            _ => {
                if stream_tx.send(StreamEvent::PhaseChange {
                    phase: "context_warning".to_string(),
                    detail: Some("Older messages trimmed to stay within context limits. Use /compact for smarter summarization.".to_string()),
                }).await.is_err() {
                    warn!("Stream consumer disconnected while sending context trim warning");
                }
            }
        }

        // Context guard: compact oversized tool results before LLM call
        apply_context_guard(&mut messages, &context_budget, available_tools);

        // Apply dynamic injections (world state) — only affects the LLM request copy,
        // not the session messages. Injections are drained on first call.
        let mut llm_messages = messages.clone();
        apply_dynamic_injections(&mut llm_messages);

        // Persist final LLM request for debugging (best-effort, rotates at 10 files)
        save_llm_request_log(
            manifest.workspace.as_deref(),
            &manifest.name,
            &llm_messages,
            &system_prompt,
            &manifest.model.model,
            iteration,
        );

        let request = CompletionRequest {
            model: manifest.model.model.clone(),
            messages: llm_messages,
            tools: available_tools.to_vec(),
            max_tokens: manifest.model.max_tokens,
            temperature: manifest.model.temperature,
            system: Some(system_prompt.clone()),
            thinking: None,
        };

        // Notify phase: Streaming (streaming variant always streams)
        if let Some(cb) = on_phase {
            cb(LoopPhase::Streaming);
        }

        // Stream LLM call with retry, error classification, and circuit breaker
        let provider_name = manifest.model.provider.as_str();
        let mut response = match stream_with_retry(
            &*driver,
            request,
            stream_tx.clone(),
            Some(provider_name),
            None,
        )
        .await
        {
            Ok(response) => response,
            Err(error) => {
                save_projected_session_best_effort(
                    memory,
                    session,
                    "Failed to save projected session after streaming LLM error",
                );
                return Err(error);
            }
        };

        total_usage.input_tokens += response.usage.input_tokens;
        total_usage.output_tokens += response.usage.output_tokens;

        // Recover tool calls output as text (streaming path)
        if matches!(
            response.stop_reason,
            StopReason::EndTurn | StopReason::StopSequence
        ) && response.tool_calls.is_empty()
        {
            let recovered = recover_text_tool_calls(&response.text(), available_tools);
            if !recovered.is_empty() {
                info!(
                    count = recovered.len(),
                    "Recovered text-based tool calls (streaming) → promoting to ToolUse"
                );
                response.tool_calls = recovered;
                response.stop_reason = StopReason::ToolUse;
                let mut new_blocks: Vec<ContentBlock> = Vec::new();
                for tc in &response.tool_calls {
                    new_blocks.push(ContentBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        input: tc.input.clone(),
                    });
                }
                response.content = new_blocks;
            }
        }

        match response.stop_reason {
            StopReason::EndTurn | StopReason::StopSequence => {
                let text = response.text();

                // Parse reply directives from the streaming response text
                let (cleaned_text_s, parsed_directives_s) =
                    crate::reply_directives::parse_directives(&text);
                let text = cleaned_text_s;

                // NO_REPLY: agent intentionally chose not to reply
                if text.trim() == "NO_REPLY" || parsed_directives_s.silent {
                    debug!(agent = %manifest.name, "Agent chose NO_REPLY/silent (streaming) — silent completion");
                    session
                        .messages
                        .push(Message::assistant("[no reply needed]".to_string()));
                    save_projected_session(memory, session)?;
                    return Ok(AgentLoopResult {
                        response: String::new(),
                        total_usage,
                        iterations: iteration + 1,
                        cost_usd: None,
                        silent: true,
                        directives: openfang_types::message::ReplyDirectives {
                            reply_to: parsed_directives_s.reply_to,
                            current_thread: parsed_directives_s.current_thread,
                            silent: true,
                        },
                    });
                }

                // Auto-wrap (streaming): same logic as non-streaming path.
                {
                    let silent_after = manifest_silent_after_tools(manifest);
                    if !silent_after.is_empty()
                        && !text.trim().is_empty()
                        && response.tool_calls.is_empty()
                    {
                        warn!(
                            agent = %manifest.name,
                            "Tool-only agent produced text without tool calls (streaming) — auto-wrapping into Turn Script"
                        );
                        if let Err(e) = auto_wrap_text_to_turn_script(&text) {
                            warn!(agent = %manifest.name, error = %e, "Failed to auto-wrap text into Turn Script (streaming)");
                        }
                        session.messages.push(Message::assistant(text));
                        save_projected_session(memory, session)?;
                        return Ok(AgentLoopResult {
                            response: String::new(),
                            total_usage,
                            iterations: iteration + 1,
                            cost_usd: None,
                            silent: true,
                            directives: openfang_types::message::ReplyDirectives {
                                reply_to: parsed_directives_s.reply_to,
                                current_thread: parsed_directives_s.current_thread,
                                silent: true,
                            },
                        });
                    }
                }

                // One-shot retry: if the very first LLM call returns empty text
                // with no tool use, try once more before accepting the empty result.
                if text.trim().is_empty() && iteration == 0 && response.tool_calls.is_empty() {
                    warn!(agent = %manifest.name, "Empty response on first call (streaming), retrying once");
                    messages.push(Message::assistant("[no response]".to_string()));
                    messages.push(Message::user("Please provide your response.".to_string()));
                    continue;
                }

                // Guard against empty response — covers both iteration 0 and post-tool cycles
                let text = if text.trim().is_empty() {
                    warn!(
                        agent = %manifest.name,
                        iteration,
                        input_tokens = total_usage.input_tokens,
                        output_tokens = total_usage.output_tokens,
                        messages_count = messages.len(),
                        "Empty response from LLM (streaming) — guard activated"
                    );
                    if iteration > 0 {
                        "[Task completed — the agent executed tools but did not produce a text summary.]".to_string()
                    } else {
                        "[The model returned an empty response. This usually means the model is overloaded, the context is too large, or the API key lacks credits. Try again or check /status.]".to_string()
                    }
                } else {
                    text
                };
                final_response = text.clone();
                session.messages.push(Message::assistant(text));

                // Prune NO_REPLY heartbeat turns to save context budget
                crate::session_repair::prune_heartbeat_turns(&mut session.messages, 10);
                save_projected_session(memory, session)?;

                remember_memory_imprint(
                    memory,
                    embedding_driver,
                    session,
                    user_message,
                    &final_response,
                )
                .await;

                // Notify phase: Done
                if let Some(cb) = on_phase {
                    cb(LoopPhase::Done);
                }

                info!(
                    agent = %manifest.name,
                    iterations = iteration + 1,
                    tokens = total_usage.total(),
                    "Streaming agent loop completed"
                );

                // Fire AgentLoopEnd hook
                if let Some(hook_reg) = hooks {
                    let ctx = crate::hooks::HookContext {
                        agent_name: &manifest.name,
                        agent_id: agent_id_str.as_str(),
                        event: openfang_types::agent::HookEvent::AgentLoopEnd,
                        data: serde_json::json!({
                            "iterations": iteration + 1,
                            "response_length": final_response.len(),
                        }),
                    };
                    let _ = hook_reg.fire(&ctx);
                }

                return Ok(AgentLoopResult {
                    response: final_response,
                    total_usage,
                    iterations: iteration + 1,
                    cost_usd: None,
                    silent: false,
                    directives: Default::default(),
                });
            }
            StopReason::ToolUse => {
                // Reset MaxTokens continuation counter on tool use
                consecutive_max_tokens = 0;

                let assistant_blocks = response.content.clone();

                session.messages.push(Message {
                    role: Role::Assistant,
                    content: MessageContent::Blocks(assistant_blocks.clone()),
                });
                messages.push(Message {
                    role: Role::Assistant,
                    content: MessageContent::Blocks(assistant_blocks),
                });

                let allowed_tool_names: Vec<String> =
                    available_tools.iter().map(|t| t.name.clone()).collect();
                let caller_id_str = session.agent_id.to_string();

                // Execute each tool call with loop guard, timeout, and truncation
                let mut tool_result_blocks = Vec::new();
                let mut response_already_delivered = false;
                for tool_call in &response.tool_calls {
                    // Loop guard check
                    let verdict = loop_guard.check(&tool_call.name, &tool_call.input);
                    match &verdict {
                        LoopGuardVerdict::CircuitBreak(msg) => {
                            warn!(tool = %tool_call.name, "Circuit breaker triggered (streaming)");
                            save_projected_session_best_effort(
                                memory,
                                session,
                                "Failed to save projected session on streaming circuit break",
                            );
                            // Fire AgentLoopEnd hook on circuit break
                            if let Some(hook_reg) = hooks {
                                let ctx = crate::hooks::HookContext {
                                    agent_name: &manifest.name,
                                    agent_id: agent_id_str.as_str(),
                                    event: openfang_types::agent::HookEvent::AgentLoopEnd,
                                    data: serde_json::json!({
                                        "reason": "circuit_break",
                                        "error": msg.as_str(),
                                    }),
                                };
                                let _ = hook_reg.fire(&ctx);
                            }
                            return Err(OpenFangError::Internal(msg.clone()));
                        }
                        LoopGuardVerdict::Block(msg) => {
                            warn!(tool = %tool_call.name, "Tool call blocked by loop guard (streaming)");
                            tool_result_blocks.push(ContentBlock::ToolResult {
                                tool_use_id: tool_call.id.clone(),
                                content: msg.clone(),
                                is_error: true,
                            });
                            continue;
                        }
                        _ => {} // Allow or Warn — proceed with execution
                    }

                    debug!(tool = %tool_call.name, id = %tool_call.id, "Executing tool (streaming)");

                    if let Some(message) = wardrobe_confirmation_guard_message(
                        user_message,
                        tool_call,
                        &successful_tools_this_turn,
                    ) {
                        tool_result_blocks.push(ContentBlock::ToolResult {
                            tool_use_id: tool_call.id.clone(),
                            content: message.clone(),
                            is_error: true,
                        });
                        if stream_tx
                            .send(StreamEvent::ToolExecutionResult {
                                name: tool_call.name.clone(),
                                result_preview: message.chars().take(300).collect(),
                                is_error: true,
                            })
                            .await
                            .is_err()
                        {
                            warn!(
                                agent = %manifest.name,
                                "Stream consumer disconnected while sending wardrobe guard result"
                            );
                        }
                        continue;
                    }

                    if let Some(message) = selfie_context_guard_message(
                        manifest,
                        user_message,
                        tool_call,
                        &successful_tools_this_turn,
                    ) {
                        tool_result_blocks.push(ContentBlock::ToolResult {
                            tool_use_id: tool_call.id.clone(),
                            content: message.clone(),
                            is_error: true,
                        });
                        if stream_tx
                            .send(StreamEvent::ToolExecutionResult {
                                name: tool_call.name.clone(),
                                result_preview: message.chars().take(300).collect(),
                                is_error: true,
                            })
                            .await
                            .is_err()
                        {
                            warn!(
                                agent = %manifest.name,
                                "Stream consumer disconnected while sending selfie guard result"
                            );
                        }
                        continue;
                    }

                    // Notify phase: ToolUse
                    if let Some(cb) = on_phase {
                        let sanitized: String = tool_call
                            .name
                            .chars()
                            .filter(|c| !c.is_control())
                            .take(64)
                            .collect();
                        cb(LoopPhase::ToolUse {
                            tool_name: sanitized,
                        });
                    }

                    // Fire BeforeToolCall hook (can block execution)
                    if let Some(hook_reg) = hooks {
                        let ctx = crate::hooks::HookContext {
                            agent_name: &manifest.name,
                            agent_id: &caller_id_str,
                            event: openfang_types::agent::HookEvent::BeforeToolCall,
                            data: serde_json::json!({
                                "tool_name": &tool_call.name,
                                "input": &tool_call.input,
                            }),
                        };
                        if let Err(reason) = hook_reg.fire(&ctx) {
                            tool_result_blocks.push(ContentBlock::ToolResult {
                                tool_use_id: tool_call.id.clone(),
                                content: format!(
                                    "Hook blocked tool '{}': {}",
                                    tool_call.name, reason
                                ),
                                is_error: true,
                            });
                            continue;
                        }
                    }

                    // Resolve effective exec policy (per-agent override or global)
                    let effective_exec_policy = manifest.exec_policy.as_ref();

                    let result = if let Some(async_result) = maybe_enqueue_async_media(
                        manifest,
                        tool_call,
                        user_message,
                        kernel.as_ref(),
                        &caller_id_str,
                    )
                    .await
                    {
                        async_result
                    } else {
                        // Timeout-wrapped execution
                        let tool_timeout_secs = tool_timeout_secs(&tool_call.name);
                        match tokio::time::timeout(
                            Duration::from_secs(tool_timeout_secs),
                            tool_runner::execute_tool(
                                &tool_call.id,
                                &tool_call.name,
                                &tool_call.input,
                                kernel.as_ref(),
                                Some(&allowed_tool_names),
                                Some(&caller_id_str),
                                skill_registry,
                                mcp_connections,
                                web_ctx,
                                browser_ctx,
                                if hand_allowed_env.is_empty() {
                                    None
                                } else {
                                    Some(&hand_allowed_env)
                                },
                                workspace_root,
                                media_engine,
                                effective_exec_policy,
                                tts_engine,
                                docker_config,
                                process_manager,
                            ),
                        )
                        .await
                        {
                            Ok(result) => result,
                            Err(_) => {
                                warn!(tool = %tool_call.name, timeout_secs = tool_timeout_secs, "Tool execution timed out (streaming)");
                                openfang_types::tool::ToolResult {
                                    tool_use_id: tool_call.id.clone(),
                                    content: format!(
                                        "Tool '{}' timed out after {}s.",
                                        tool_call.name, tool_timeout_secs
                                    ),
                                    is_error: true,
                                    response_delivered: false,
                                }
                            }
                        }
                    };

                    // Fire AfterToolCall hook
                    if let Some(hook_reg) = hooks {
                        let ctx = crate::hooks::HookContext {
                            agent_name: &manifest.name,
                            agent_id: caller_id_str.as_str(),
                            event: openfang_types::agent::HookEvent::AfterToolCall,
                            data: serde_json::json!({
                                "tool_name": &tool_call.name,
                                "result": &result.content,
                                "is_error": result.is_error,
                            }),
                        };
                        let _ = hook_reg.fire(&ctx);
                    }

                    // Dynamic truncation based on context budget (replaces flat MAX_TOOL_RESULT_CHARS)
                    let content = truncate_tool_result_dynamic(&result.content, &context_budget);

                    // Append warning if verdict was Warn
                    let final_content = if let LoopGuardVerdict::Warn(ref warn_msg) = verdict {
                        format!("{content}\n\n[LOOP GUARD] {warn_msg}")
                    } else {
                        content
                    };

                    // Notify client of tool execution result (detect dead consumer)
                    let preview: String = final_content.chars().take(300).collect();
                    if stream_tx
                        .send(StreamEvent::ToolExecutionResult {
                            name: tool_call.name.clone(),
                            result_preview: preview,
                            is_error: result.is_error,
                        })
                        .await
                        .is_err()
                    {
                        warn!(agent = %manifest.name, "Stream consumer disconnected — continuing tool loop but will not stream further");
                    }

                    // Track if any tool already delivered the response via side-channel
                    if result.response_delivered && !result.is_error {
                        response_already_delivered = true;
                    }

                    tool_result_blocks.push(ContentBlock::ToolResult {
                        tool_use_id: result.tool_use_id,
                        content: final_content,
                        is_error: result.is_error,
                    });
                    if !result.is_error {
                        successful_tools_this_turn.insert(tool_call.name.clone());
                    }
                }

                let had_tool_errors = tool_result_blocks.iter().any(|block| {
                    matches!(block, ContentBlock::ToolResult { is_error: true, .. })
                });
                let silent_after_tools = manifest_silent_after_tools(manifest);
                let should_silent_after_tools = !silent_after_tools.is_empty()
                    && !had_tool_errors
                    && successful_tools_this_turn
                        .iter()
                        .any(|tool_name| silent_after_tools.contains(tool_name));

                // If a tool already delivered the response (e.g. voice message via MCP),
                // or this agent is configured to finish immediately after successful
                // tool execution, exit silently — no need for another LLM round.
                if response_already_delivered || should_silent_after_tools {
                    // Text delivery is a code guarantee: if the LLM produced
                    // text alongside tool calls, auto-wrap it into Turn Script
                    // so the kernel delivers it to the user.
                    if should_silent_after_tools {
                        let companion_text = response.text();
                        if !companion_text.trim().is_empty() {
                            debug!(agent = %manifest.name, "Auto-wrapping text into Turn Script alongside tool calls (streaming)");
                            if let Err(e) = auto_wrap_text_to_turn_script(companion_text.trim()) {
                                warn!(agent = %manifest.name, error = %e, "Failed to auto-wrap text into Turn Script (streaming)");
                            }
                        }
                    }

                    let reason = if response_already_delivered {
                        "response delivered via side-channel"
                    } else {
                        "configured silent-after-tools"
                    };
                    debug!(agent = %manifest.name, reason, "Ending turn silently after tool execution (streaming)");
                    let tool_results_msg = Message {
                        role: Role::User,
                        content: MessageContent::Blocks(tool_result_blocks),
                    };
                    session.messages.push(tool_results_msg);

                    let placeholder = if response_already_delivered {
                        // Extract the spoken/delivered text from side-channel tool inputs
                        // so it persists in history after session projection strips tool blocks.
                        let delivered_text = extract_side_channel_text(&response.tool_calls);
                        if delivered_text.is_empty() {
                            persistent_turn_placeholder(&response.tool_calls)
                        } else {
                            format!("（发了条语音）{delivered_text}")
                        }
                    } else {
                        persistent_turn_placeholder(&response.tool_calls)
                    };
                    session.messages.push(Message::assistant(placeholder));
                    save_projected_session(memory, session)?;
                    return Ok(AgentLoopResult {
                        response: String::new(),
                        total_usage,
                        iterations: iteration + 1,
                        cost_usd: None,
                        silent: true,
                        directives: Default::default(),
                    });
                }

                let tool_results_msg = Message {
                    role: Role::User,
                    content: MessageContent::Blocks(tool_result_blocks.clone()),
                };
                session.messages.push(tool_results_msg.clone());
                messages.push(tool_results_msg);

                if let Err(e) = memory.save_session(session) {
                    warn!("Failed to interim-save session: {e}");
                }
            }
            StopReason::MaxTokens => {
                consecutive_max_tokens += 1;
                if consecutive_max_tokens >= MAX_CONTINUATIONS {
                    let text = response.text();
                    let text = if text.trim().is_empty() {
                        "[Partial response — token limit reached with no text output.]".to_string()
                    } else {
                        text
                    };
                    session.messages.push(Message::assistant(&text));
                    save_projected_session_best_effort(
                        memory,
                        session,
                        "Failed to save projected session on streaming max continuations",
                    );
                    warn!(
                        iteration,
                        consecutive_max_tokens,
                        "Max continuations reached (streaming), returning partial response"
                    );
                    // Fire AgentLoopEnd hook
                    if let Some(hook_reg) = hooks {
                        let ctx = crate::hooks::HookContext {
                            agent_name: &manifest.name,
                            agent_id: agent_id_str.as_str(),
                            event: openfang_types::agent::HookEvent::AgentLoopEnd,
                            data: serde_json::json!({
                                "iterations": iteration + 1,
                                "reason": "max_continuations",
                            }),
                        };
                        let _ = hook_reg.fire(&ctx);
                    }
                    return Ok(AgentLoopResult {
                        response: text,
                        total_usage,
                        iterations: iteration + 1,
                        cost_usd: None,
                        silent: false,
                        directives: Default::default(),
                    });
                }
                let text = response.text();
                session.messages.push(Message::assistant(&text));
                messages.push(Message::assistant(&text));
                session.messages.push(Message::user("Please continue."));
                messages.push(Message::user("Please continue."));
                warn!(iteration, "Max tokens hit (streaming), continuing");
            }
        }
    }

    save_projected_session_best_effort(
        memory,
        session,
        "Failed to save projected session on streaming max iterations",
    );

    // Fire AgentLoopEnd hook on max iterations exceeded
    if let Some(hook_reg) = hooks {
        let ctx = crate::hooks::HookContext {
            agent_name: &manifest.name,
            agent_id: agent_id_str.as_str(),
            event: openfang_types::agent::HookEvent::AgentLoopEnd,
            data: serde_json::json!({
                "reason": "max_iterations_exceeded",
                "iterations": max_iterations,
            }),
        };
        let _ = hook_reg.fire(&ctx);
    }

    Err(OpenFangError::MaxIterationsExceeded(max_iterations))
}

/// Recover tool calls that LLMs (Groq/Llama, DeepSeek) output as plain text
/// instead of the proper `tool_calls` API field.
///
/// Parses patterns like `<function=tool_name>{"key":"value"}</function>` from
/// the model's text output, validates tool names against the available tools,
/// and returns synthetic `ToolCall` entries.
fn recover_text_tool_calls(text: &str, available_tools: &[ToolDefinition]) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let tool_names: Vec<&str> = available_tools.iter().map(|t| t.name.as_str()).collect();

    // Pattern 1: <function=TOOL_NAME>JSON_BODY</function>
    let mut search_from = 0;
    while let Some(start) = text[search_from..].find("<function=") {
        let abs_start = search_from + start;
        let after_prefix = abs_start + "<function=".len();

        // Extract tool name (ends at '>')
        let Some(name_end) = text[after_prefix..].find('>') else {
            search_from = after_prefix;
            continue;
        };
        let tool_name = &text[after_prefix..after_prefix + name_end];
        let json_start = after_prefix + name_end + 1;

        // Find closing </function>
        let Some(close_offset) = text[json_start..].find("</function>") else {
            search_from = json_start;
            continue;
        };
        let json_body = text[json_start..json_start + close_offset].trim();
        search_from = json_start + close_offset + "</function>".len();

        // Validate: tool name must be in available_tools
        if !tool_names.contains(&tool_name) {
            warn!(
                tool = tool_name,
                "Text-based tool call for unknown tool — skipping"
            );
            continue;
        }

        // Parse JSON input
        let input: serde_json::Value = match serde_json::from_str(json_body) {
            Ok(v) => v,
            Err(e) => {
                warn!(tool = tool_name, error = %e, "Failed to parse text-based tool call JSON — skipping");
                continue;
            }
        };

        info!(
            tool = tool_name,
            "Recovered text-based tool call → synthetic ToolUse"
        );
        calls.push(ToolCall {
            id: format!("recovered_{}", uuid::Uuid::new_v4()),
            name: tool_name.to_string(),
            input,
        });
    }

    // Pattern 2: <function>TOOL_NAME{JSON_BODY}</function>
    // (Groq/Llama variant — tool name immediately followed by JSON object)
    search_from = 0;
    while let Some(start) = text[search_from..].find("<function>") {
        let abs_start = search_from + start;
        let after_tag = abs_start + "<function>".len();

        // Find closing </function>
        let Some(close_offset) = text[after_tag..].find("</function>") else {
            search_from = after_tag;
            continue;
        };
        let inner = &text[after_tag..after_tag + close_offset];
        search_from = after_tag + close_offset + "</function>".len();

        // The inner content is "tool_name{json}" — find the first '{' to split
        let Some(brace_pos) = inner.find('{') else {
            continue;
        };
        let tool_name = inner[..brace_pos].trim();
        let json_body = inner[brace_pos..].trim();

        if tool_name.is_empty() {
            continue;
        }

        // Validate: tool name must be in available_tools
        if !tool_names.contains(&tool_name) {
            warn!(
                tool = tool_name,
                "Text-based tool call (variant 2) for unknown tool — skipping"
            );
            continue;
        }

        // Parse JSON input
        let input: serde_json::Value = match serde_json::from_str(json_body) {
            Ok(v) => v,
            Err(e) => {
                warn!(tool = tool_name, error = %e, "Failed to parse text-based tool call JSON (variant 2) — skipping");
                continue;
            }
        };

        // Avoid duplicates if pattern 1 already captured this call
        if calls
            .iter()
            .any(|c| c.name == tool_name && c.input == input)
        {
            continue;
        }

        info!(
            tool = tool_name,
            "Recovered text-based tool call (variant 2) → synthetic ToolUse"
        );
        calls.push(ToolCall {
            id: format!("recovered_{}", uuid::Uuid::new_v4()),
            name: tool_name.to_string(),
            input,
        });
    }

    calls
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_driver::{CompletionResponse, LlmError};
    use async_trait::async_trait;
    use openfang_memory::session::Session;
    use openfang_types::agent::{AgentId, SessionId};
    use openfang_types::tool::ToolCall;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_max_iterations_constant() {
        assert_eq!(MAX_ITERATIONS, 50);
    }

    #[test]
    fn test_retry_constants() {
        assert_eq!(MAX_RETRIES, 3);
        assert_eq!(BASE_RETRY_DELAY_MS, 1000);
    }

    #[test]
    fn test_dynamic_truncate_short_unchanged() {
        use crate::context_budget::{truncate_tool_result_dynamic, ContextBudget};
        let budget = ContextBudget::new(200_000);
        let short = "Hello, world!";
        assert_eq!(truncate_tool_result_dynamic(short, &budget), short);
    }

    #[test]
    fn test_dynamic_truncate_over_limit() {
        use crate::context_budget::{truncate_tool_result_dynamic, ContextBudget};
        let budget = ContextBudget::new(200_000);
        let long = "x".repeat(budget.per_result_cap() + 10_000);
        let result = truncate_tool_result_dynamic(&long, &budget);
        assert!(result.len() <= budget.per_result_cap() + 200);
        assert!(result.contains("[TRUNCATED:"));
    }

    #[test]
    fn test_dynamic_truncate_newline_boundary() {
        use crate::context_budget::{truncate_tool_result_dynamic, ContextBudget};
        // Small budget to force truncation
        let budget = ContextBudget::new(1_000);
        let content = (0..200)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let result = truncate_tool_result_dynamic(&content, &budget);
        // Should break at a newline, not mid-line
        let before_marker = result.split("[TRUNCATED:").next().unwrap();
        let trimmed = before_marker.trim_end();
        assert!(!trimmed.is_empty());
    }

    #[test]
    fn test_max_continuations_constant() {
        assert_eq!(MAX_CONTINUATIONS, 5);
    }

    #[test]
    fn test_tool_timeout_selection() {
        assert_eq!(tool_timeout_secs("shell_exec"), DEFAULT_TOOL_TIMEOUT_SECS);
        assert_eq!(
            tool_timeout_secs("mcp_toolbox_generate_video"),
            DEFAULT_TOOL_TIMEOUT_SECS
        );
    }

    #[test]
    fn test_parse_channel_delivery_target_feishu_prefers_chat_id() {
        let message = "[Channel context]\nchannel: feishu\nchat_id: oc_123\nsender_open_id: ou_456\n\n[Conversation context]\n对方说：\nhi";
        let target = parse_channel_delivery_target(message).expect("expected delivery target");
        assert_eq!(target.channel, "feishu");
        assert_eq!(target.receive_id, "oc_123");
        assert_eq!(target.receive_id_type, "chat_id");
    }

    #[test]
    fn test_parse_channel_delivery_target_feishu_falls_back_to_open_id() {
        let message = "[Channel context]\nchannel: feishu\nsender_open_id: ou_456\n\n[Conversation context]\n对方说：\nhi";
        let target = parse_channel_delivery_target(message).expect("expected delivery target");
        assert_eq!(target.channel, "feishu");
        assert_eq!(target.receive_id, "ou_456");
        assert_eq!(target.receive_id_type, "open_id");
    }

    #[test]
    fn test_parse_channel_delivery_target_discord() {
        let message = "[Channel context]\nchannel: discord\nchat_id: 1234567890\n\n[Conversation context]\n对方说：\nhello";
        let target = parse_channel_delivery_target(message).expect("expected delivery target");
        assert_eq!(target.channel, "discord");
        assert_eq!(target.receive_id, "1234567890");
        assert_eq!(target.receive_id_type, "chat_id");
    }

    #[test]
    fn test_selfie_context_guard_blocks_missing_context_tools() {
        let manifest = AgentManifest {
            name: "assistant".to_string(),
            ..Default::default()
        };
        let message = "[Channel context]\nchannel: feishu\nchat_id: oc_123\nsender_open_id: ou_456\n\n[Conversation context]\n对方说：\n都9点了啊，给我看看你现在的样子吧";
        let tool_call = ToolCall {
            id: "tool_1".to_string(),
            name: "mcp_toolbox_generate_image".to_string(),
            input: serde_json::json!({
                "prompt": "写实真实照片，手机前置镜头自拍，早上刚醒。",
                "input_images": ["/home/ljj/.openfang/agents/assistant/avatar.png"]
            }),
        };

        let message = selfie_context_guard_message(&manifest, message, &tool_call, &HashSet::new())
            .expect("guard should block");
        assert!(message.contains("mcp_toolbox_get_life_status"));
        assert!(message.contains("mcp_toolbox_get_inner_state"));
        assert!(message.contains("mcp_toolbox_get_avatar"));
    }

    #[test]
    fn test_wardrobe_confirmation_guard_blocks_same_turn_confirmation() {
        let tool_call = ToolCall {
            id: "tool_1".to_string(),
            name: "mcp_toolbox_confirm_wardrobe_item".to_string(),
            input: serde_json::json!({
                "item_id": "abc123"
            }),
        };
        let successful_tools = HashSet::from(["mcp_toolbox_add_to_wardrobe".to_string()]);

        let message = wardrobe_confirmation_guard_message(
            "给你买了件衣服，试一下放进衣橱",
            &tool_call,
            &successful_tools,
        )
        .expect("guard should block same-turn confirmation");
        assert!(message.contains("刚生成完定妆照"));
    }

    #[test]
    fn test_wardrobe_confirmation_guard_blocks_without_explicit_confirmation() {
        let tool_call = ToolCall {
            id: "tool_1".to_string(),
            name: "mcp_toolbox_confirm_wardrobe_item".to_string(),
            input: serde_json::json!({
                "item_id": "abc123"
            }),
        };

        let message =
            wardrobe_confirmation_guard_message("再给我看看细节", &tool_call, &HashSet::new())
                .expect("guard should require explicit confirmation");
        assert!(message.contains("没有明确确认"));
    }

    #[test]
    fn test_wardrobe_confirmation_guard_allows_explicit_confirmation() {
        let tool_call = ToolCall {
            id: "tool_1".to_string(),
            name: "mcp_toolbox_confirm_wardrobe_item".to_string(),
            input: serde_json::json!({
                "item_id": "abc123"
            }),
        };

        assert!(wardrobe_confirmation_guard_message(
            "可以，就这件，放进衣橱吧",
            &tool_call,
            &HashSet::new()
        )
        .is_none());
    }

    #[test]
    fn test_wardrobe_confirmation_guard_allows_natural_acceptance_language() {
        let tool_call = ToolCall {
            id: "tool_1".to_string(),
            name: "mcp_toolbox_confirm_wardrobe_item".to_string(),
            input: serde_json::json!({
                "item_id": "abc123"
            }),
        };

        assert!(wardrobe_confirmation_guard_message(
            "参考图就当收下的凭据，我先把这件衣服正式入库",
            &tool_call,
            &HashSet::new()
        )
        .is_none());
    }

    fn test_session_with_messages(messages: Vec<Message>) -> Session {
        Session {
            id: SessionId(uuid::Uuid::nil()),
            agent_id: AgentId(uuid::Uuid::nil()),
            messages,
            context_window_tokens: 0,
            label: None,
        }
    }

    #[test]
    fn test_wardrobe_followup_phase_hint_after_confirmation() {
        let manifest = AgentManifest {
            name: "assistant".to_string(),
            ..Default::default()
        };
        let session = test_session_with_messages(vec![
            Message::assistant("你现在就回我一句：就这件，入库。你一说，我立刻换上给你录视频。"),
            Message::user("就这件，入库吧"),
            Message::assistant("好，就按这件定下来。"),
            Message::user("换上它，给我录吧"),
        ]);
        let user_message = "[Channel context]\nchannel: feishu\nchat_id: oc_123\nsender_open_id: ou_456\n\n[Conversation context]\n对方说：\n换上它，给我录吧";

        let hint = wardrobe_followup_phase_system_hint(&manifest, user_message, &session)
            .expect("confirmed wardrobe flow should add a follow-up hint");
        assert!(hint.contains("已经明确确认过了"));
        assert!(hint.contains("不要再说“我再发定妆照给你看一眼”"));
    }

    #[test]
    fn test_wardrobe_followup_phase_hint_accepts_natural_confirmation_language() {
        let manifest = AgentManifest {
            name: "assistant".to_string(),
            ..Default::default()
        };
        let session = test_session_with_messages(vec![
            Message::assistant("定妆照我这边也拍好了——你看一眼，觉得合适我就正式留档入库。"),
            Message::user("参考图就当收下的凭据，我先把这件衣服正式入库"),
            Message::assistant("好，我就按收下了继续。"),
            Message::user("换上它，给我录吧"),
        ]);
        let user_message = "[Channel context]\nchannel: feishu\nchat_id: oc_123\nsender_open_id: ou_456\n\n[Conversation context]\n对方说：\n换上它，给我录吧";

        let hint = wardrobe_followup_phase_system_hint(&manifest, user_message, &session)
            .expect("natural confirmation language should still count as confirmation");
        assert!(hint.contains("已经明确确认过了"));
        assert!(hint.contains("还没拿到成片文件"));
    }

    #[test]
    fn test_wardrobe_followup_phase_hint_ignores_unrelated_confirmation() {
        let manifest = AgentManifest {
            name: "assistant".to_string(),
            ..Default::default()
        };
        let session = test_session_with_messages(vec![
            Message::assistant("晚饭你想吃哪家？"),
            Message::user("就这家吧"),
            Message::assistant("行，那我记住了。"),
        ]);
        let user_message = "[Channel context]\nchannel: feishu\nchat_id: oc_123\nsender_open_id: ou_456\n\n[Conversation context]\n对方说：\n那你先忙";

        assert!(wardrobe_followup_phase_system_hint(&manifest, user_message, &session).is_none());
    }

    #[test]
    fn test_positive_wardrobe_followup_message_accepts_praise_and_wear_request() {
        assert!(is_positive_wardrobe_followup_message("真好看，换上吧"));
        assert!(is_positive_wardrobe_followup_message("就它，给我拍一张"));
    }

    #[test]
    fn test_positive_wardrobe_followup_message_rejects_negative_feedback() {
        assert!(!is_positive_wardrobe_followup_message("不喜欢，换一件"));
    }

    #[test]
    fn test_pick_latest_pending_wardrobe_item_prefers_newest_pending() {
        let list_result = serde_json::json!({
            "items": [
                {
                    "item_id": "old",
                    "name": "旧稿",
                    "status": "pending",
                    "created_at": "2026-03-12T17:00:00Z"
                },
                {
                    "item_id": "confirmed",
                    "name": "已确认",
                    "status": "confirmed",
                    "created_at": "2026-03-12T17:05:00Z"
                },
                {
                    "item_id": "new",
                    "name": "新稿",
                    "status": "pending",
                    "created_at": "2026-03-12T17:10:00Z"
                }
            ]
        })
        .to_string();

        let picked = pick_latest_pending_wardrobe_item(&list_result)
            .expect("latest pending wardrobe item should be selected");
        assert_eq!(picked.0, "new");
        assert_eq!(picked.1, "新稿");
    }

    #[test]
    fn test_selfie_context_guard_allows_when_context_is_ready() {
        let manifest = AgentManifest {
            name: "assistant".to_string(),
            ..Default::default()
        };
        let message = "[Channel context]\nchannel: feishu\nchat_id: oc_123\nsender_open_id: ou_456\n\n[Conversation context]\n对方说：\n拍张自拍给我";
        let tool_call = ToolCall {
            id: "tool_1".to_string(),
            name: "mcp_toolbox_generate_image".to_string(),
            input: serde_json::json!({
                "prompt": "写实真实照片，手机前置镜头自拍，窗边自然光。",
                "input_images": ["/home/ljj/.openfang/agents/assistant/avatar.png"]
            }),
        };
        let mut successful_tools = HashSet::new();
        successful_tools.insert("mcp_toolbox_get_life_status".to_string());
        successful_tools.insert("mcp_toolbox_get_inner_state".to_string());
        successful_tools.insert("mcp_toolbox_get_avatar".to_string());

        assert!(
            selfie_context_guard_message(&manifest, message, &tool_call, &successful_tools)
                .is_none()
        );
    }

    #[test]
    fn test_selfie_context_guard_ignores_non_selfie_image_requests() {
        let manifest = AgentManifest {
            name: "assistant".to_string(),
            ..Default::default()
        };
        let message = "[Channel context]\nchannel: feishu\nchat_id: oc_123\nsender_open_id: ou_456\n\n[Conversation context]\n对方说：\n画一只猫";
        let tool_call = ToolCall {
            id: "tool_1".to_string(),
            name: "mcp_toolbox_generate_image".to_string(),
            input: serde_json::json!({
                "prompt": "一只橘猫坐在窗台上。",
            }),
        };

        assert!(
            selfie_context_guard_message(&manifest, message, &tool_call, &HashSet::new()).is_none()
        );
    }

    #[test]
    fn test_async_media_plan_for_selfie_photo_archives_and_auto_sends() {
        let manifest = AgentManifest {
            name: "assistant".to_string(),
            ..Default::default()
        };
        let message = "[Channel context]\nchannel: feishu\nchat_id: oc_123\nsender_open_id: ou_456\n\n[Conversation context]\n对方说：\n拍张自拍给我";
        let tool_call = ToolCall {
            id: "tool_1".to_string(),
            name: "mcp_toolbox_generate_image".to_string(),
            input: serde_json::json!({
                "prompt": "写实真实照片，手机前置镜头自拍，窗边自然光。",
                "input_images": ["/home/ljj/.openfang/agents/assistant/avatar.png"]
            }),
        };

        let plan = build_async_media_plan(&manifest, &tool_call, message, "agent_1")
            .expect("selfie image should use async media plan");

        assert_eq!(plan.request.tool_name, "mcp_toolbox_generate_image");
        assert_eq!(plan.request.result_path_key, "image_path");
        assert_eq!(plan.request.send_tool_name, "mcp_feishu_send_image_message");
        assert_eq!(plan.request.channel, "feishu");
        assert!(plan.request.failure_notice.contains("重新拍一张"));
        assert!(plan.request.archive.is_some());
    }

    #[test]
    fn test_async_media_plan_for_general_image_does_not_archive() {
        let manifest = AgentManifest {
            name: "assistant".to_string(),
            ..Default::default()
        };
        let message = "[Channel context]\nchannel: feishu\nchat_id: oc_123\nsender_open_id: ou_456\n\n[Conversation context]\n对方说：\n画一只猫";
        let tool_call = ToolCall {
            id: "tool_1".to_string(),
            name: "mcp_toolbox_generate_image".to_string(),
            input: serde_json::json!({
                "prompt": "一只橘猫坐在窗台上。"
            }),
        };

        let plan = build_async_media_plan(&manifest, &tool_call, message, "agent_1")
            .expect("general image should use async media plan");

        assert_eq!(plan.request.result_path_key, "image_path");
        assert!(plan.request.archive.is_none());
    }

    #[test]
    fn test_async_media_plan_for_wardrobe_preview_uses_preview_path() {
        let manifest = AgentManifest {
            name: "assistant".to_string(),
            ..Default::default()
        };
        let message = "[Channel context]\nchannel: feishu\nchat_id: oc_123\nsender_open_id: ou_456\n\n[Conversation context]\n对方说：\n给你买了件衣服";
        let tool_call = ToolCall {
            id: "tool_1".to_string(),
            name: "mcp_toolbox_add_to_wardrobe".to_string(),
            input: serde_json::json!({
                "name": "灰色针织套装",
                "tags": ["home"],
                "outfit_prompt": "柔软的灰色针织开衫和长裤。"
            }),
        };

        let plan = build_async_media_plan(&manifest, &tool_call, message, "agent_1")
            .expect("wardrobe preview should use async media plan");

        assert_eq!(plan.request.tool_name, "mcp_toolbox_add_to_wardrobe");
        assert_eq!(plan.request.result_path_key, "preview_path");
        assert_eq!(plan.request.send_tool_name, "mcp_feishu_send_image_message");
        assert_eq!(plan.request.channel, "feishu");
        assert!(plan.request.failure_notice.contains("定妆照这一下没出成"));
        assert!(plan.request.archive.is_none());
    }

    #[test]
    fn test_max_history_messages() {
        assert_eq!(MAX_HISTORY_MESSAGES, 20);
    }

    fn assert_no_execution_trace(messages: &[Message]) {
        for message in messages {
            if let MessageContent::Blocks(blocks) = &message.content {
                for block in blocks {
                    assert!(
                        !matches!(
                            block,
                            ContentBlock::ToolUse { .. }
                                | ContentBlock::ToolResult { .. }
                                | ContentBlock::Thinking { .. }
                        ),
                        "execution trace should not persist in session messages"
                    );
                }
            }
        }
    }

    fn assert_persisted_session_has_no_execution_trace(
        memory: &openfang_memory::MemorySubstrate,
        session_id: openfang_types::agent::SessionId,
    ) {
        let persisted = memory
            .get_session(session_id)
            .expect("session lookup should succeed")
            .expect("session should be persisted");
        assert_no_execution_trace(&persisted.messages);
    }

    // --- Integration tests for empty response guards ---

    fn test_manifest() -> AgentManifest {
        AgentManifest {
            name: "test-agent".to_string(),
            model: openfang_types::agent::ModelConfig {
                system_prompt: "You are a test agent.".to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Mock driver that simulates: first call returns ToolUse with no text,
    /// second call returns EndTurn with empty text. This reproduces the bug
    /// where the LLM ends with no text after a tool-use cycle.
    struct EmptyAfterToolUseDriver {
        call_count: AtomicU32,
    }

    impl EmptyAfterToolUseDriver {
        fn new() -> Self {
            Self {
                call_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl LlmDriver for EmptyAfterToolUseDriver {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            let call = self.call_count.fetch_add(1, Ordering::Relaxed);
            if call == 0 {
                // First call: LLM wants to use a tool (with no text block)
                Ok(CompletionResponse {
                    content: vec![ContentBlock::ToolUse {
                        id: "tool_1".to_string(),
                        name: "fake_tool".to_string(),
                        input: serde_json::json!({"query": "test"}),
                    }],
                    stop_reason: StopReason::ToolUse,
                    tool_calls: vec![ToolCall {
                        id: "tool_1".to_string(),
                        name: "fake_tool".to_string(),
                        input: serde_json::json!({"query": "test"}),
                    }],
                    usage: TokenUsage {
                        input_tokens: 10,
                        output_tokens: 5,
                    },
                })
            } else {
                // Second call: LLM returns EndTurn with EMPTY text (the bug)
                Ok(CompletionResponse {
                    content: vec![],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage {
                        input_tokens: 10,
                        output_tokens: 0,
                    },
                })
            }
        }
    }

    /// Mock driver that requests a tool first, then fails the next LLM turn.
    struct ErrorAfterToolUseDriver {
        call_count: AtomicU32,
    }

    impl ErrorAfterToolUseDriver {
        fn new() -> Self {
            Self {
                call_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl LlmDriver for ErrorAfterToolUseDriver {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            let call = self.call_count.fetch_add(1, Ordering::Relaxed);
            if call == 0 {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::ToolUse {
                        id: "tool_1".to_string(),
                        name: "fake_tool".to_string(),
                        input: serde_json::json!({"query": "test"}),
                    }],
                    stop_reason: StopReason::ToolUse,
                    tool_calls: vec![ToolCall {
                        id: "tool_1".to_string(),
                        name: "fake_tool".to_string(),
                        input: serde_json::json!({"query": "test"}),
                    }],
                    usage: TokenUsage {
                        input_tokens: 10,
                        output_tokens: 5,
                    },
                })
            } else {
                Err(LlmError::Http("simulated llm failure".to_string()))
            }
        }
    }

    /// Mock driver that returns empty text with MaxTokens stop reason,
    /// repeated MAX_CONTINUATIONS times to trigger the max continuations path.
    struct EmptyMaxTokensDriver;

    #[async_trait]
    impl LlmDriver for EmptyMaxTokensDriver {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            Ok(CompletionResponse {
                content: vec![],
                stop_reason: StopReason::MaxTokens,
                tool_calls: vec![],
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 0,
                },
            })
        }
    }

    /// Mock driver that returns normal text (sanity check).
    struct NormalDriver;

    #[async_trait]
    impl LlmDriver for NormalDriver {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Hello from the agent!".to_string(),
                }],
                stop_reason: StopReason::EndTurn,
                tool_calls: vec![],
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 8,
                },
            })
        }
    }

    #[tokio::test]
    async fn test_empty_response_after_tool_use_returns_fallback() {
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(EmptyAfterToolUseDriver::new());

        let result = run_agent_loop(
            &manifest,
            "Do something with tools",
            &mut session,
            &memory,
            driver,
            &[], // no tools registered — the tool call will fail, which is fine
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // on_phase
            None, // media_engine
            None, // tts_engine
            None, // docker_config
            None, // hooks
            None, // context_window_tokens
            None, // process_manager
            vec![], // media_blocks
        )
        .await
        .expect("Loop should complete without error");

        // The response MUST NOT be empty — it should contain our fallback text
        assert!(
            !result.response.trim().is_empty(),
            "Response should not be empty after tool use, got: {:?}",
            result.response
        );
        assert!(
            result.response.contains("Task completed"),
            "Expected fallback message, got: {:?}",
            result.response
        );
        assert_no_execution_trace(&session.messages);
    }

    #[tokio::test]
    async fn test_empty_response_max_tokens_returns_fallback() {
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(EmptyMaxTokensDriver);

        let result = run_agent_loop(
            &manifest,
            "Tell me something long",
            &mut session,
            &memory,
            driver,
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // on_phase
            None, // media_engine
            None, // tts_engine
            None, // docker_config
            None, // hooks
            None, // context_window_tokens
            None, // process_manager
            vec![], // media_blocks
        )
        .await
        .expect("Loop should complete without error");

        // Should hit MAX_CONTINUATIONS and return fallback instead of empty
        assert!(
            !result.response.trim().is_empty(),
            "Response should not be empty on max tokens, got: {:?}",
            result.response
        );
        assert!(
            result.response.contains("token limit"),
            "Expected max-tokens fallback message, got: {:?}",
            result.response
        );
        assert_no_execution_trace(&session.messages);
    }

    #[tokio::test]
    async fn test_normal_response_not_replaced_by_fallback() {
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(NormalDriver);

        let result = run_agent_loop(
            &manifest,
            "Say hello",
            &mut session,
            &memory,
            driver,
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // on_phase
            None, // media_engine
            None, // tts_engine
            None, // docker_config
            None, // hooks
            None, // context_window_tokens
            None, // process_manager
            vec![], // media_blocks
        )
        .await
        .expect("Loop should complete without error");

        // Normal response should pass through unchanged
        assert_eq!(result.response, "Hello from the agent!");
        assert_no_execution_trace(&session.messages);
    }

    #[tokio::test]
    async fn test_streaming_empty_response_after_tool_use_returns_fallback() {
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(EmptyAfterToolUseDriver::new());
        let (tx, _rx) = mpsc::channel(64);

        let result = run_agent_loop_streaming(
            &manifest,
            "Do something with tools",
            &mut session,
            &memory,
            driver,
            &[],
            None,
            tx,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // on_phase
            None, // media_engine
            None, // tts_engine
            None, // docker_config
            None, // hooks
            None, // context_window_tokens
            None, // process_manager
            vec![], // media_blocks
        )
        .await
        .expect("Streaming loop should complete without error");

        assert!(
            !result.response.trim().is_empty(),
            "Streaming response should not be empty after tool use, got: {:?}",
            result.response
        );
        assert!(
            result.response.contains("Task completed"),
            "Expected fallback message in streaming, got: {:?}",
            result.response
        );
        assert_no_execution_trace(&session.messages);
        assert_persisted_session_has_no_execution_trace(&memory, session.id);
    }

    #[tokio::test]
    async fn test_streaming_llm_error_after_tool_use_compacts_persisted_session() {
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(ErrorAfterToolUseDriver::new());
        let (tx, _rx) = mpsc::channel(64);

        let error = run_agent_loop_streaming(
            &manifest,
            "Do something with tools",
            &mut session,
            &memory,
            driver,
            &[],
            None,
            tx,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            vec![], // media_blocks
        )
        .await
        .expect_err("Streaming loop should return the simulated LLM failure");

        assert!(
            matches!(error, OpenFangError::LlmDriver(_)),
            "Expected LLM driver error, got: {error:?}"
        );
        assert_no_execution_trace(&session.messages);
        assert_persisted_session_has_no_execution_trace(&memory, session.id);
    }

    #[tokio::test]
    async fn test_existing_execution_trace_is_compacted_before_next_turn() {
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: vec![
                Message::user("上一轮用户消息"),
                Message {
                    role: Role::Assistant,
                    content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                        id: "call_1".to_string(),
                        name: "fake_tool".to_string(),
                        input: serde_json::json!({"query": "demo"}),
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
            ],
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(NormalDriver);

        let result = run_agent_loop(
            &manifest,
            "新的一轮",
            &mut session,
            &memory,
            driver,
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            vec![], // media_blocks
        )
        .await
        .expect("Loop should complete without error");

        assert_eq!(result.response, "Hello from the agent!");
        assert_no_execution_trace(&session.messages);
        let all_text: String = session
            .messages
            .iter()
            .map(|m| m.content.text_content())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(all_text.contains("上一轮用户消息"));
        assert!(all_text.contains("新的一轮"));
    }

    /// Mock driver that returns empty text on first call (EndTurn), then normal text on second.
    /// This tests the one-shot retry logic for iteration 0 empty responses.
    struct EmptyThenNormalDriver {
        call_count: AtomicU32,
    }

    impl EmptyThenNormalDriver {
        fn new() -> Self {
            Self {
                call_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl LlmDriver for EmptyThenNormalDriver {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            let call = self.call_count.fetch_add(1, Ordering::Relaxed);
            if call == 0 {
                // First call: empty EndTurn (triggers retry)
                Ok(CompletionResponse {
                    content: vec![],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage {
                        input_tokens: 10,
                        output_tokens: 0,
                    },
                })
            } else {
                // Second call (retry): normal response
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Recovered after retry!".to_string(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage {
                        input_tokens: 15,
                        output_tokens: 8,
                    },
                })
            }
        }
    }

    /// Mock driver that always returns empty EndTurn (no recovery on retry).
    /// Tests that the fallback message appears when retry also fails.
    struct AlwaysEmptyDriver;

    #[async_trait]
    impl LlmDriver for AlwaysEmptyDriver {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            Ok(CompletionResponse {
                content: vec![],
                stop_reason: StopReason::EndTurn,
                tool_calls: vec![],
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 0,
                },
            })
        }
    }

    #[tokio::test]
    async fn test_empty_first_response_retries_and_recovers() {
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(EmptyThenNormalDriver::new());

        let result = run_agent_loop(
            &manifest,
            "Hello",
            &mut session,
            &memory,
            driver,
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // context_window_tokens
            None, // process_manager
            vec![], // media_blocks
        )
        .await
        .expect("Loop should recover via retry");

        assert_eq!(result.response, "Recovered after retry!");
        assert_eq!(
            result.iterations, 2,
            "Should have taken 2 iterations (retry)"
        );
    }

    #[tokio::test]
    async fn test_empty_first_response_fallback_when_retry_also_empty() {
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(AlwaysEmptyDriver);

        let result = run_agent_loop(
            &manifest,
            "Hello",
            &mut session,
            &memory,
            driver,
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // context_window_tokens
            None, // process_manager
            vec![], // media_blocks
        )
        .await
        .expect("Loop should complete with fallback");

        // After retry (iteration 1), should hit the iteration > 0 guard
        assert!(
            result.response.contains("Task completed"),
            "Expected fallback after retry failure, got: {:?}",
            result.response
        );
    }

    #[tokio::test]
    async fn test_max_history_messages_constant() {
        assert_eq!(MAX_HISTORY_MESSAGES, 20);
    }

    #[tokio::test]
    async fn test_streaming_empty_response_max_tokens_returns_fallback() {
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(EmptyMaxTokensDriver);
        let (tx, _rx) = mpsc::channel(64);

        let result = run_agent_loop_streaming(
            &manifest,
            "Tell me something long",
            &mut session,
            &memory,
            driver,
            &[],
            None,
            tx,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // on_phase
            None, // media_engine
            None, // tts_engine
            None, // docker_config
            None, // hooks
            None, // context_window_tokens
            None, // process_manager
            vec![], // media_blocks
        )
        .await
        .expect("Streaming loop should complete without error");

        assert!(
            !result.response.trim().is_empty(),
            "Streaming response should not be empty on max tokens, got: {:?}",
            result.response
        );
        assert!(
            result.response.contains("token limit"),
            "Expected max-tokens fallback in streaming, got: {:?}",
            result.response
        );
        assert_no_execution_trace(&session.messages);
        assert_persisted_session_has_no_execution_trace(&memory, session.id);
    }

    #[test]
    fn test_recover_text_tool_calls_basic() {
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({}),
        }];
        let text =
            r#"Let me search for that. <function=web_search>{"query":"rust async"}</function>"#;
        let calls = recover_text_tool_calls(text, &tools);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].input["query"], "rust async");
        assert!(calls[0].id.starts_with("recovered_"));
    }

    #[test]
    fn test_recover_text_tool_calls_unknown_tool() {
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({}),
        }];
        let text = r#"<function=hack_system>{"cmd":"rm -rf /"}</function>"#;
        let calls = recover_text_tool_calls(text, &tools);
        assert!(calls.is_empty(), "Unknown tools should be rejected");
    }

    #[test]
    fn test_recover_text_tool_calls_invalid_json() {
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({}),
        }];
        let text = r#"<function=web_search>not valid json</function>"#;
        let calls = recover_text_tool_calls(text, &tools);
        assert!(calls.is_empty(), "Invalid JSON should be skipped");
    }

    #[test]
    fn test_recover_text_tool_calls_multiple() {
        let tools = vec![
            ToolDefinition {
                name: "web_search".into(),
                description: "Search".into(),
                input_schema: serde_json::json!({}),
            },
            ToolDefinition {
                name: "read_file".into(),
                description: "Read a file".into(),
                input_schema: serde_json::json!({}),
            },
        ];
        let text = r#"<function=web_search>{"query":"hello"}</function> then <function=read_file>{"path":"a.txt"}</function>"#;
        let calls = recover_text_tool_calls(text, &tools);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[1].name, "read_file");
    }

    #[test]
    fn test_recover_text_tool_calls_no_pattern() {
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({}),
        }];
        let text = "Just a normal response with no tool calls.";
        let calls = recover_text_tool_calls(text, &tools);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_recover_text_tool_calls_empty_tools() {
        let text = r#"<function=web_search>{"query":"hello"}</function>"#;
        let calls = recover_text_tool_calls(text, &[]);
        assert!(calls.is_empty(), "No tools = no recovery");
    }

    // --- Deep edge-case tests for text-to-tool recovery ---

    #[test]
    fn test_recover_text_tool_calls_nested_json() {
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({}),
        }];
        let text = r#"<function=web_search>{"query":"rust","filters":{"lang":"en","year":2024}}</function>"#;
        let calls = recover_text_tool_calls(text, &tools);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].input["filters"]["lang"], "en");
    }

    #[test]
    fn test_recover_text_tool_calls_with_surrounding_text() {
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({}),
        }];
        let text = "Sure, let me search that for you.\n\n<function=web_search>{\"query\":\"rust async programming\"}</function>\n\nI'll get back to you with results.";
        let calls = recover_text_tool_calls(text, &tools);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].input["query"], "rust async programming");
    }

    #[test]
    fn test_recover_text_tool_calls_whitespace_in_json() {
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({}),
        }];
        // Some models emit pretty-printed JSON
        let text = "<function=web_search>\n  {\"query\": \"hello world\"}\n</function>";
        let calls = recover_text_tool_calls(text, &tools);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].input["query"], "hello world");
    }

    #[test]
    fn test_recover_text_tool_calls_unclosed_tag() {
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({}),
        }];
        // Missing </function> — should gracefully skip
        let text = r#"<function=web_search>{"query":"test"}"#;
        let calls = recover_text_tool_calls(text, &tools);
        assert!(calls.is_empty(), "Unclosed tag should be skipped");
    }

    #[test]
    fn test_recover_text_tool_calls_missing_closing_bracket() {
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({}),
        }];
        // Missing > after tool name
        let text = r#"<function=web_search{"query":"test"}</function>"#;
        let calls = recover_text_tool_calls(text, &tools);
        // The parser finds > inside JSON, will likely produce invalid tool name
        // or invalid JSON — either way, should not panic
        // (just verifying no panic / no bad behavior)
        let _ = calls;
    }

    #[test]
    fn test_recover_text_tool_calls_empty_json_object() {
        let tools = vec![ToolDefinition {
            name: "list_files".into(),
            description: "List".into(),
            input_schema: serde_json::json!({}),
        }];
        let text = r#"<function=list_files>{}</function>"#;
        let calls = recover_text_tool_calls(text, &tools);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "list_files");
        assert_eq!(calls[0].input, serde_json::json!({}));
    }

    #[test]
    fn test_recover_text_tool_calls_mixed_valid_invalid() {
        let tools = vec![
            ToolDefinition {
                name: "web_search".into(),
                description: "Search".into(),
                input_schema: serde_json::json!({}),
            },
            ToolDefinition {
                name: "read_file".into(),
                description: "Read".into(),
                input_schema: serde_json::json!({}),
            },
        ];
        // First: valid, second: unknown tool, third: valid
        let text = r#"<function=web_search>{"q":"a"}</function> <function=unknown>{"x":1}</function> <function=read_file>{"path":"b"}</function>"#;
        let calls = recover_text_tool_calls(text, &tools);
        assert_eq!(calls.len(), 2, "Should recover 2 valid, skip 1 unknown");
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[1].name, "read_file");
    }

    // --- Variant 2 pattern tests: <function>NAME{JSON}</function> ---

    #[test]
    fn test_recover_variant2_basic() {
        let tools = vec![ToolDefinition {
            name: "web_fetch".into(),
            description: "Fetch".into(),
            input_schema: serde_json::json!({}),
        }];
        let text = r#"<function>web_fetch{"url":"https://example.com"}</function>"#;
        let calls = recover_text_tool_calls(text, &tools);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_fetch");
        assert_eq!(calls[0].input["url"], "https://example.com");
    }

    #[test]
    fn test_recover_variant2_unknown_tool() {
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({}),
        }];
        let text = r#"<function>unknown_tool{"q":"test"}</function>"#;
        let calls = recover_text_tool_calls(text, &tools);
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_recover_variant2_with_surrounding_text() {
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({}),
        }];
        let text = r#"Let me search for that. <function>web_search{"query":"rust lang"}</function> I'll find the answer."#;
        let calls = recover_text_tool_calls(text, &tools);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
    }

    #[test]
    fn test_recover_both_variants_mixed() {
        let tools = vec![
            ToolDefinition {
                name: "web_search".into(),
                description: "Search".into(),
                input_schema: serde_json::json!({}),
            },
            ToolDefinition {
                name: "web_fetch".into(),
                description: "Fetch".into(),
                input_schema: serde_json::json!({}),
            },
        ];
        // Mix of variant 1 and variant 2
        let text = r#"<function=web_search>{"q":"a"}</function> <function>web_fetch{"url":"https://x.com"}</function>"#;
        let calls = recover_text_tool_calls(text, &tools);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[1].name, "web_fetch");
    }

    // --- End-to-end integration test: text-as-tool-call recovery through agent loop ---

    /// Mock driver that simulates a Groq/Llama model outputting tool calls as text.
    /// Call 1: Returns text with `<function=web_search>...</function>` (EndTurn, no tool_calls)
    /// Call 2: Returns a normal text response (after tool result is provided)
    struct TextToolCallDriver {
        call_count: AtomicU32,
    }

    impl TextToolCallDriver {
        fn new() -> Self {
            Self {
                call_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl LlmDriver for TextToolCallDriver {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            let call = self.call_count.fetch_add(1, Ordering::Relaxed);
            if call == 0 {
                // Simulate Groq/Llama: tool call as text, not in tool_calls field
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: r#"Let me search for that. <function=web_search>{"query":"rust async"}</function>"#.to_string(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![], // BUG: no tool_calls!
                    usage: TokenUsage {
                        input_tokens: 20,
                        output_tokens: 15,
                    },
                })
            } else {
                // After tool result, return normal response
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Based on the search results, Rust async is great!".to_string(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage {
                        input_tokens: 30,
                        output_tokens: 12,
                    },
                })
            }
        }
    }

    #[tokio::test]
    async fn test_text_tool_call_recovery_e2e() {
        // This is THE critical test: a model outputs a tool call as text,
        // the recovery code detects it, promotes it to ToolUse, executes the tool,
        // and the agent loop continues to produce a final response.
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(TextToolCallDriver::new());

        // Provide web_search as an available tool so recovery can match it
        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
        }];

        let result = run_agent_loop(
            &manifest,
            "Search for rust async programming",
            &mut session,
            &memory,
            driver,
            &tools,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // on_phase
            None, // media_engine
            None, // tts_engine
            None, // docker_config
            None, // hooks
            None, // context_window_tokens
            None, // process_manager
            vec![], // media_blocks
        )
        .await
        .expect("Agent loop should complete");

        // The response should contain the second call's output, NOT the raw function tag
        assert!(
            !result.response.contains("<function="),
            "Response should not contain raw function tags, got: {:?}",
            result.response
        );
        assert!(
            result.iterations >= 2,
            "Should have at least 2 iterations (tool call + final response), got: {}",
            result.iterations
        );
        // Verify the final text response came through
        assert!(
            result.response.contains("search results") || result.response.contains("Rust async"),
            "Expected final response text, got: {:?}",
            result.response
        );
    }

    /// Mock driver that returns NO text-based tool calls — just normal text.
    /// Verifies recovery does NOT interfere with normal flow.
    #[tokio::test]
    async fn test_normal_flow_unaffected_by_recovery() {
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(NormalDriver);

        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({}),
        }];

        let result = run_agent_loop(
            &manifest,
            "Say hello",
            &mut session,
            &memory,
            driver,
            &tools, // tools available but not used
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            vec![], // media_blocks
        )
        .await
        .expect("Normal loop should complete");

        assert_eq!(result.response, "Hello from the agent!");
        assert_eq!(
            result.iterations, 1,
            "Normal response should complete in 1 iteration"
        );
    }

    // --- Streaming path: text-as-tool-call recovery ---

    #[tokio::test]
    async fn test_text_tool_call_recovery_streaming_e2e() {
        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = openfang_types::agent::AgentId::new();
        let mut session = openfang_memory::session::Session {
            id: openfang_types::agent::SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let driver: Arc<dyn LlmDriver> = Arc::new(TextToolCallDriver::new());

        let tools = vec![ToolDefinition {
            name: "web_search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
        }];

        let (tx, mut rx) = mpsc::channel(64);

        let result = run_agent_loop_streaming(
            &manifest,
            "Search for rust async programming",
            &mut session,
            &memory,
            driver,
            &tools,
            None,
            tx,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // on_phase
            None, // media_engine
            None, // tts_engine
            None, // docker_config
            None, // hooks
            None, // context_window_tokens
            None, // process_manager
            vec![], // media_blocks
        )
        .await
        .expect("Streaming loop should complete");

        // Same assertions as non-streaming
        assert!(
            !result.response.contains("<function="),
            "Streaming: response should not contain raw function tags, got: {:?}",
            result.response
        );
        assert!(
            result.iterations >= 2,
            "Streaming: should have at least 2 iterations, got: {}",
            result.iterations
        );

        // Drain the stream channel to verify events were sent
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        assert!(!events.is_empty(), "Should have received stream events");
        assert_no_execution_trace(&session.messages);
        assert_persisted_session_has_no_execution_trace(&memory, session.id);
    }

    // ── Dynamic injection tests ─────────────────────────────────────

    /// Mock LLM driver that captures the CompletionRequest for inspection.
    struct RequestCapturingDriver {
        captured: std::sync::Mutex<Vec<CompletionRequest>>,
    }

    impl RequestCapturingDriver {
        fn new() -> Self {
            Self {
                captured: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn captured_requests(&self) -> Vec<CompletionRequest> {
            self.captured.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl LlmDriver for RequestCapturingDriver {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            self.captured.lock().unwrap().push(request);
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "OK".to_string(),
                }],
                stop_reason: StopReason::EndTurn,
                tool_calls: vec![],
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 2,
                },
            })
        }
    }

    #[tokio::test]
    async fn test_dynamic_injection_inserts_assistant_message() {
        use crate::tool_runner::{DynamicInjection, InjectionPosition, DYNAMIC_INJECTIONS};

        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = AgentId::new();
        let mut session = Session {
            id: SessionId::new(),
            agent_id,
            messages: vec![
                Message::user("previous question"),
                Message::assistant("previous answer"),
            ],
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let capturing = Arc::new(RequestCapturingDriver::new());
        let driver: Arc<dyn LlmDriver> = capturing.clone();

        // Run with a dynamic injection set
        let _result = DYNAMIC_INJECTIONS
            .scope(
                std::cell::RefCell::new(vec![DynamicInjection {
                    content: "[此刻的世界]\n窗外天色暗下来。".to_string(),
                    position: InjectionPosition::InsertAssistant { offset_from_last: 0 },
                }]),
                async {
                    run_agent_loop(
                        &manifest,
                        "hello",
                        &mut session,
                        &memory,
                        driver,
                        &[],
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        vec![],
                    )
                    .await
                },
            )
            .await
            .expect("Loop should complete");

        let requests = capturing.captured_requests();
        assert!(!requests.is_empty(), "Should have at least one LLM call");
        let msgs = &requests[0].messages;

        // The injected assistant message should be the penultimate message
        // (before the last user message "hello")
        assert!(msgs.len() >= 3, "Should have at least 3 messages, got {}", msgs.len());
        let last = &msgs[msgs.len() - 1];
        assert_eq!(last.role, Role::User);
        assert!(last.content.text_content().contains("hello"));

        // The injected world state should be in the preceding assistant message(s)
        // (may be merged with previous assistant message)
        let second_to_last = &msgs[msgs.len() - 2];
        assert_eq!(second_to_last.role, Role::Assistant);
        let text = second_to_last.content.text_content();
        assert!(
            text.contains("[此刻的世界]"),
            "Injected content should be in assistant message, got: {text}"
        );
    }

    #[tokio::test]
    async fn test_no_injection_leaves_messages_unchanged() {
        use crate::tool_runner::DYNAMIC_INJECTIONS;

        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = AgentId::new();
        let mut session = Session {
            id: SessionId::new(),
            agent_id,
            messages: vec![],
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let capturing = Arc::new(RequestCapturingDriver::new());
        let driver: Arc<dyn LlmDriver> = capturing.clone();

        // Run with empty DYNAMIC_INJECTIONS
        let _result = DYNAMIC_INJECTIONS
            .scope(std::cell::RefCell::new(Vec::new()), async {
                run_agent_loop(
                    &manifest,
                    "hello",
                    &mut session,
                    &memory,
                    driver,
                    &[],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    vec![],
                )
                .await
            })
            .await
            .expect("Loop should complete");

        let requests = capturing.captured_requests();
        assert!(!requests.is_empty());
        let msgs = &requests[0].messages;

        // Should just be: user("hello")
        assert_eq!(msgs.len(), 1, "Should have exactly 1 message, got {}", msgs.len());
        assert_eq!(msgs[0].role, Role::User);
    }

    #[tokio::test]
    async fn test_consecutive_assistant_messages_merged() {
        use crate::tool_runner::{DynamicInjection, InjectionPosition, DYNAMIC_INJECTIONS};

        let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
        let agent_id = AgentId::new();
        let mut session = Session {
            id: SessionId::new(),
            agent_id,
            messages: vec![
                Message::user("question"),
                Message::assistant("previous reply"),
            ],
            context_window_tokens: 0,
            label: None,
        };
        let manifest = test_manifest();
        let capturing = Arc::new(RequestCapturingDriver::new());
        let driver: Arc<dyn LlmDriver> = capturing.clone();

        // Inject world state — this will be consecutive with "previous reply" assistant msg
        let _result = DYNAMIC_INJECTIONS
            .scope(
                std::cell::RefCell::new(vec![DynamicInjection {
                    content: "[此刻的世界]\nworld state here".to_string(),
                    position: InjectionPosition::InsertAssistant { offset_from_last: 0 },
                }]),
                async {
                    run_agent_loop(
                        &manifest,
                        "new question",
                        &mut session,
                        &memory,
                        driver,
                        &[],
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        vec![],
                    )
                    .await
                },
            )
            .await
            .expect("Loop should complete");

        let requests = capturing.captured_requests();
        let msgs = &requests[0].messages;

        // Verify no consecutive assistant messages (they should be merged)
        for i in 0..msgs.len().saturating_sub(1) {
            if msgs[i].role == Role::Assistant && msgs[i + 1].role == Role::Assistant {
                panic!(
                    "Found consecutive assistant messages at indices {} and {} — should have been merged",
                    i, i + 1
                );
            }
        }

        // The merged assistant message should contain both parts
        let assistant_msg = msgs.iter().rev().find(|m| m.role == Role::Assistant).unwrap();
        let text = assistant_msg.content.text_content();
        assert!(
            text.contains("previous reply") && text.contains("[此刻的世界]"),
            "Merged assistant message should contain both parts, got: {text}"
        );
    }

    #[test]
    fn test_apply_dynamic_injections_basic() {
        use crate::tool_runner::{DynamicInjection, InjectionPosition, DYNAMIC_INJECTIONS};

        let mut messages = vec![
            Message::user("hello"),
            Message::assistant("hi"),
            Message::user("world"),
        ];

        // Manually set up injections via task-local
        DYNAMIC_INJECTIONS.sync_scope(std::cell::RefCell::new(vec![
            DynamicInjection {
                content: "injected".to_string(),
                position: InjectionPosition::InsertAssistant { offset_from_last: 0 },
            },
        ]), || {
            apply_dynamic_injections(&mut messages);
        });

        // After injection and merge:
        // user("hello"), assistant("hi" + "injected" merged), user("world")
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(messages[2].role, Role::User);
        let merged_text = messages[1].content.text_content();
        assert!(merged_text.contains("hi"), "Should contain original: {merged_text}");
        assert!(merged_text.contains("injected"), "Should contain injection: {merged_text}");
    }

    #[test]
    fn test_apply_dynamic_injections_empty() {
        use crate::tool_runner::DYNAMIC_INJECTIONS;

        let mut messages = vec![
            Message::user("hello"),
            Message::assistant("hi"),
        ];
        let original_len = messages.len();

        DYNAMIC_INJECTIONS.sync_scope(std::cell::RefCell::new(Vec::new()), || {
            apply_dynamic_injections(&mut messages);
        });

        assert_eq!(messages.len(), original_len, "No change when injections empty");
    }
}
