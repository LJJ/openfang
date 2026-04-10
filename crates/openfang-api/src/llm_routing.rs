//! LLM routing API endpoints.
//!
//! Manages model assignments for all LLM call sites (agent brains, MCP toolbox,
//! director/cascade, filming). Reads/writes `.openfang/llm_routing.json` and
//! delegates agent model changes to the kernel's hot-swap mechanism.

use crate::routes::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use std::sync::Arc;

/// GET /api/llm-routing — Aggregated view of all LLM model assignments.
///
/// Returns agent models (from kernel registry) + MCP slots (from llm_routing.json).
pub async fn get_llm_routing(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let home = &state.kernel.config.home_dir;
    let routing_path = home.join("llm_routing.json");

    // Read llm_routing.json
    let routing: serde_json::Value = match std::fs::read_to_string(&routing_path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or(serde_json::json!({})),
        Err(_) => serde_json::json!({}),
    };

    // Read agent models from kernel registry
    let agents: Vec<serde_json::Value> = state
        .kernel
        .registry
        .list()
        .iter()
        .map(|entry| {
            let fallbacks: Vec<serde_json::Value> = entry
                .manifest
                .fallback_models
                .iter()
                .map(|fb| {
                    serde_json::json!({
                        "model": fb.model,
                        "provider": fb.provider,
                    })
                })
                .collect();
            serde_json::json!({
                "id": format!("{}", entry.id),
                "name": entry.manifest.name,
                "model": entry.manifest.model.model,
                "provider": entry.manifest.model.provider,
                "fallbacks": fallbacks,
            })
        })
        .collect();

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "agents": agents,
            "slots": routing.get("slots").cloned().unwrap_or(serde_json::json!({})),
            "providers": routing.get("providers").cloned().unwrap_or(serde_json::json!({})),
            "known_models": routing.get("known_models").cloned().unwrap_or(serde_json::json!([])),
        })),
    )
}

/// PUT /api/llm-routing — Update a slot's model assignment.
///
/// Body: `{ "slot": "simulation", "field": "primary"|"fallback", "value": "model-id" }`
///
/// - Agent slots (prefix `agent:`): delegates to `kernel.set_agent_model()` for hot-swap.
/// - MCP slots: writes to `llm_routing.json` (MCP hot-reloads via file mtime check).
/// - Director/cascade slots: also syncs to `world/cascade_config.json`.
pub async fn put_llm_routing(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let slot = match body["slot"].as_str() {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing 'slot' field"})),
            )
        }
    };
    let field = body["field"].as_str().unwrap_or("primary").to_string();
    let value = match body["value"].as_str() {
        Some(v) if !v.is_empty() => v.to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing 'value' field"})),
            )
        }
    };

    // ── Agent slot → kernel hot-swap (primary) or agent.toml edit (fallback) ──
    if let Some(agent_name) = slot.strip_prefix("agent:") {
        // Primary model → kernel hot-swap
        if field == "primary" {
            let agent_entry = state
                .kernel
                .registry
                .list()
                .iter()
                .find(|e| e.manifest.name == agent_name)
                .cloned();
            return match agent_entry {
                Some(entry) => match state.kernel.set_agent_model(entry.id, &value) {
                    Ok(()) => (
                        StatusCode::OK,
                        Json(serde_json::json!({"status": "ok", "slot": slot, "field": field, "value": value, "method": "kernel_hot_swap"})),
                    ),
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": format!("set_agent_model failed: {e}")})),
                    ),
                },
                None => (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({"error": format!("Agent '{agent_name}' not found")})),
                ),
            };
        }

        // Fallback model → edit agent.toml directly
        // field = "fallback.0", "fallback.1", etc.
        if let Some(idx_str) = field.strip_prefix("fallback.") {
            let idx: usize = match idx_str.parse() {
                Ok(i) => i,
                Err(_) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({"error": "Invalid fallback index"})),
                    )
                }
            };
            let home = &state.kernel.config.home_dir;
            let toml_path = home.join("agents").join(agent_name).join("agent.toml");
            return match update_agent_fallback_model(&toml_path, idx, &value) {
                Ok(()) => (
                    StatusCode::OK,
                    Json(serde_json::json!({"status": "ok", "slot": slot, "field": field, "value": value, "method": "agent_toml_edit", "note": "Restart required for fallback changes to take effect"})),
                ),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e})),
                ),
            };
        }

        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("Unknown field '{field}' for agent slot")})),
        );
    }

    // ── MCP slot → update llm_routing.json ──────────────────────────────
    let home = &state.kernel.config.home_dir;
    let routing_path = home.join("llm_routing.json");

    let mut config: serde_json::Value = match std::fs::read_to_string(&routing_path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or(serde_json::json!({"slots": {}})),
        Err(_) => serde_json::json!({"slots": {}}),
    };

    if config.get("slots").is_none() {
        config["slots"] = serde_json::json!({});
    }
    if config["slots"].get(&slot).is_none() {
        config["slots"][&slot] = serde_json::json!({});
    }
    config["slots"][&slot][&field] = serde_json::json!(value);

    let pretty = serde_json::to_string_pretty(&config).unwrap_or_default();
    if let Err(e) = std::fs::write(&routing_path, &pretty) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Write failed: {e}")})),
        );
    }

    // Sync director/cascade fields to cascade_config.json
    sync_cascade_config(home, &slot, &field, &value);

    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "ok", "slot": slot, "field": field, "value": value, "method": "file_write"})),
    )
}

/// Keep cascade_config.json in sync when director/cascade models change.
fn sync_cascade_config(home: &std::path::Path, slot: &str, field: &str, value: &str) {
    const MAP: &[(&str, &str, &str)] = &[
        ("director", "primary", "director_model"),
        ("director", "fallback", "director_fallback_model"),
        ("cascade:present", "primary", "cascade_model_present"),
        ("cascade:present", "fallback", "cascade_fallback_model_present"),
        ("cascade:absent", "primary", "cascade_model_absent"),
        ("cascade:absent", "fallback", "cascade_fallback_model_absent"),
    ];
    for (s, f, json_key) in MAP {
        if *s == slot && *f == field {
            let path = home.join("world").join("cascade_config.json");
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(mut obj) = serde_json::from_str::<serde_json::Value>(&text) {
                    obj[*json_key] = serde_json::json!(value);
                    let _ = std::fs::write(
                        &path,
                        serde_json::to_string_pretty(&obj).unwrap_or_default(),
                    );
                }
            }
        }
    }
}

/// Update the model field of the Nth `[[fallback_models]]` entry in an agent.toml.
///
/// Uses TOML parsing to preserve structure; only the `model` value changes.
fn update_agent_fallback_model(
    toml_path: &std::path::Path,
    index: usize,
    new_model: &str,
) -> Result<(), String> {
    let content =
        std::fs::read_to_string(toml_path).map_err(|e| format!("Read agent.toml: {e}"))?;
    let mut doc: toml::Value =
        toml::from_str(&content).map_err(|e| format!("Parse agent.toml: {e}"))?;

    let fallbacks = doc
        .get_mut("fallback_models")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| "No fallback_models array in agent.toml".to_string())?;

    let len = fallbacks.len();
    let entry = fallbacks
        .get_mut(index)
        .ok_or_else(|| format!("Fallback index {index} out of range (len={len})"))?;

    if let Some(table) = entry.as_table_mut() {
        table.insert("model".into(), toml::Value::String(new_model.to_string()));
    } else {
        return Err("Fallback entry is not a table".into());
    }

    let output = toml::to_string_pretty(&doc).map_err(|e| format!("Serialize: {e}"))?;
    std::fs::write(toml_path, output).map_err(|e| format!("Write agent.toml: {e}"))?;
    Ok(())
}

/// POST /api/llm-routing/test — Test a model's API connectivity.
///
/// Body: `{ "model": "deepseek-chat", "base_url": "https://...", "api_key_env": "DEEPSEEK_API_KEY", "message?": "hi" }`
///
/// Reads the actual API key from the specified env var, sends a minimal
/// chat completion request, and reports latency + truncated response.
pub async fn test_llm_routing(
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let model = match body["model"].as_str() {
        Some(m) if !m.is_empty() => m.to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing 'model'"})),
            )
        }
    };
    let base_url = match body["base_url"].as_str() {
        Some(u) if !u.is_empty() => u.to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing 'base_url'"})),
            )
        }
    };
    let api_key_env = body["api_key_env"].as_str().unwrap_or("");
    let api_key = if api_key_env.is_empty() {
        String::new()
    } else {
        std::env::var(api_key_env).unwrap_or_default()
    };
    if api_key.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("Env '{api_key_env}' is empty or not set")})),
        );
    }

    let message = body["message"].as_str().unwrap_or("Say one word.");
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    // GPT-5.x requires max_completion_tokens instead of max_tokens
    let token_key = if model.starts_with("gpt-5") {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };

    let request_body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.1,
        token_key: 50,
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default();

    let start = std::time::Instant::now();
    let result = client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&request_body)
        .send()
        .await;
    let latency_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(resp) => {
            let http_status = resp.status().as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            if (200..300).contains(&http_status) {
                let parsed: serde_json::Value =
                    serde_json::from_str(&body_text).unwrap_or_default();
                let content = parsed["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .chars()
                    .take(200)
                    .collect::<String>();
                (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "status": "ok", "model": model,
                        "latency_ms": latency_ms, "response": content,
                    })),
                )
            } else {
                (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "status": "error", "model": model,
                        "latency_ms": latency_ms, "http_status": http_status,
                        "error": body_text.chars().take(300).collect::<String>(),
                    })),
                )
            }
        }
        Err(e) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "error", "model": model,
                "latency_ms": latency_ms, "error": e.to_string(),
            })),
        ),
    }
}
