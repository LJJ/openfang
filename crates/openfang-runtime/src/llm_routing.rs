//! LLM routing resolution — shared logic for resolving providers from llm_routing.json.
//!
//! Mirrors the MCP layer's `llm_routing.js` resolveProvider logic:
//! reads `providers` section, matches model ID by prefix, returns provider + credentials.
//!
//! Used by kernel (compact driver, slot resolution) and runtime (session compact).

use crate::model_catalog::ModelCatalog;
use std::sync::RwLock;

/// Resolved provider info: (driver_provider, api_key_env, base_url).
pub type ProviderInfo = (String, String, Option<String>);

/// Resolve provider, API key env, and base URL for a model.
///
/// Priority:
/// 1. `llm_routing.json` providers section (prefix match — same as MCP layer)
/// 2. Model catalog (hardcoded provider entries)
/// 3. Hardcoded prefix fallback
pub fn resolve_provider(
    routing_config: &serde_json::Value,
    model_id: &str,
    model_catalog: &RwLock<ModelCatalog>,
) -> ProviderInfo {
    // 1. llm_routing.json providers section (prefix match)
    if let Some(info) = resolve_from_routing_providers(routing_config, model_id) {
        return info;
    }

    // 2. Model catalog
    if let Some(info) = resolve_from_catalog(model_id, model_catalog) {
        return info;
    }

    // 3. Hardcoded prefix fallback
    resolve_from_prefix(model_id)
}

/// Read a slot's primary model from llm_routing.json.
pub fn read_slot_model(routing_config: &serde_json::Value, slot: &str) -> Option<String> {
    routing_config
        .get("slots")?
        .get(slot)?
        .get("primary")?
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// Load and parse llm_routing.json from the home directory.
pub fn load_routing_config(home_dir: &std::path::Path) -> Option<serde_json::Value> {
    let path = home_dir.join("llm_routing.json");
    let content = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

// ── Internal ────────────────────────────────────────────────────────

/// Match model ID against llm_routing.json providers prefixes.
/// Same logic as MCP layer's `resolveProvider(modelId)`.
fn resolve_from_routing_providers(
    routing_config: &serde_json::Value,
    model_id: &str,
) -> Option<ProviderInfo> {
    let providers = routing_config.get("providers")?.as_object()?;

    for (_name, pconf) in providers {
        let prefixes = pconf
            .get("prefixes")
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default();

        for prefix in &prefixes {
            let pfx = prefix.as_str()?;
            if !model_id.starts_with(pfx) {
                continue;
            }

            let api_key_env = pconf
                .get("api_key_env")
                .and_then(|v| v.as_str())
                .unwrap_or("OPENAI_API_KEY")
                .to_string();

            let base_url = pconf
                .get("base_url")
                .and_then(|v| v.as_str())
                .filter(|u| !u.is_empty())
                .map(|u| u.to_string())
                .or_else(|| {
                    pconf
                        .get("base_url_env")
                        .and_then(|v| v.as_str())
                        .and_then(|env| std::env::var(env).ok())
                });

            // Map prefix to driver type
            let provider = prefix_to_driver(pfx);

            return Some((provider, api_key_env, base_url));
        }
    }
    None
}

/// Resolve from the hardcoded model catalog.
fn resolve_from_catalog(
    model_id: &str,
    model_catalog: &RwLock<ModelCatalog>,
) -> Option<ProviderInfo> {
    let catalog = model_catalog.read().ok()?;
    let entry = catalog.find_model(model_id)?;
    let provider = entry.provider.clone();
    let api_key_env = catalog
        .get_provider(&provider)
        .map(|p| p.api_key_env.clone())
        .unwrap_or_else(|| "OPENAI_API_KEY".to_string());
    let base_url = catalog
        .get_provider(&provider)
        .and_then(|p| {
            if p.base_url.is_empty() {
                None
            } else {
                Some(p.base_url.clone())
            }
        });
    Some((provider, api_key_env, base_url))
}

/// Last-resort prefix-based provider guessing.
fn resolve_from_prefix(model_id: &str) -> ProviderInfo {
    let provider = if model_id.starts_with("deepseek") {
        "deepseek"
    } else if model_id.starts_with("gemini") {
        "gemini"
    } else if model_id.starts_with("MiniMax") || model_id.starts_with("M2") {
        "minimax"
    } else if model_id.starts_with("gpt-") {
        "azure"
    } else {
        "openai"
    };
    (provider.to_string(), "OPENAI_API_KEY".to_string(), None)
}

/// Map a model prefix to the corresponding driver type name.
fn prefix_to_driver(prefix: &str) -> String {
    // These map llm_routing.json prefix → kernel driver name
    if prefix.starts_with("gemini") {
        "gemini".to_string()
    } else if prefix.starts_with("claude") {
        "openai".to_string() // claude goes through OpenAI-compat proxy
    } else {
        prefix.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_from_routing_providers_gemini() {
        let config: serde_json::Value = serde_json::json!({
            "providers": {
                "vertex": {
                    "api_key_env": "GOOGLE_APPLICATION_CREDENTIALS",
                    "prefixes": ["gemini"]
                }
            }
        });
        let info = resolve_from_routing_providers(&config, "gemini-3.1-flash-lite-preview");
        assert!(info.is_some());
        let (provider, key_env, _) = info.unwrap();
        assert_eq!(provider, "gemini");
        assert_eq!(key_env, "GOOGLE_APPLICATION_CREDENTIALS");
    }

    #[test]
    fn test_resolve_from_routing_providers_deepseek() {
        let config: serde_json::Value = serde_json::json!({
            "providers": {
                "deepseek": {
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url": "https://api.deepseek.com/v1",
                    "prefixes": ["deepseek"]
                }
            }
        });
        let info = resolve_from_routing_providers(&config, "deepseek-chat");
        assert!(info.is_some());
        let (provider, key_env, base_url) = info.unwrap();
        assert_eq!(provider, "deepseek");
        assert_eq!(key_env, "DEEPSEEK_API_KEY");
        assert_eq!(base_url.unwrap(), "https://api.deepseek.com/v1");
    }

    #[test]
    fn test_resolve_from_prefix_fallback() {
        let (p, _, _) = resolve_from_prefix("gemini-3.1-flash-lite-preview");
        assert_eq!(p, "gemini");
        let (p, _, _) = resolve_from_prefix("deepseek-chat");
        assert_eq!(p, "deepseek");
        let (p, _, _) = resolve_from_prefix("gpt-5.4");
        assert_eq!(p, "azure");
        let (p, _, _) = resolve_from_prefix("unknown-model");
        assert_eq!(p, "openai");
    }

    #[test]
    fn test_read_slot_model() {
        let config: serde_json::Value = serde_json::json!({
            "slots": {
                "compact": { "primary": "deepseek-chat" },
                "empty": { "primary": "" }
            }
        });
        assert_eq!(read_slot_model(&config, "compact").unwrap(), "deepseek-chat");
        assert!(read_slot_model(&config, "empty").is_none());
        assert!(read_slot_model(&config, "missing").is_none());
    }
}
