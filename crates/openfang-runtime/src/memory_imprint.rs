//! Episodic memory projection — extract meaningful memory fragments from turns.
//!
//! The mechanism (scoring, fragment selection, sentence splitting, rendering)
//! is generic runtime infrastructure. All content (cue words, trivial messages,
//! scoring weights, entity keywords, templates) is externalized to a per-agent
//! config file referenced by `metadata.memory_imprint.config` in agent.toml.
//!
//! If no config is provided, this module is disabled for that agent.

use serde::Deserialize;
use std::path::Path;

/// Loaded memory imprint configuration. All content is agent-specific.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryImprintConfig {
    /// Words/phrases that signal memory-worthy content.
    #[serde(default)]
    pub cue_words: Vec<String>,
    /// Messages too short/trivial to remember.
    #[serde(default)]
    pub trivial_messages: Vec<String>,
    /// Prefixes that indicate internal action summaries (not real dialogue).
    #[serde(default)]
    pub internal_summary_prefixes: Vec<String>,
    /// Entity keywords (e.g., "公子") for scoring bonus.
    #[serde(default)]
    pub entity_keywords: Vec<String>,
    /// Scoring weights.
    #[serde(default)]
    pub scoring: ScoringConfig,
    /// Rendering templates.
    #[serde(default)]
    pub templates: TemplateConfig,
}

/// Scoring weights for fragment evaluation.
#[derive(Debug, Clone, Deserialize)]
pub struct ScoringConfig {
    /// Score bonus when a cue word matches.
    #[serde(default = "default_cue_match")]
    pub cue_match: usize,
    /// Score bonus when an entity keyword matches.
    #[serde(default = "default_entity_match")]
    pub entity_match: usize,
    /// Score bonus when quoted text is present.
    #[serde(default = "default_quote_match")]
    pub quote_match: usize,
    /// Character count range [min, max] for length bonus.
    #[serde(default = "default_length_bonus_range")]
    pub length_bonus_range: [usize; 2],
    /// Minimum score to avoid trivial-exchange filtering.
    #[serde(default = "default_trivial_threshold")]
    pub trivial_threshold: usize,
    /// Maximum characters in a fragment before truncation.
    #[serde(default = "default_max_fragment_chars")]
    pub max_fragment_chars: usize,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            cue_match: default_cue_match(),
            entity_match: default_entity_match(),
            quote_match: default_quote_match(),
            length_bonus_range: default_length_bonus_range(),
            trivial_threshold: default_trivial_threshold(),
            max_fragment_chars: default_max_fragment_chars(),
        }
    }
}

fn default_cue_match() -> usize { 2 }
fn default_entity_match() -> usize { 1 }
fn default_quote_match() -> usize { 1 }
fn default_length_bonus_range() -> [usize; 2] { [6, 80] }
fn default_trivial_threshold() -> usize { 3 }
fn default_max_fragment_chars() -> usize { 56 }

/// Templates for rendering memory fragments.
#[derive(Debug, Clone, Deserialize)]
pub struct TemplateConfig {
    /// When user and assistant fragments are identical.
    /// Placeholder: `{fragment}`
    #[serde(default)]
    pub both_same: String,
    /// When user fragment is a question.
    /// Placeholders: `{entity}`, `{user}`, `{assistant}`
    #[serde(default)]
    pub question_reply: String,
    /// When both fragments differ (non-question).
    /// Placeholders: `{entity}`, `{user}`, `{assistant}`
    #[serde(default)]
    pub both_different: String,
    /// When only user fragment is available.
    /// Placeholders: `{entity}`, `{user}`
    #[serde(default)]
    pub user_only: String,
    /// When only assistant fragment is available.
    /// Placeholders: `{entity}`, `{assistant}`
    #[serde(default)]
    pub assistant_only: String,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            both_same: String::new(),
            question_reply: String::new(),
            both_different: String::new(),
            user_only: String::new(),
            assistant_only: String::new(),
        }
    }
}

/// System-like fragment markers that indicate non-dialogue content.
/// These are generic (not agent-specific) so they stay in code.
const SYSTEM_FRAGMENTS: &[&str] = &[
    "[Channel context]",
    "[Task completed",
    "[The model returned an empty response",
    "[no response]",
    "[SYSTEM]",
    "tool call",
    "tool result",
    "NO_REPLY",
    "OK. NO_REPLY",
];

/// Load a memory imprint config from a TOML file.
///
/// Returns `None` if the path doesn't exist or can't be parsed.
pub fn load_config(path: &Path) -> Option<MemoryImprintConfig> {
    let content = std::fs::read_to_string(path).ok()?;
    toml::from_str(&content).ok()
}

/// Resolve config path from agent manifest metadata.
///
/// Looks for `metadata.memory_imprint.config` (string path relative to agent dir).
/// Returns `None` if not configured.
pub fn config_path_from_metadata(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
    agent_dir: &Path,
) -> Option<std::path::PathBuf> {
    let rel_path = metadata
        .get("memory_imprint")?
        .get("config")?
        .as_str()?;
    Some(agent_dir.join(rel_path))
}

/// Project a memory imprint from a user-assistant exchange.
///
/// Returns `None` if the exchange is trivial, system-like, or below threshold.
pub fn project_memory_imprint(
    config: &MemoryImprintConfig,
    user_message: &str,
    assistant_response: &str,
) -> Option<String> {
    let user = normalize_user_message(user_message);
    let assistant = normalize_text(assistant_response);

    if user.is_empty() || assistant.is_empty() {
        return None;
    }

    if is_system_like(&user) || is_system_like(&assistant) {
        return None;
    }

    if is_internal_summary(config, &assistant) {
        return None;
    }

    let user_fragment = select_fragment(config, &user);
    let assistant_fragment = select_fragment(config, &assistant);
    let score = fragment_score(config, user_fragment.as_deref())
        + fragment_score(config, assistant_fragment.as_deref());

    if score < config.scoring.trivial_threshold
        && is_trivial_exchange(config, &user, &assistant)
    {
        return None;
    }

    render_memory_imprint(config, user_fragment.as_deref(), assistant_fragment.as_deref())
}

fn normalize_user_message(message: &str) -> String {
    let mut in_channel_context = false;
    let mut parts = Vec::new();

    for raw_line in message.lines() {
        let line = raw_line.trim();
        if !in_channel_context && line == "[Channel context]" {
            in_channel_context = true;
            continue;
        }
        if in_channel_context {
            if line.is_empty() {
                in_channel_context = false;
            }
            continue;
        }
        if !line.is_empty() {
            parts.push(line);
        }
    }

    normalize_text(&parts.join(" "))
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_system_like(text: &str) -> bool {
    SYSTEM_FRAGMENTS.iter().any(|needle| text.contains(needle))
}

fn is_trivial_exchange(config: &MemoryImprintConfig, user: &str, assistant: &str) -> bool {
    is_trivial_message(config, user)
        && assistant.chars().count() <= 24
        && fragment_score(config, Some(assistant)) < config.scoring.trivial_threshold
}

fn is_trivial_message(config: &MemoryImprintConfig, text: &str) -> bool {
    config.trivial_messages.iter().any(|t| t == text) || text.chars().count() <= 2
}

fn is_internal_summary(config: &MemoryImprintConfig, text: &str) -> bool {
    let trimmed = text.trim();
    config
        .internal_summary_prefixes
        .iter()
        .any(|prefix| trimmed.starts_with(prefix.as_str()))
}

fn select_fragment(config: &MemoryImprintConfig, text: &str) -> Option<String> {
    let mut candidates: Vec<String> = split_sentences(text)
        .into_iter()
        .map(|s| s.trim().trim_matches(['"', '\'']))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by_key(|candidate| {
        std::cmp::Reverse(fragment_score(config, Some(candidate)))
    });
    let best = candidates.into_iter().next()?;
    Some(trim_fragment(config, &best))
}

fn split_sentences(text: &str) -> Vec<&str> {
    text.split(['。', '！', '？', '!', '?', '\n', ';', '；'])
        .collect()
}

fn trim_fragment(config: &MemoryImprintConfig, text: &str) -> String {
    let cleaned = text
        .trim()
        .trim_matches(['，', ',', '。', '！', '？', '、', '：', ':']);
    let mut out = cleaned.to_string();
    let max = config.scoring.max_fragment_chars;
    if out.chars().count() > max {
        out = out.chars().take(max).collect::<String>();
        out.push('…');
    }
    out
}

fn fragment_score(config: &MemoryImprintConfig, text: Option<&str>) -> usize {
    let Some(text) = text else {
        return 0;
    };
    let len = text.chars().count();
    let mut score = 0;

    let [min_len, max_len] = config.scoring.length_bonus_range;
    if (min_len..=max_len).contains(&len) {
        score += 1;
    }
    if text.contains('"') || text.contains('\u{201c}') || text.contains('\u{201d}') {
        score += config.scoring.quote_match;
    }
    if config.cue_words.iter().any(|cue| text.contains(cue.as_str())) {
        score += config.scoring.cue_match;
    }
    if config.entity_keywords.iter().any(|kw| text.contains(kw.as_str())) {
        score += config.scoring.entity_match;
    }
    score
}

fn render_memory_imprint(
    config: &MemoryImprintConfig,
    user: Option<&str>,
    assistant: Option<&str>,
) -> Option<String> {
    let entity = config
        .entity_keywords
        .first()
        .map(|s| s.as_str())
        .unwrap_or("user");

    match (user, assistant) {
        (Some(u), Some(a)) if u == a => {
            if config.templates.both_same.is_empty() {
                return None;
            }
            Some(config.templates.both_same.replace("{fragment}", u))
        }
        (Some(u), Some(a))
            if u.contains('吗') || u.contains('?') || u.contains('？') =>
        {
            if config.templates.question_reply.is_empty() {
                return None;
            }
            Some(
                config.templates.question_reply
                    .replace("{entity}", entity)
                    .replace("{user}", u)
                    .replace("{assistant}", a),
            )
        }
        (Some(u), Some(a)) => {
            if config.templates.both_different.is_empty() {
                return None;
            }
            Some(
                config.templates.both_different
                    .replace("{entity}", entity)
                    .replace("{user}", u)
                    .replace("{assistant}", a),
            )
        }
        (Some(u), None) => {
            if config.templates.user_only.is_empty() {
                return None;
            }
            Some(
                config.templates.user_only
                    .replace("{entity}", entity)
                    .replace("{user}", u),
            )
        }
        (None, Some(a)) => {
            if config.templates.assistant_only.is_empty() {
                return None;
            }
            Some(
                config.templates.assistant_only
                    .replace("{entity}", entity)
                    .replace("{assistant}", a),
            )
        }
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> MemoryImprintConfig {
        MemoryImprintConfig {
            cue_words: vec!["不喜欢".into(), "记得".into(), "亲近".into()],
            trivial_messages: vec!["好".into(), "嗯".into(), "ok".into()],
            internal_summary_prefixes: vec!["给公子回了消息".into()],
            entity_keywords: vec!["公子".into()],
            scoring: ScoringConfig::default(),
            templates: TemplateConfig {
                both_same: "你记得那时心里留住的是这句话：\"{fragment}\"。".into(),
                question_reply: "你记得{entity}问过你\"{user}\"，你当时回了\"{assistant}\"。".into(),
                both_different: "你记得{entity}说过\"{user}\"。你也把自己当时那句\"{assistant}\"留在了心里。".into(),
                user_only: "你记得{entity}说过\"{user}\"。".into(),
                assistant_only: "你记得自己当时回过{entity}\"{assistant}\"。".into(),
            },
        }
    }

    #[test]
    fn skips_trivial_exchange() {
        let config = test_config();
        let memory = project_memory_imprint(&config, "好", "在。");
        assert!(memory.is_none());
    }

    #[test]
    fn strips_channel_context() {
        let config = test_config();
        let user = "[Channel context]\nchannel: feishu\nchat_id: xx\n\n你是不是不开心";
        let memory = project_memory_imprint(
            &config,
            user,
            "我在群里会收着些，不太敢把对你的亲近摆出来。",
        )
        .expect("memory imprint");
        assert!(!memory.contains("channel"));
    }

    #[test]
    fn remembers_dialogue_fragment() {
        let config = test_config();
        let memory = project_memory_imprint(
            &config,
            "我不喜欢你在群里太冷",
            "我在群里会收着些，不太敢把对你的亲近摆出来。",
        )
        .expect("memory imprint");
        assert!(memory.contains("不喜欢"));
    }

    #[test]
    fn skips_internal_action_summaries() {
        let config = test_config();
        let memory = project_memory_imprint(&config, "你在干嘛", "给公子回了消息，说我在家看书。");
        assert!(memory.is_none());
    }

    #[test]
    fn disabled_without_config() {
        // No config = no memory imprint (the caller checks config existence)
        // This test just verifies the function works with minimal config
        let config = MemoryImprintConfig {
            cue_words: vec![],
            trivial_messages: vec![],
            internal_summary_prefixes: vec![],
            entity_keywords: vec![],
            scoring: ScoringConfig::default(),
            templates: TemplateConfig::default(),
        };
        // With empty templates, rendering returns None
        let memory = project_memory_imprint(&config, "hello", "world");
        assert!(memory.is_none());
    }
}
