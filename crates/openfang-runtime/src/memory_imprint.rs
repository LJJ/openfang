const SYSTEM_FRAGMENTS: &[&str] = &[
    "[Channel context]",
    "[Task completed",
    "[The model returned an empty response",
    "[no response]",
    "[SYSTEM] Please output your Turn Script directly",
    "tool call",
    "tool result",
    "chat_id:",
    "sender_open_id:",
    "channel:",
    "NO_REPLY",
    "OK. NO_REPLY",
];

const INTERNAL_SUMMARY_PREFIXES: &[&str] = &[
    "给公子回了消息",
    "给他回了消息",
    "给公子发了消息",
    "看起来 Turn Script 文件路径不对",
    "既然 `pending.json` 为空",
];

const MEMORY_CUES: &[&str] = &[
    "记得",
    "喜欢",
    "不喜欢",
    "在意",
    "想",
    "要",
    "不开心",
    "开心",
    "群里",
    "私下",
    "语音",
    "视频",
    "衣服",
    "定妆照",
    "衣橱",
    "挑",
    "礼物",
    "照片",
    "自拍",
    "以后",
    "下次",
    "继续",
    "等会",
    "稍等",
    "马上",
    "很对",
    "可以",
    "不行",
    "只给",
    "亲近",
    "偏心",
];

const TRIVIAL_MESSAGES: &[&str] = &[
    "好",
    "好的",
    "嗯",
    "嗯嗯",
    "哦",
    "ok",
    "OK",
    "收到",
    "在吗",
    "在么",
    "在不在",
    "发",
    "行",
    "可以",
    "知道了",
    "是吗",
    "在",
    "1",
];

pub fn project_memory_imprint(user_message: &str, assistant_response: &str) -> Option<String> {
    let user = normalize_user_message(user_message);
    let assistant = normalize_text(assistant_response);

    if user.is_empty() || assistant.is_empty() {
        return None;
    }

    if is_system_like(&user) || is_system_like(&assistant) {
        return None;
    }

    if is_internal_summary(&assistant) {
        return None;
    }

    let user_fragment = select_fragment(&user);
    let assistant_fragment = select_fragment(&assistant);
    let score =
        fragment_score(user_fragment.as_deref()) + fragment_score(assistant_fragment.as_deref());

    if score < 3 && is_trivial_exchange(&user, &assistant) {
        return None;
    }

    render_memory_imprint(user_fragment.as_deref(), assistant_fragment.as_deref())
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

fn is_trivial_exchange(user: &str, assistant: &str) -> bool {
    is_trivial_message(user)
        && assistant.chars().count() <= 24
        && fragment_score(Some(assistant)) < 3
}

fn is_trivial_message(text: &str) -> bool {
    TRIVIAL_MESSAGES.contains(&text) || text.chars().count() <= 2
}

fn is_internal_summary(text: &str) -> bool {
    let trimmed = text.trim();
    INTERNAL_SUMMARY_PREFIXES
        .iter()
        .any(|prefix| trimmed.starts_with(prefix))
        || (trimmed.starts_with("拍了张")
            && (trimmed.contains("发给公子") || trimmed.contains("发给他")))
}

fn select_fragment(text: &str) -> Option<String> {
    let mut candidates: Vec<String> = split_sentences(text)
        .into_iter()
        .map(|s| s.trim().trim_matches(['"', '\'']))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by_key(|candidate| std::cmp::Reverse(fragment_score(Some(candidate))));
    let best = candidates.into_iter().next()?;
    Some(trim_fragment(&best))
}

fn split_sentences(text: &str) -> Vec<&str> {
    text.split(['。', '！', '？', '!', '?', '\n', ';', '；'])
        .collect()
}

fn trim_fragment(text: &str) -> String {
    let cleaned = text
        .trim()
        .trim_matches(['，', ',', '。', '！', '？', '、', '：', ':']);
    let mut out = cleaned.to_string();
    if out.chars().count() > 56 {
        out = out.chars().take(56).collect::<String>();
        out.push('…');
    }
    out
}

fn fragment_score(text: Option<&str>) -> usize {
    let Some(text) = text else {
        return 0;
    };
    let len = text.chars().count();
    let mut score = 0;

    if (6..=80).contains(&len) {
        score += 1;
    }
    if text.contains('“') || text.contains('”') || text.contains('"') {
        score += 1;
    }
    if MEMORY_CUES.iter().any(|cue| text.contains(cue)) {
        score += 2;
    }
    if text.contains("公子") {
        score += 1;
    }
    score
}

fn render_memory_imprint(user: Option<&str>, assistant: Option<&str>) -> Option<String> {
    match (user, assistant) {
        (Some(user), Some(assistant)) if user == assistant => {
            Some(format!("你记得那时心里留住的是这句话：“{}”。", user))
        }
        (Some(user), Some(assistant))
            if user.contains('吗') || user.contains('?') || user.contains('？') =>
        {
            Some(format!(
                "你记得公子问过你“{}”，你当时回了“{}”。",
                user, assistant
            ))
        }
        (Some(user), Some(assistant)) => Some(format!(
            "你记得公子说过“{}”。你也把自己当时那句“{}”留在了心里。",
            user, assistant
        )),
        (Some(user), None) => Some(format!("你记得公子说过“{}”。", user)),
        (None, Some(assistant)) => Some(format!("你记得自己当时回过公子“{}”。", assistant)),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::project_memory_imprint;

    #[test]
    fn skips_trivial_exchange() {
        let memory = project_memory_imprint("好", "在。");
        assert!(memory.is_none());
    }

    #[test]
    fn strips_channel_context() {
        let user = "[Channel context]\nchannel: feishu\nchat_id: xx\n\n你是不是不开心";
        let memory = project_memory_imprint(user, "我在群里会收着些，不太敢把对你的亲近摆出来。")
            .expect("memory imprint");
        assert!(!memory.contains("channel"));
        assert!(memory.contains("你是不是不开心"));
    }

    #[test]
    fn remembers_dialogue_fragment() {
        let memory = project_memory_imprint(
            "我不喜欢你在群里太冷",
            "我在群里会收着些，不太敢把对你的亲近摆出来。",
        )
        .expect("memory imprint");
        assert!(memory.contains("我不喜欢你在群里太冷"));
        assert!(memory.contains("亲近"));
        assert!(!memory.contains("User asked"));
    }

    #[test]
    fn skips_internal_action_summaries() {
        let memory = project_memory_imprint("你在干嘛", "给公子回了消息，说我在家看书。");
        assert!(memory.is_none());
    }
}
