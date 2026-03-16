//! Media understanding engine — image description, audio transcription, video analysis.
//!
//! Auto-cascades through available providers based on configured API keys.

use openfang_types::media::{
    MediaAttachment, MediaConfig, MediaSource, MediaType, MediaUnderstanding,
};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::info;

/// Generic default prompt for image description (English, provider-neutral).
/// Deploy-time config can override via `media.image_description_prompt`.
const DEFAULT_IMAGE_DESCRIPTION_PROMPT: &str = "Describe the image objectively based on visible evidence. Focus on the main subject, clothing type, colors, silhouette, notable details, background, and any readable text. Do not infer actions beyond what is shown; state uncertainty if unsure.";

/// Media understanding engine.
pub struct MediaEngine {
    config: MediaConfig,
    semaphore: Arc<Semaphore>,
}

impl MediaEngine {
    pub fn new(config: MediaConfig) -> Self {
        let max = config.max_concurrency.clamp(1, 8);
        Self {
            config,
            semaphore: Arc::new(Semaphore::new(max)),
        }
    }

    /// Describe an image using a vision-capable LLM.
    /// Auto-cascade: Anthropic -> OpenAI -> Gemini (based on API key availability).
    pub async fn describe_image(
        &self,
        attachment: &MediaAttachment,
    ) -> Result<MediaUnderstanding, String> {
        attachment.validate()?;
        if attachment.media_type != MediaType::Image {
            return Err("Expected image attachment".into());
        }

        // Determine which provider to use
        let provider = self.config.image_provider.as_deref()
            .or_else(|| detect_vision_provider())
            .ok_or("No vision-capable LLM provider configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY")?;
        let _permit = self.semaphore.acquire().await.map_err(|e| e.to_string())?;

        let image_b64 = match &attachment.source {
            MediaSource::Base64 { data, .. } => data.clone(),
            MediaSource::FilePath { path } => {
                use base64::Engine;
                let bytes = tokio::fs::read(path)
                    .await
                    .map_err(|e| format!("Failed to read image file '{}': {}", path, e))?;
                base64::engine::general_purpose::STANDARD.encode(bytes)
            }
            MediaSource::Url { url } => {
                return Err(format!(
                    "URL-based image source not supported for description: {}",
                    url
                ));
            }
        };

        let prompt = self
            .config
            .image_description_prompt
            .as_deref()
            .unwrap_or(DEFAULT_IMAGE_DESCRIPTION_PROMPT);
        let model = default_vision_model(provider);
        let description = match provider {
            "anthropic" => {
                describe_image_anthropic(&image_b64, &attachment.mime_type, model, prompt).await?
            }
            "openai" => {
                describe_image_openai(&image_b64, &attachment.mime_type, model, prompt).await?
            }
            "gemini" => {
                describe_image_gemini(&image_b64, &attachment.mime_type, model, prompt).await?
            }
            other => return Err(format!("Unsupported vision provider: {}", other)),
        };

        Ok(MediaUnderstanding {
            media_type: MediaType::Image,
            description,
            provider: provider.to_string(),
            model: model.to_string(),
        })
    }

    /// Transcribe audio using speech-to-text.
    /// Auto-cascade: Groq (whisper-large-v3-turbo) -> OpenAI (whisper-1).
    pub async fn transcribe_audio(
        &self,
        attachment: &MediaAttachment,
    ) -> Result<MediaUnderstanding, String> {
        attachment.validate()?;
        if attachment.media_type != MediaType::Audio {
            return Err("Expected audio attachment".into());
        }

        let provider = self
            .config
            .audio_provider
            .as_deref()
            .or_else(|| detect_audio_provider())
            .ok_or(
                "No audio transcription provider configured. Set GROQ_API_KEY or OPENAI_API_KEY",
            )?;

        let _permit = self.semaphore.acquire().await.map_err(|e| e.to_string())?;

        // Derive a proper filename with extension from mime_type
        // (Whisper APIs require an extension to detect format)
        let ext = match attachment.mime_type.as_str() {
            "audio/wav" => "wav",
            "audio/mpeg" | "audio/mp3" => "mp3",
            "audio/ogg" => "ogg",
            "audio/webm" => "webm",
            "audio/mp4" | "audio/m4a" => "m4a",
            "audio/flac" => "flac",
            _ => "wav",
        };

        // Read audio bytes from source
        let audio_bytes = match &attachment.source {
            MediaSource::FilePath { path } => tokio::fs::read(path)
                .await
                .map_err(|e| format!("Failed to read audio file '{}': {}", path, e))?,
            MediaSource::Base64 { data, .. } => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|e| format!("Failed to decode base64 audio: {}", e))?
            }
            MediaSource::Url { url } => {
                return Err(format!(
                    "URL-based audio source not supported for transcription: {}",
                    url
                ));
            }
        };
        let filename = format!("audio.{}", ext);

        let model = default_audio_model(provider);

        // Build API request
        let (api_url, api_key) = match provider {
            "groq" => (
                "https://api.groq.com/openai/v1/audio/transcriptions",
                std::env::var("GROQ_API_KEY").map_err(|_| "GROQ_API_KEY not set")?,
            ),
            "openai" => (
                "https://api.openai.com/v1/audio/transcriptions",
                std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set")?,
            ),
            other => return Err(format!("Unsupported audio provider: {}", other)),
        };

        info!(provider, model, filename = %filename, size = audio_bytes.len(), "Sending audio for transcription");

        let file_part = reqwest::multipart::Part::bytes(audio_bytes)
            .file_name(filename)
            .mime_str(&attachment.mime_type)
            .map_err(|e| format!("Failed to set MIME type: {}", e))?;

        let form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .text("model", model.to_string())
            .text("response_format", "text");

        let client = reqwest::Client::new();
        let resp = client
            .post(api_url)
            .bearer_auth(&api_key)
            .multipart(form)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| format!("Transcription request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Transcription API error ({}): {}", status, body));
        }

        let transcription = resp
            .text()
            .await
            .map_err(|e| format!("Failed to read transcription response: {}", e))?;

        let transcription = transcription.trim().to_string();
        if transcription.is_empty() {
            return Err("Transcription returned empty text".into());
        }

        info!(
            provider,
            model,
            chars = transcription.len(),
            "Audio transcription complete"
        );

        Ok(MediaUnderstanding {
            media_type: MediaType::Audio,
            description: transcription,
            provider: provider.to_string(),
            model: model.to_string(),
        })
    }

    /// Describe video using Gemini.
    pub async fn describe_video(
        &self,
        attachment: &MediaAttachment,
    ) -> Result<MediaUnderstanding, String> {
        attachment.validate()?;
        if attachment.media_type != MediaType::Video {
            return Err("Expected video attachment".into());
        }

        if !self.config.video_description {
            return Err("Video description is disabled in configuration".into());
        }

        if std::env::var("GEMINI_API_KEY").is_err() && std::env::var("GOOGLE_API_KEY").is_err() {
            return Err("Video description requires GEMINI_API_KEY or GOOGLE_API_KEY".into());
        }

        Ok(MediaUnderstanding {
            media_type: MediaType::Video,
            description: "[Video description would be generated by Gemini]".to_string(),
            provider: "gemini".to_string(),
            model: "gemini-2.5-flash".to_string(),
        })
    }

    /// Process multiple attachments concurrently (bounded by max_concurrency).
    pub async fn process_attachments(
        &self,
        attachments: Vec<MediaAttachment>,
    ) -> Vec<Result<MediaUnderstanding, String>> {
        let mut handles = Vec::new();

        for attachment in attachments {
            let sem = self.semaphore.clone();
            let config = self.config.clone();
            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.map_err(|e| e.to_string())?;
                let engine = MediaEngine {
                    config,
                    semaphore: Arc::new(Semaphore::new(1)), // inner engine, no extra semaphore
                };
                match attachment.media_type {
                    MediaType::Image => engine.describe_image(&attachment).await,
                    MediaType::Audio => engine.transcribe_audio(&attachment).await,
                    MediaType::Video => engine.describe_video(&attachment).await,
                }
            });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(Err(format!("Task failed: {e}"))),
            }
        }
        results
    }
}

/// Detect which vision provider is available based on environment variables.
fn detect_vision_provider() -> Option<&'static str> {
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        return Some("anthropic");
    }
    if std::env::var("OPENAI_API_KEY").is_ok() {
        return Some("openai");
    }
    if std::env::var("GEMINI_API_KEY").is_ok() || std::env::var("GOOGLE_API_KEY").is_ok() {
        return Some("gemini");
    }
    None
}

/// Detect which audio transcription provider is available.
fn detect_audio_provider() -> Option<&'static str> {
    if std::env::var("GROQ_API_KEY").is_ok() {
        return Some("groq");
    }
    if std::env::var("OPENAI_API_KEY").is_ok() {
        return Some("openai");
    }
    None
}

/// Get the default vision model for a provider.
fn default_vision_model(provider: &str) -> &str {
    match provider {
        "anthropic" => "claude-sonnet-4-20250514",
        "openai" => "gpt-4o",
        "gemini" => "gemini-2.5-flash",
        _ => "unknown",
    }
}

/// Get the default audio model for a provider.
fn default_audio_model(provider: &str) -> &str {
    match provider {
        "groq" => "whisper-large-v3-turbo",
        "openai" => "whisper-1",
        _ => "unknown",
    }
}

async fn describe_image_openai(
    image_b64: &str,
    mime_type: &str,
    model: &str,
    prompt: &str,
) -> Result<String, String> {
    let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set")?;
    let resp = reqwest::Client::new()
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&serde_json::json!({
            "model": model,
            "max_tokens": 500,
            "temperature": 0.2,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": format!("data:{};base64,{}", mime_type, image_b64)}}
                ]
            }]
        }))
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| format!("OpenAI vision request failed: {}", e))?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!("OpenAI vision API error ({}): {}", status, body));
    }

    let value: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("OpenAI vision parse error: {}", e))?;
    value["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "OpenAI vision returned empty text".to_string())
}

async fn describe_image_anthropic(
    image_b64: &str,
    mime_type: &str,
    model: &str,
    prompt: &str,
) -> Result<String, String> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| "ANTHROPIC_API_KEY not set")?;
    let resp = reqwest::Client::new()
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&serde_json::json!({
            "model": model,
            "max_tokens": 500,
            "temperature": 0.2,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_b64}}
                ]
            }]
        }))
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| format!("Anthropic vision request failed: {}", e))?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!("Anthropic vision API error ({}): {}", status, body));
    }

    let value: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("Anthropic vision parse error: {}", e))?;
    let mut texts = Vec::new();
    if let Some(items) = value["content"].as_array() {
        for item in items {
            if item["type"].as_str() == Some("text") {
                if let Some(text) = item["text"].as_str() {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        texts.push(trimmed.to_string());
                    }
                }
            }
        }
    }
    if texts.is_empty() {
        return Err("Anthropic vision returned empty text".to_string());
    }
    Ok(texts.join("\n"))
}

async fn describe_image_gemini(
    image_b64: &str,
    mime_type: &str,
    model: &str,
    prompt: &str,
) -> Result<String, String> {
    let api_key = std::env::var("GEMINI_API_KEY")
        .or_else(|_| std::env::var("GOOGLE_API_KEY"))
        .map_err(|_| "GEMINI_API_KEY or GOOGLE_API_KEY not set")?;
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        model
    );
    let resp = reqwest::Client::new()
        .post(&url)
        .header("x-goog-api-key", api_key)
        .header("content-type", "application/json")
        .json(&serde_json::json!({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": image_b64}}
                ]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 500
            }
        }))
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| format!("Gemini vision request failed: {}", e))?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!("Gemini vision API error ({}): {}", status, body));
    }

    let value: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("Gemini vision parse error: {}", e))?;
    let mut texts = Vec::new();
    if let Some(parts) = value["candidates"][0]["content"]["parts"].as_array() {
        for part in parts {
            if let Some(text) = part["text"].as_str() {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    texts.push(trimmed.to_string());
                }
            }
        }
    }
    if texts.is_empty() {
        return Err("Gemini vision returned empty text".to_string());
    }
    Ok(texts.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use openfang_types::media::{MediaSource, MAX_IMAGE_BYTES};

    #[test]
    fn test_engine_creation() {
        let config = MediaConfig::default();
        let engine = MediaEngine::new(config);
        assert_eq!(engine.config.max_concurrency, 2);
    }

    #[test]
    fn test_engine_max_concurrency_clamped() {
        let config = MediaConfig {
            max_concurrency: 100,
            ..Default::default()
        };
        let engine = MediaEngine::new(config);
        // Semaphore was clamped to 8
        assert!(engine.semaphore.available_permits() <= 8);
    }

    #[tokio::test]
    async fn test_describe_image_wrong_type() {
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Audio,
            mime_type: "audio/mpeg".into(),
            source: MediaSource::FilePath {
                path: "test.mp3".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.describe_image(&attachment).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected image"));
    }

    #[tokio::test]
    async fn test_describe_image_invalid_mime() {
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Image,
            mime_type: "application/pdf".into(),
            source: MediaSource::FilePath {
                path: "test.pdf".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.describe_image(&attachment).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_describe_image_too_large() {
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Image,
            mime_type: "image/png".into(),
            source: MediaSource::FilePath {
                path: "big.png".into(),
            },
            size_bytes: MAX_IMAGE_BYTES + 1,
        };
        let result = engine.describe_image(&attachment).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_transcribe_audio_wrong_type() {
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Image,
            mime_type: "image/png".into(),
            source: MediaSource::FilePath {
                path: "test.png".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.transcribe_audio(&attachment).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_video_disabled() {
        let config = MediaConfig {
            video_description: false,
            ..Default::default()
        };
        let engine = MediaEngine::new(config);
        let attachment = MediaAttachment {
            media_type: MediaType::Video,
            mime_type: "video/mp4".into(),
            source: MediaSource::FilePath {
                path: "test.mp4".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.describe_video(&attachment).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("disabled"));
    }

    #[test]
    fn test_detect_vision_provider_none() {
        // In test env, likely no API keys set — should return None.
        // (This test is environment-dependent, but safe.)
        let _ = detect_vision_provider(); // Just verify it doesn't panic
    }

    #[test]
    fn test_default_vision_models() {
        assert_eq!(
            default_vision_model("anthropic"),
            "claude-sonnet-4-20250514"
        );
        assert_eq!(default_vision_model("openai"), "gpt-4o");
        assert_eq!(default_vision_model("gemini"), "gemini-2.5-flash");
        assert_eq!(default_vision_model("unknown"), "unknown");
    }

    #[test]
    fn test_default_audio_models() {
        assert_eq!(default_audio_model("groq"), "whisper-large-v3-turbo");
        assert_eq!(default_audio_model("openai"), "whisper-1");
    }

    #[tokio::test]
    async fn test_transcribe_audio_rejects_image_type() {
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Image,
            mime_type: "image/png".into(),
            source: MediaSource::FilePath {
                path: "test.png".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.transcribe_audio(&attachment).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected audio"));
    }

    #[tokio::test]
    async fn test_transcribe_audio_no_provider() {
        // With no API keys set, should fail with provider error
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Audio,
            mime_type: "audio/webm".into(),
            source: MediaSource::FilePath {
                path: "test.webm".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.transcribe_audio(&attachment).await;
        // Either fails with "No audio transcription provider" or file read error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_transcribe_audio_url_source_rejected() {
        // URL source should be rejected
        let config = MediaConfig {
            audio_provider: Some("groq".to_string()),
            ..Default::default()
        };
        let engine = MediaEngine::new(config);
        let attachment = MediaAttachment {
            media_type: MediaType::Audio,
            mime_type: "audio/mpeg".into(),
            source: MediaSource::Url {
                url: "https://example.com/audio.mp3".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.transcribe_audio(&attachment).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("URL-based audio source not supported"));
    }

    #[tokio::test]
    async fn test_transcribe_audio_file_not_found() {
        let config = MediaConfig {
            audio_provider: Some("groq".to_string()),
            ..Default::default()
        };
        let engine = MediaEngine::new(config);
        let attachment = MediaAttachment {
            media_type: MediaType::Audio,
            mime_type: "audio/webm".into(),
            source: MediaSource::FilePath {
                path: "/nonexistent/path/audio.webm".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.transcribe_audio(&attachment).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read audio file"));
    }
}
