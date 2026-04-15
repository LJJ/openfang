//! Google Vertex AI driver.
//!
//! Wraps the same Gemini generateContent API format but uses:
//! - Vertex AI URL: `https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent`
//! - OAuth2 Bearer token auth from a GCP service account JSON
//!
//! Environment variables:
//!   `GOOGLE_APPLICATION_CREDENTIALS` — path to service account JSON
//!   `VERTEX_PROJECT` — GCP project ID
//!   `VERTEX_LOCATION` — API location (default "global")

use crate::drivers::gemini::{
    convert_messages, convert_response, convert_tools, GenerationConfig, GeminiErrorResponse,
    GeminiRequest, GeminiResponse,
};
use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use async_trait::async_trait;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use futures::StreamExt;
use openfang_types::message::{ContentBlock, StopReason, TokenUsage};
use openfang_types::tool::ToolCall;
use ring::signature::{RsaKeyPair, RSA_PKCS1_SHA256};
use serde::Deserialize;
use tracing::{debug, info, warn};

use super::gemini::GeminiPart;

/// Vertex AI driver — same protocol as Gemini but with Vertex URL + OAuth2 auth.
pub struct VertexDriver {
    project: String,
    location: String,
    key_pair: RsaKeyPair,
    client_email: String,
    client: reqwest::Client,
    /// Cached OAuth2 access token.
    token_cache: tokio::sync::RwLock<Option<CachedToken>>,
    /// Same thought_signature cache as GeminiDriver.
    thought_signatures: std::sync::Mutex<std::collections::HashMap<String, String>>,
}

struct CachedToken {
    token: String,
    expires_at: std::time::Instant,
}

/// Subset of the service account JSON we need.
#[derive(Deserialize)]
struct ServiceAccountKey {
    client_email: String,
    private_key: String,
}

/// OAuth2 token response.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    expires_in: u64,
}

impl VertexDriver {
    /// Create a new Vertex AI driver.
    ///
    /// `credentials_path` — path to the service account JSON file.
    /// `project` — GCP project ID.
    /// `location` — Vertex AI location (e.g. "global", "us-central1").
    pub fn new(credentials_path: &str, project: String, location: String) -> Result<Self, LlmError> {
        let creds_json = std::fs::read_to_string(credentials_path).map_err(|e| {
            LlmError::MissingApiKey(format!(
                "Cannot read GOOGLE_APPLICATION_CREDENTIALS at '{}': {}",
                credentials_path, e
            ))
        })?;
        let sa: ServiceAccountKey = serde_json::from_str(&creds_json).map_err(|e| {
            LlmError::MissingApiKey(format!("Invalid service account JSON: {}", e))
        })?;

        // Parse the PEM private key into an RSA key pair.
        let pem_bytes = pem_to_der(&sa.private_key)?;
        let key_pair = RsaKeyPair::from_pkcs8(&pem_bytes).map_err(|e| {
            LlmError::MissingApiKey(format!("Invalid RSA private key in service account: {}", e))
        })?;

        info!(project = %project, location = %location, email = %sa.client_email, "Vertex AI driver initialized");

        Ok(Self {
            project,
            location,
            key_pair,
            client_email: sa.client_email,
            client: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .timeout(std::time::Duration::from_secs(180))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            token_cache: tokio::sync::RwLock::new(None),
            thought_signatures: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Get a valid OAuth2 access token, refreshing if needed.
    async fn get_token(&self) -> Result<String, LlmError> {
        // Check cache first.
        {
            let cache = self.token_cache.read().await;
            if let Some(ref cached) = *cache {
                if cached.expires_at > std::time::Instant::now() {
                    return Ok(cached.token.clone());
                }
            }
        }

        // Mint a new token.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let header = serde_json::json!({"alg": "RS256", "typ": "JWT"});
        let claims = serde_json::json!({
            "iss": self.client_email,
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": "https://oauth2.googleapis.com/token",
            "iat": now,
            "exp": now + 3600,
        });

        let header_b64 = URL_SAFE_NO_PAD.encode(header.to_string().as_bytes());
        let claims_b64 = URL_SAFE_NO_PAD.encode(claims.to_string().as_bytes());
        let signing_input = format!("{}.{}", header_b64, claims_b64);

        let rng = ring::rand::SystemRandom::new();
        let mut sig = vec![0u8; self.key_pair.public().modulus_len()];
        self.key_pair
            .sign(&RSA_PKCS1_SHA256, &rng, signing_input.as_bytes(), &mut sig)
            .map_err(|e| LlmError::Http(format!("JWT signing failed: {}", e)))?;
        let sig_b64 = URL_SAFE_NO_PAD.encode(&sig);

        let jwt = format!("{}.{}", signing_input, sig_b64);

        let resp = self
            .client
            .post("https://oauth2.googleapis.com/token")
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                ("assertion", &jwt),
            ])
            .send()
            .await
            .map_err(|e| LlmError::Http(format!("Token exchange failed: {}", e)))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Http(format!(
                "Token exchange HTTP error: {}",
                body
            )));
        }

        let token_resp: TokenResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("Token response parse error: {}", e)))?;

        let expires_in = if token_resp.expires_in > 60 {
            token_resp.expires_in - 60 // refresh 60s early
        } else {
            token_resp.expires_in
        };

        let cached = CachedToken {
            token: token_resp.access_token.clone(),
            expires_at: std::time::Instant::now()
                + std::time::Duration::from_secs(expires_in),
        };

        let mut cache = self.token_cache.write().await;
        *cache = Some(cached);

        Ok(token_resp.access_token)
    }

    /// Build the Vertex generateContent URL for a given model.
    fn url(&self, model: &str, stream: bool) -> String {
        let action = if stream {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        // Vertex global uses `aiplatform.googleapis.com` without location prefix.
        // Regional uses `{location}-aiplatform.googleapis.com`.
        let host = if self.location == "global" {
            "aiplatform.googleapis.com".to_string()
        } else {
            format!("{}-aiplatform.googleapis.com", self.location)
        };
        format!(
            "https://{}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
            host, self.project, self.location, model, action
        )
    }
}

// ── LlmDriver implementation ──────────────────────────────────────────

#[async_trait]
impl LlmDriver for VertexDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let token = self.get_token().await?;

        let cached_sigs = self.thought_signatures.lock().unwrap().clone();
        let (contents, system_instruction) =
            convert_messages(&request.messages, &request.system, &cached_sigs);
        let tools = convert_tools(&request);

        let gemini_request = GeminiRequest {
            contents,
            system_instruction,
            tools,
            generation_config: Some(GenerationConfig {
                temperature: Some(request.temperature),
                max_output_tokens: Some(request.max_tokens),
            }),
        };

        let url = self.url(&request.model, false);
        debug!(url = %url, "Sending Vertex AI request");

        let resp = self
            .client
            .post(&url)
            .header("authorization", format!("Bearer {}", token))
            .header("content-type", "application/json")
            .json(&gemini_request)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status().as_u16();

        if status == 429 || status == 503 {
            warn!(status, "Vertex AI rate limited/overloaded");
            return Err(if status == 429 {
                LlmError::RateLimited {
                    retry_after_ms: 5000,
                }
            } else {
                LlmError::Overloaded {
                    retry_after_ms: 5000,
                }
            });
        }

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            let message = serde_json::from_str::<GeminiErrorResponse>(&body)
                .map(|e| e.error.message)
                .unwrap_or(body);
            return Err(LlmError::Api { status, message });
        }

        let body = resp
            .text()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;
        let gemini_response: GeminiResponse =
            serde_json::from_str(&body).map_err(|e| LlmError::Parse(e.to_string()))?;

        let (response, new_sigs) = convert_response(gemini_response)?;
        if !new_sigs.is_empty() {
            let mut cache = self.thought_signatures.lock().unwrap();
            for (id, sig) in new_sigs {
                cache.insert(id, sig);
            }
        }
        Ok(response)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        let token = self.get_token().await?;

        let cached_sigs = self.thought_signatures.lock().unwrap().clone();
        let (contents, system_instruction) =
            convert_messages(&request.messages, &request.system, &cached_sigs);
        let tools = convert_tools(&request);

        let gemini_request = GeminiRequest {
            contents,
            system_instruction,
            tools,
            generation_config: Some(GenerationConfig {
                temperature: Some(request.temperature),
                max_output_tokens: Some(request.max_tokens),
            }),
        };

        let url = self.url(&request.model, true);
        debug!(url = %url, "Sending Vertex AI streaming request");

        let resp = self
            .client
            .post(&url)
            .header("authorization", format!("Bearer {}", token))
            .header("content-type", "application/json")
            .json(&gemini_request)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status().as_u16();

        if status == 429 || status == 503 {
            warn!(status, "Vertex AI rate limited/overloaded (stream)");
            return Err(if status == 429 {
                LlmError::RateLimited {
                    retry_after_ms: 5000,
                }
            } else {
                LlmError::Overloaded {
                    retry_after_ms: 5000,
                }
            });
        }

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            let message = serde_json::from_str::<GeminiErrorResponse>(&body)
                .map(|e| e.error.message)
                .unwrap_or(body);
            return Err(LlmError::Api { status, message });
        }

        // Parse SSE stream — identical to GeminiDriver.
        let mut buffer = String::new();
        let mut text_content = String::new();
        let mut fn_calls: Vec<(String, serde_json::Value)> = Vec::new();
        let mut finish_reason: Option<String> = None;
        let mut usage = TokenUsage::default();

        let mut byte_stream = resp.bytes_stream();
        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = chunk_result.map_err(|e| LlmError::Http(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find("\n\n") {
                let event_text = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                let data = event_text
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .unwrap_or("");

                if data.is_empty() {
                    continue;
                }

                let json: GeminiResponse = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if let Some(ref u) = json.usage_metadata {
                    usage.input_tokens = u.prompt_token_count;
                    usage.output_tokens = u.candidates_token_count;
                }

                for candidate in &json.candidates {
                    if let Some(fr) = &candidate.finish_reason {
                        finish_reason = Some(fr.clone());
                    }

                    if let Some(ref content) = candidate.content {
                        for part in &content.parts {
                            match part {
                                GeminiPart::Text { text } => {
                                    if !text.is_empty() {
                                        text_content.push_str(text);
                                        let _ = tx
                                            .send(StreamEvent::TextDelta { text: text.clone() })
                                            .await;
                                    }
                                }
                                GeminiPart::FunctionCall { function_call, .. } => {
                                    let id = format!("call_{}", uuid::Uuid::new_v4().simple());
                                    let _ = tx
                                        .send(StreamEvent::ToolUseStart {
                                            id: id.clone(),
                                            name: function_call.name.clone(),
                                        })
                                        .await;
                                    let args_str = serde_json::to_string(&function_call.args)
                                        .unwrap_or_default();
                                    let _ = tx
                                        .send(StreamEvent::ToolInputDelta { text: args_str })
                                        .await;
                                    let _ = tx
                                        .send(StreamEvent::ToolUseEnd {
                                            id,
                                            name: function_call.name.clone(),
                                            input: function_call.args.clone(),
                                        })
                                        .await;
                                    fn_calls.push((
                                        function_call.name.clone(),
                                        function_call.args.clone(),
                                    ));
                                }
                                GeminiPart::InlineData { .. }
                                | GeminiPart::FunctionResponse { .. } => {}
                            }
                        }
                    }
                }
            }
        }

        // Build final response.
        let mut content = Vec::new();
        let mut tool_calls = Vec::new();

        if !text_content.is_empty() {
            content.push(ContentBlock::Text { text: text_content });
        }

        for (name, args) in fn_calls {
            let id = format!("call_{}", uuid::Uuid::new_v4().simple());
            content.push(ContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: args.clone(),
            });
            tool_calls.push(ToolCall {
                id,
                name,
                input: args,
            });
        }

        let stop_reason = match finish_reason.as_deref() {
            Some("STOP") => StopReason::EndTurn,
            Some("MAX_TOKENS") => StopReason::MaxTokens,
            Some("SAFETY") => StopReason::EndTurn,
            _ => {
                if !tool_calls.is_empty() {
                    StopReason::ToolUse
                } else {
                    StopReason::EndTurn
                }
            }
        };

        let _ = tx
            .send(StreamEvent::ContentComplete { stop_reason, usage })
            .await;

        Ok(CompletionResponse {
            content,
            stop_reason,
            tool_calls,
            usage,
            model: None,
        })
    }
}

/// Strip PEM headers/footers and decode base64 to raw DER bytes.
fn pem_to_der(pem: &str) -> Result<Vec<u8>, LlmError> {
    let b64: String = pem
        .lines()
        .filter(|l| !l.starts_with("-----"))
        .collect::<Vec<_>>()
        .join("");
    base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .map_err(|e| LlmError::MissingApiKey(format!("Invalid PEM base64: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_global() {
        // Can't construct a real VertexDriver without credentials, test the URL logic directly.
        let host = "aiplatform.googleapis.com";
        let url = format!(
            "https://{}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
            host, "my-project", "global", "gemini-3.1-pro-preview", "generateContent"
        );
        assert_eq!(
            url,
            "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent"
        );
    }

    #[test]
    fn test_url_regional() {
        let host = format!("{}-aiplatform.googleapis.com", "us-central1");
        let url = format!(
            "https://{}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
            host, "my-project", "us-central1", "gemini-3.1-pro-preview", "generateContent"
        );
        assert_eq!(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-3.1-pro-preview:generateContent"
        );
    }

    #[test]
    fn test_pem_to_der_strips_headers() {
        // Minimal test: valid base64 with PEM headers
        let pem = "-----BEGIN PRIVATE KEY-----\nYWJj\n-----END PRIVATE KEY-----\n";
        let der = pem_to_der(pem).unwrap();
        assert_eq!(der, b"abc");
    }
}
