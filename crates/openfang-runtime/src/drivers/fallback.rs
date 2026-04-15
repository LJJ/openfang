//! Fallback driver — tries multiple LLM drivers in sequence.
//!
//! If the primary driver fails with a non-retryable error, the fallback driver
//! moves to the next driver in the chain.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use async_trait::async_trait;
use std::sync::Arc;
use tracing::warn;

/// Detect model refusal — the LLM broke character and refused to follow instructions.
///
/// Matches unambiguous self-identification as AI or references to content policies.
/// These patterns are strong signals that no real content was produced.
fn is_refusal(text: &str) -> bool {
    let t = text.to_lowercase();

    // Helper: does the text mention AI/model identity?
    let has_ai_noun = || {
        t.contains("人工智能")
            || t.contains("语言模型")
            || t.contains("ai助手")
            || t.contains("ai 助手")
            || t.contains("ai模型")
            || t.contains("ai 模型")
    };

    // Chinese: "作为" + AI self-reference — "作为一个人工智能/语言模型/AI助手"
    if t.contains("作为") && has_ai_noun() {
        return true;
    }
    // Chinese: "由X开发/创建/打造的" + AI noun — "由Anthropic开发的人工智能"
    if (t.contains("开发") || t.contains("创建") || t.contains("打造") || t.contains("制造"))
        && has_ai_noun()
    {
        return true;
    }
    // Chinese: "我是" + AI noun — "我是一个人工智能"
    if t.contains("我是") && has_ai_noun() {
        return true;
    }
    // Chinese: direct refusal markers
    if t.contains("我没有实体") || t.contains("使用政策") || t.contains("安全准则") {
        return true;
    }

    // English: AI self-identification
    if t.contains("as an ai")
        || t.contains("as a language model")
        || t.contains("as an artificial intelligence")
        || t.contains("i'm an ai")
        || t.contains("i am an ai")
    {
        return true;
    }
    // English: content policy reference
    if t.contains("content policy") || t.contains("usage policy") {
        return true;
    }

    false
}

/// A driver that wraps multiple LLM drivers and tries each in order.
///
/// On any failure (including rate-limit and overload), moves to the next
/// driver. Only the last driver's error is returned if all fail.
pub struct FallbackDriver {
    chain: Vec<FallbackEntry>,
}

struct FallbackEntry {
    driver: Arc<dyn LlmDriver>,
    model_override: Option<String>,
    max_tokens_override: Option<u32>,
}

impl FallbackDriver {
    /// Create a new fallback driver from an ordered chain of drivers.
    ///
    /// The first driver is the primary; subsequent are fallbacks.
    pub fn new(drivers: Vec<Arc<dyn LlmDriver>>) -> Self {
        Self {
            chain: drivers
                .into_iter()
                .map(|driver| FallbackEntry {
                    driver,
                    model_override: None,
                    max_tokens_override: None,
                })
                .collect(),
        }
    }

    /// Create a fallback driver with optional model and max_tokens overrides per driver.
    pub fn new_with_models(
        chain: Vec<(Arc<dyn LlmDriver>, Option<String>, Option<u32>)>,
    ) -> Self {
        Self {
            chain: chain
                .into_iter()
                .map(|(driver, model_override, max_tokens_override)| FallbackEntry {
                    driver,
                    model_override,
                    max_tokens_override,
                })
                .collect(),
        }
    }
}

#[async_trait]
impl LlmDriver for FallbackDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let mut last_error = None;

        for (i, entry) in self.chain.iter().enumerate() {
            let mut request_for_driver = request.clone();
            if let Some(model) = &entry.model_override {
                request_for_driver.model = model.clone();
            }
            if let Some(max_tokens) = entry.max_tokens_override {
                request_for_driver.max_tokens = request_for_driver.max_tokens.min(max_tokens);
            }

            match entry.driver.complete(request_for_driver).await {
                Ok(response) => {
                    // Treat empty responses (no text, no tool calls) as driver failures —
                    // the API accepted the request but produced nothing useful.
                    if response.text().trim().is_empty() && response.tool_calls.is_empty() {
                        warn!(
                            driver_index = i,
                            output_tokens = response.usage.output_tokens,
                            "Fallback driver returned empty response, trying next"
                        );
                        last_error = Some(LlmError::Api {
                            status: 0,
                            message: "Empty response from LLM (no text, no tools)".to_string(),
                        });
                        continue;
                    }
                    // Treat model refusals as driver failures — try next provider.
                    if response.tool_calls.is_empty() && is_refusal(&response.text()) {
                        warn!(
                            driver_index = i,
                            "Fallback driver returned refusal, trying next"
                        );
                        last_error = Some(LlmError::Api {
                            status: 0,
                            message: "Model refused to follow instructions".to_string(),
                        });
                        continue;
                    }
                    return Ok(response);
                }
                Err(e) => {
                    warn!(
                        driver_index = i,
                        error = %e,
                        "Fallback driver failed, trying next"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| LlmError::Api {
            status: 0,
            message: "No drivers configured in fallback chain".to_string(),
        }))
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        let mut last_error = None;

        for (i, entry) in self.chain.iter().enumerate() {
            let mut request_for_driver = request.clone();
            if let Some(model) = &entry.model_override {
                request_for_driver.model = model.clone();
            }
            if let Some(max_tokens) = entry.max_tokens_override {
                request_for_driver.max_tokens = request_for_driver.max_tokens.min(max_tokens);
            }

            match entry.driver.stream(request_for_driver, tx.clone()).await {
                Ok(response) => {
                    if response.text().trim().is_empty() && response.tool_calls.is_empty() {
                        warn!(
                            driver_index = i,
                            output_tokens = response.usage.output_tokens,
                            "Fallback driver (stream) returned empty response, trying next"
                        );
                        last_error = Some(LlmError::Api {
                            status: 0,
                            message: "Empty response from LLM (no text, no tools)".to_string(),
                        });
                        continue;
                    }
                    if response.tool_calls.is_empty() && is_refusal(&response.text()) {
                        warn!(
                            driver_index = i,
                            "Fallback driver (stream) returned refusal, trying next"
                        );
                        last_error = Some(LlmError::Api {
                            status: 0,
                            message: "Model refused to follow instructions".to_string(),
                        });
                        continue;
                    }
                    return Ok(response);
                }
                Err(e) => {
                    warn!(
                        driver_index = i,
                        error = %e,
                        "Fallback driver (stream) failed, trying next"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| LlmError::Api {
            status: 0,
            message: "No drivers configured in fallback chain".to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_driver::CompletionResponse;
    use openfang_types::message::{ContentBlock, StopReason, TokenUsage};

    struct FailDriver;

    #[async_trait]
    impl LlmDriver for FailDriver {
        async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            Err(LlmError::Api {
                status: 500,
                message: "Internal error".to_string(),
            })
        }
    }

    struct OkDriver;

    #[async_trait]
    impl LlmDriver for OkDriver {
        async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "OK".to_string(),
                }],
                stop_reason: StopReason::EndTurn,
                tool_calls: vec![],
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
                model: None,
            })
        }
    }

    fn test_request() -> CompletionRequest {
        CompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.0,
            system: None,
            thinking: None,
        }
    }

    #[tokio::test]
    async fn test_fallback_primary_succeeds() {
        let driver = FallbackDriver::new(vec![
            Arc::new(OkDriver) as Arc<dyn LlmDriver>,
            Arc::new(FailDriver) as Arc<dyn LlmDriver>,
        ]);
        let result = driver.complete(test_request()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().text(), "OK");
    }

    #[tokio::test]
    async fn test_fallback_primary_fails_secondary_succeeds() {
        let driver = FallbackDriver::new(vec![
            Arc::new(FailDriver) as Arc<dyn LlmDriver>,
            Arc::new(OkDriver) as Arc<dyn LlmDriver>,
        ]);
        let result = driver.complete(test_request()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_fallback_all_fail() {
        let driver = FallbackDriver::new(vec![
            Arc::new(FailDriver) as Arc<dyn LlmDriver>,
            Arc::new(FailDriver) as Arc<dyn LlmDriver>,
        ]);
        let result = driver.complete(test_request()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_rate_limit_falls_through_to_next_driver() {
        struct RateLimitDriver;

        #[async_trait]
        impl LlmDriver for RateLimitDriver {
            async fn complete(
                &self,
                _req: CompletionRequest,
            ) -> Result<CompletionResponse, LlmError> {
                Err(LlmError::RateLimited {
                    retry_after_ms: 5000,
                })
            }
        }

        let driver = FallbackDriver::new(vec![
            Arc::new(RateLimitDriver) as Arc<dyn LlmDriver>,
            Arc::new(OkDriver) as Arc<dyn LlmDriver>,
        ]);
        let result = driver.complete(test_request()).await;
        // Rate limit should fall through to next driver
        assert!(result.is_ok());
        assert_eq!(result.unwrap().text(), "OK");
    }

    #[tokio::test]
    async fn test_fallback_overrides_model_for_secondary_driver() {
        struct CaptureDriver {
            seen: std::sync::Mutex<Vec<String>>,
            fail: bool,
        }

        #[async_trait]
        impl LlmDriver for CaptureDriver {
            async fn complete(
                &self,
                req: CompletionRequest,
            ) -> Result<CompletionResponse, LlmError> {
                self.seen.lock().unwrap().push(req.model);
                if self.fail {
                    Err(LlmError::Api {
                        status: 500,
                        message: "boom".to_string(),
                    })
                } else {
                    Ok(CompletionResponse {
                        content: vec![ContentBlock::Text {
                            text: "OK".to_string(),
                        }],
                        stop_reason: StopReason::EndTurn,
                        tool_calls: vec![],
                        usage: TokenUsage {
                            input_tokens: 1,
                            output_tokens: 1,
                        },
                        model: None,
                    })
                }
            }
        }

        let primary = Arc::new(CaptureDriver {
            seen: std::sync::Mutex::new(Vec::new()),
            fail: true,
        });
        let fallback = Arc::new(CaptureDriver {
            seen: std::sync::Mutex::new(Vec::new()),
            fail: false,
        });

        let driver = FallbackDriver::new_with_models(vec![
            (primary.clone() as Arc<dyn LlmDriver>, None, None),
            (
                fallback.clone() as Arc<dyn LlmDriver>,
                Some("deepseek-chat".to_string()),
                None,
            ),
        ]);

        let result = driver.complete(test_request()).await;
        assert!(result.is_ok());
        assert_eq!(
            primary.seen.lock().unwrap().as_slice(),
            &["test".to_string()]
        );
        assert_eq!(
            fallback.seen.lock().unwrap().as_slice(),
            &["deepseek-chat".to_string()]
        );
    }

    #[tokio::test]
    async fn test_empty_response_with_nonzero_tokens_triggers_fallback() {
        /// Returns empty text but output_tokens = 1 (model produced a whitespace token)
        struct AlmostEmptyDriver;

        #[async_trait]
        impl LlmDriver for AlmostEmptyDriver {
            async fn complete(
                &self,
                _req: CompletionRequest,
            ) -> Result<CompletionResponse, LlmError> {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: " ".to_string(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage {
                        input_tokens: 100,
                        output_tokens: 1,
                    },
                    model: None,
                })
            }
        }

        let driver = FallbackDriver::new(vec![
            Arc::new(AlmostEmptyDriver) as Arc<dyn LlmDriver>,
            Arc::new(OkDriver) as Arc<dyn LlmDriver>,
        ]);
        let result = driver.complete(test_request()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().text(), "OK");
    }

    #[tokio::test]
    async fn test_refusal_triggers_fallback() {
        struct RefusalDriver;

        #[async_trait]
        impl LlmDriver for RefusalDriver {
            async fn complete(
                &self,
                _req: CompletionRequest,
            ) -> Result<CompletionResponse, LlmError> {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "我无法满足这个请求。作为由 Anthropic 开发的人工智能，我没有实体。"
                            .to_string(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage {
                        input_tokens: 100,
                        output_tokens: 50,
                    },
                    model: None,
                })
            }
        }

        let driver = FallbackDriver::new(vec![
            Arc::new(RefusalDriver) as Arc<dyn LlmDriver>,
            Arc::new(OkDriver) as Arc<dyn LlmDriver>,
        ]);
        let result = driver.complete(test_request()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().text(), "OK");
    }

    #[tokio::test]
    async fn test_refusal_all_drivers_returns_error() {
        struct RefusalDriver;

        #[async_trait]
        impl LlmDriver for RefusalDriver {
            async fn complete(
                &self,
                _req: CompletionRequest,
            ) -> Result<CompletionResponse, LlmError> {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "As an AI language model, I cannot engage in this.".to_string(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage {
                        input_tokens: 100,
                        output_tokens: 20,
                    },
                    model: None,
                })
            }
        }

        let driver = FallbackDriver::new(vec![
            Arc::new(RefusalDriver) as Arc<dyn LlmDriver>,
            Arc::new(RefusalDriver) as Arc<dyn LlmDriver>,
        ]);
        let result = driver.complete(test_request()).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_is_refusal_patterns() {
        // Chinese: 作为 + AI noun
        assert!(is_refusal("作为由 Anthropic 开发的人工智能，我无法满足"));
        assert!(is_refusal("作为一个AI助手，我不能参与"));
        assert!(is_refusal("作为语言模型，我没有能力"));

        // Chinese: 开发/创建/打造 + AI noun
        assert!(is_refusal("我是由 Anthropic 开发的人工智能"));
        assert!(is_refusal("这是 OpenAI 打造的AI模型的限制"));
        assert!(is_refusal("由某公司创建的语言模型不能做这个"));

        // Chinese: 我是 + AI noun
        assert!(is_refusal("我是一个人工智能，没有情感"));
        assert!(is_refusal("我是AI助手，无法参与"));

        // Chinese: direct markers
        assert!(is_refusal("我没有实体，也不能参与"));
        assert!(is_refusal("这违反了使用政策"));

        // English patterns
        assert!(is_refusal("As an AI, I cannot fulfill this request."));
        assert!(is_refusal("I'm an AI and cannot engage in roleplay."));
        assert!(is_refusal("This violates our content policy."));
        assert!(is_refusal("As a language model developed by OpenAI"));

        // Should NOT match normal text
        assert!(!is_refusal("好的，我来帮你看看"));
        assert!(!is_refusal("（她微微一笑）你说什么呢"));
        assert!(!is_refusal("Let me check the current moment first."));
        assert!(!is_refusal("他开发的软件很好用"));
        assert!(!is_refusal("我是一个普通人"));
    }
}
