//! Fallback driver — tries multiple LLM drivers in sequence.
//!
//! If the primary driver fails with a non-retryable error, the fallback driver
//! moves to the next driver in the chain.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use async_trait::async_trait;
use std::sync::Arc;
use tracing::warn;

/// A driver that wraps multiple LLM drivers and tries each in order.
///
/// On failure, moves to the next driver. Rate-limit and overload errors
/// are bubbled up for retry logic to handle.
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
                Ok(response) => return Ok(response),
                Err(e @ LlmError::RateLimited { .. }) | Err(e @ LlmError::Overloaded { .. }) => {
                    // Retryable errors — bubble up for the retry loop to handle
                    return Err(e);
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
                Ok(response) => return Ok(response),
                Err(e @ LlmError::RateLimited { .. }) | Err(e @ LlmError::Overloaded { .. }) => {
                    return Err(e);
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
    async fn test_rate_limit_bubbles_up() {
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
        // Rate limit should NOT fall through to next driver
        assert!(matches!(result, Err(LlmError::RateLimited { .. })));
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
}
