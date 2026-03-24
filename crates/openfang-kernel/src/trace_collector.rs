//! Trace collector — kernel-level service for distributed tracing.
//!
//! Wraps [`TraceStore`] with best-effort semantics: all operations log on error
//! but never propagate failures to callers, ensuring tracing never disrupts
//! the main execution path.

use openfang_memory::trace_store::{SpanKind, TraceRecorder, TraceSpan, TraceStore};
use tracing::{info, warn};

/// Kernel-level trace collector service.
#[derive(Clone)]
pub struct TraceCollector {
    store: TraceStore,
}

impl TraceCollector {
    /// Create a new trace collector wrapping the given store.
    pub fn new(store: TraceStore) -> Self {
        Self { store }
    }

    /// Begin a new trace. Returns the generated trace_id.
    pub fn begin_trace(&self, trigger_type: &str, agent_id: &str, agent_name: &str) -> String {
        let trace_id = uuid::Uuid::new_v4().to_string();
        if let Err(e) = self.store.create_trace(&trace_id, trigger_type, agent_id, agent_name) {
            warn!(error = %e, "Failed to create trace");
        }
        trace_id
    }

    /// End a trace with final status and token totals.
    pub fn end_trace(
        &self,
        trace_id: &str,
        status: &str,
        input_tokens: u64,
        output_tokens: u64,
        llm_calls: u64,
    ) {
        if let Err(e) =
            self.store
                .complete_trace(trace_id, status, input_tokens, output_tokens, llm_calls)
        {
            warn!(trace_id = %trace_id, error = %e, "Failed to complete trace");
        }
    }

    /// Record a span within a trace.
    pub fn record_span_data(&self, span: TraceSpan) {
        if let Err(e) = self.store.record_span(&span) {
            warn!(trace_id = %span.trace_id, span_name = %span.name, error = %e, "Failed to record span");
        }
    }

    /// Get a reference to the underlying store (for API queries).
    pub fn store(&self) -> &TraceStore {
        &self.store
    }

    /// Cleanup old traces. Returns the number of traces deleted.
    pub fn cleanup(&self, max_count: usize, max_age_secs: u64) -> usize {
        match self.store.cleanup(max_count, max_age_secs) {
            Ok(n) => {
                if n > 0 {
                    info!(removed = n, "Trace cleanup completed");
                }
                n
            }
            Err(e) => {
                warn!(error = %e, "Trace cleanup failed");
                0
            }
        }
    }
}

impl TraceRecorder for TraceCollector {
    fn record_span(&self, span: TraceSpan) {
        self.record_span_data(span);
    }
}

/// Helper to create a completed span in one call.
pub fn make_span(
    trace_id: &str,
    parent_span_id: Option<&str>,
    name: &str,
    kind: SpanKind,
    started_at: &str,
    ended_at: &str,
    duration_ms: i64,
    input: Option<String>,
    output: Option<String>,
    metadata_json: String,
    token_input: Option<u64>,
    token_output: Option<u64>,
) -> TraceSpan {
    TraceSpan {
        id: uuid::Uuid::new_v4().to_string(),
        trace_id: trace_id.to_string(),
        parent_span_id: parent_span_id.map(String::from),
        name: name.to_string(),
        kind,
        started_at: started_at.to_string(),
        ended_at: Some(ended_at.to_string()),
        duration_ms: Some(duration_ms),
        input,
        output,
        metadata_json,
        token_input,
        token_output,
    }
}
