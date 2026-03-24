//! Trace store — records execution traces and spans for distributed tracing.

use chrono::Utc;
use openfang_types::error::{OpenFangError, OpenFangResult};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::{Arc, Mutex};

// ── Data types ────────────────────────────────────────────────────────

/// Kind of a trace span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpanKind {
    Hook,
    Llm,
    Tool,
    Custom,
}

impl fmt::Display for SpanKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpanKind::Hook => write!(f, "hook"),
            SpanKind::Llm => write!(f, "llm"),
            SpanKind::Tool => write!(f, "tool"),
            SpanKind::Custom => write!(f, "custom"),
        }
    }
}

impl SpanKind {
    pub fn from_str(s: &str) -> Self {
        match s {
            "hook" => SpanKind::Hook,
            "llm" => SpanKind::Llm,
            "tool" => SpanKind::Tool,
            _ => SpanKind::Custom,
        }
    }
}

/// A single span within a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    pub id: String,
    pub trace_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub kind: SpanKind,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub duration_ms: Option<i64>,
    pub input: Option<String>,
    pub output: Option<String>,
    pub metadata_json: String,
    pub token_input: Option<u64>,
    pub token_output: Option<u64>,
}

/// Summary of a trace (for list views).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSummary {
    pub id: String,
    pub trigger_type: String,
    pub agent_id: String,
    pub agent_name: String,
    pub status: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub total_duration_ms: Option<i64>,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_llm_calls: u64,
}

/// Full trace detail with all spans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceDetail {
    pub trace: TraceSummary,
    pub spans: Vec<TraceSpan>,
}

// ── TraceRecorder trait ───────────────────────────────────────────────

/// Trait for recording trace spans. Implemented by TraceCollector in the kernel
/// crate, used by the runtime crate via task-local context.
pub trait TraceRecorder: Send + Sync {
    fn record_span(&self, span: TraceSpan);
}

// ── TraceStore ────────────────────────────────────────────────────────

/// Trace store backed by SQLite.
#[derive(Clone)]
pub struct TraceStore {
    conn: Arc<Mutex<Connection>>,
}

impl TraceStore {
    /// Create a new trace store wrapping the given connection.
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Create a new trace record.
    pub fn create_trace(
        &self,
        id: &str,
        trigger_type: &str,
        agent_id: &str,
        agent_name: &str,
    ) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO traces (id, trigger_type, agent_id, agent_name, status, started_at)
             VALUES (?1, ?2, ?3, ?4, 'running', ?5)",
            rusqlite::params![id, trigger_type, agent_id, agent_name, now],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Complete a trace with final status and token totals.
    pub fn complete_trace(
        &self,
        id: &str,
        status: &str,
        total_input_tokens: u64,
        total_output_tokens: u64,
        total_llm_calls: u64,
    ) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE traces SET
                status = ?2,
                ended_at = ?3,
                total_duration_ms = CAST((julianday(?3) - julianday(started_at)) * 86400000 AS INTEGER),
                total_input_tokens = ?4,
                total_output_tokens = ?5,
                total_llm_calls = ?6
             WHERE id = ?1",
            rusqlite::params![
                id,
                status,
                now,
                total_input_tokens as i64,
                total_output_tokens as i64,
                total_llm_calls as i64,
            ],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Record a span within a trace.
    pub fn record_span(&self, span: &TraceSpan) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO trace_spans (id, trace_id, parent_span_id, name, kind, started_at, ended_at, duration_ms, input, output, metadata_json, token_input, token_output)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            rusqlite::params![
                span.id,
                span.trace_id,
                span.parent_span_id,
                span.name,
                span.kind.to_string(),
                span.started_at,
                span.ended_at,
                span.duration_ms,
                span.input,
                span.output,
                span.metadata_json,
                span.token_input.map(|v| v as i64),
                span.token_output.map(|v| v as i64),
            ],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// List traces with pagination and optional filters.
    pub fn list_traces(
        &self,
        limit: usize,
        offset: usize,
        agent_filter: Option<&str>,
        trigger_filter: Option<&str>,
    ) -> OpenFangResult<(Vec<TraceSummary>, u64)> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;

        let mut where_clauses = Vec::new();
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(agent) = agent_filter {
            where_clauses.push(format!("agent_name = ?{}", params.len() + 1));
            params.push(Box::new(agent.to_string()));
        }
        if let Some(trigger) = trigger_filter {
            where_clauses.push(format!("trigger_type = ?{}", params.len() + 1));
            params.push(Box::new(trigger.to_string()));
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        // Count total
        let count_sql = format!("SELECT COUNT(*) FROM traces {where_sql}");
        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();
        let total: u64 = conn
            .query_row(&count_sql, params_refs.as_slice(), |row| {
                row.get::<_, i64>(0).map(|v| v as u64)
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        // Query with pagination
        let query_sql = format!(
            "SELECT id, trigger_type, agent_id, agent_name, status, started_at, ended_at,
                    total_duration_ms, total_input_tokens, total_output_tokens, total_llm_calls
             FROM traces {where_sql}
             ORDER BY started_at DESC
             LIMIT ?{} OFFSET ?{}",
            params.len() + 1,
            params.len() + 2,
        );
        params.push(Box::new(limit as i64));
        params.push(Box::new(offset as i64));

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&query_sql)
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let rows = stmt
            .query_map(params_refs.as_slice(), |row| {
                Ok(TraceSummary {
                    id: row.get(0)?,
                    trigger_type: row.get(1)?,
                    agent_id: row.get(2)?,
                    agent_name: row.get(3)?,
                    status: row.get(4)?,
                    started_at: row.get(5)?,
                    ended_at: row.get(6)?,
                    total_duration_ms: row.get(7)?,
                    total_input_tokens: row.get::<_, i64>(8).map(|v| v as u64)?,
                    total_output_tokens: row.get::<_, i64>(9).map(|v| v as u64)?,
                    total_llm_calls: row.get::<_, i64>(10).map(|v| v as u64)?,
                })
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| OpenFangError::Memory(e.to_string()))?);
        }
        Ok((results, total))
    }

    /// Get a trace with all its spans.
    pub fn get_trace(&self, id: &str) -> OpenFangResult<Option<TraceDetail>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;

        // Get trace
        let trace = conn
            .query_row(
                "SELECT id, trigger_type, agent_id, agent_name, status, started_at, ended_at,
                        total_duration_ms, total_input_tokens, total_output_tokens, total_llm_calls
                 FROM traces WHERE id = ?1",
                rusqlite::params![id],
                |row| {
                    Ok(TraceSummary {
                        id: row.get(0)?,
                        trigger_type: row.get(1)?,
                        agent_id: row.get(2)?,
                        agent_name: row.get(3)?,
                        status: row.get(4)?,
                        started_at: row.get(5)?,
                        ended_at: row.get(6)?,
                        total_duration_ms: row.get(7)?,
                        total_input_tokens: row.get::<_, i64>(8).map(|v| v as u64)?,
                        total_output_tokens: row.get::<_, i64>(9).map(|v| v as u64)?,
                        total_llm_calls: row.get::<_, i64>(10).map(|v| v as u64)?,
                    })
                },
            )
            .optional()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let trace = match trace {
            Some(t) => t,
            None => return Ok(None),
        };

        // Get spans
        let mut stmt = conn
            .prepare(
                "SELECT id, trace_id, parent_span_id, name, kind, started_at, ended_at,
                        duration_ms, input, output, metadata_json, token_input, token_output
                 FROM trace_spans WHERE trace_id = ?1
                 ORDER BY started_at ASC",
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let rows = stmt
            .query_map(rusqlite::params![id], |row| {
                Ok(TraceSpan {
                    id: row.get(0)?,
                    trace_id: row.get(1)?,
                    parent_span_id: row.get(2)?,
                    name: row.get(3)?,
                    kind: SpanKind::from_str(
                        &row.get::<_, String>(4).unwrap_or_else(|_| "custom".to_string()),
                    ),
                    started_at: row.get(5)?,
                    ended_at: row.get(6)?,
                    duration_ms: row.get(7)?,
                    input: row.get(8)?,
                    output: row.get(9)?,
                    metadata_json: row.get::<_, String>(10).unwrap_or_else(|_| "{}".to_string()),
                    token_input: row.get::<_, Option<i64>>(11).ok().flatten().map(|v| v as u64),
                    token_output: row.get::<_, Option<i64>>(12).ok().flatten().map(|v| v as u64),
                })
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let mut spans = Vec::new();
        for row in rows {
            spans.push(row.map_err(|e| OpenFangError::Memory(e.to_string()))?);
        }

        Ok(Some(TraceDetail { trace, spans }))
    }

    /// Delete traces older than `max_age_secs` or exceeding `max_count`.
    /// Returns the number of traces deleted.
    pub fn cleanup(&self, max_count: usize, max_age_secs: u64) -> OpenFangResult<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;

        // Collect trace IDs to delete: older than max_age OR beyond max_count
        let mut ids_to_delete: Vec<String> = Vec::new();

        // By age
        {
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT id FROM traces WHERE started_at < datetime('now', '-{max_age_secs} seconds')"
                ))
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            for row in rows {
                if let Ok(id) = row {
                    ids_to_delete.push(id);
                }
            }
        }

        // By count: keep only the newest max_count
        {
            let mut stmt = conn
                .prepare(
                    "SELECT id FROM traces ORDER BY started_at DESC LIMIT -1 OFFSET ?1",
                )
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            let rows = stmt
                .query_map(rusqlite::params![max_count as i64], |row| {
                    row.get::<_, String>(0)
                })
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            for row in rows {
                if let Ok(id) = row {
                    if !ids_to_delete.contains(&id) {
                        ids_to_delete.push(id);
                    }
                }
            }
        }

        if ids_to_delete.is_empty() {
            return Ok(0);
        }

        // Delete spans first, then traces
        let placeholders: Vec<String> = ids_to_delete
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect();
        let placeholder_str = placeholders.join(",");

        let params_refs: Vec<&dyn rusqlite::types::ToSql> = ids_to_delete
            .iter()
            .map(|s| s as &dyn rusqlite::types::ToSql)
            .collect();

        conn.execute(
            &format!("DELETE FROM trace_spans WHERE trace_id IN ({placeholder_str})"),
            params_refs.as_slice(),
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let deleted = conn
            .execute(
                &format!("DELETE FROM traces WHERE id IN ({placeholder_str})"),
                params_refs.as_slice(),
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::run_migrations;

    fn setup() -> TraceStore {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();
        TraceStore::new(Arc::new(Mutex::new(conn)))
    }

    #[test]
    fn test_create_and_list_traces() {
        let store = setup();

        store
            .create_trace("t1", "user", "agent-1", "assistant")
            .unwrap();
        store
            .create_trace("t2", "tick", "agent-1", "assistant")
            .unwrap();

        let (traces, total) = store.list_traces(50, 0, None, None).unwrap();
        assert_eq!(total, 2);
        assert_eq!(traces.len(), 2);
        // Most recent first
        assert_eq!(traces[0].status, "running");
    }

    #[test]
    fn test_complete_trace() {
        let store = setup();

        store
            .create_trace("t1", "user", "agent-1", "assistant")
            .unwrap();
        store
            .complete_trace("t1", "completed", 1000, 500, 2)
            .unwrap();

        let detail = store.get_trace("t1").unwrap().unwrap();
        assert_eq!(detail.trace.status, "completed");
        assert_eq!(detail.trace.total_input_tokens, 1000);
        assert_eq!(detail.trace.total_output_tokens, 500);
        assert_eq!(detail.trace.total_llm_calls, 2);
        assert!(detail.trace.ended_at.is_some());
    }

    #[test]
    fn test_record_and_get_spans() {
        let store = setup();

        store
            .create_trace("t1", "user", "agent-1", "assistant")
            .unwrap();

        let span = TraceSpan {
            id: "s1".to_string(),
            trace_id: "t1".to_string(),
            parent_span_id: None,
            name: "llm:claude-opus".to_string(),
            kind: SpanKind::Llm,
            started_at: Utc::now().to_rfc3339(),
            ended_at: Some(Utc::now().to_rfc3339()),
            duration_ms: Some(2100),
            input: Some("{\"messages\":[]}".to_string()),
            output: Some("Hello world".to_string()),
            metadata_json: "{}".to_string(),
            token_input: Some(500),
            token_output: Some(100),
        };
        store.record_span(&span).unwrap();

        let detail = store.get_trace("t1").unwrap().unwrap();
        assert_eq!(detail.spans.len(), 1);
        assert_eq!(detail.spans[0].name, "llm:claude-opus");
        assert_eq!(detail.spans[0].kind, SpanKind::Llm);
        assert_eq!(detail.spans[0].token_input, Some(500));
    }

    #[test]
    fn test_list_with_filters() {
        let store = setup();

        store
            .create_trace("t1", "user", "a1", "assistant")
            .unwrap();
        store
            .create_trace("t2", "tick", "a1", "assistant")
            .unwrap();
        store
            .create_trace("t3", "user", "a2", "ziling")
            .unwrap();

        let (traces, total) = store
            .list_traces(50, 0, Some("assistant"), None)
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(traces.len(), 2);

        let (traces, total) = store
            .list_traces(50, 0, None, Some("tick"))
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(traces[0].trigger_type, "tick");
    }

    #[test]
    fn test_get_nonexistent_trace() {
        let store = setup();
        let result = store.get_trace("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_cleanup_by_count() {
        let store = setup();

        for i in 0..5 {
            store
                .create_trace(&format!("t{i}"), "user", "a1", "assistant")
                .unwrap();
            // Record a span for each
            store
                .record_span(&TraceSpan {
                    id: format!("s{i}"),
                    trace_id: format!("t{i}"),
                    parent_span_id: None,
                    name: "test".to_string(),
                    kind: SpanKind::Custom,
                    started_at: Utc::now().to_rfc3339(),
                    ended_at: None,
                    duration_ms: None,
                    input: None,
                    output: None,
                    metadata_json: "{}".to_string(),
                    token_input: None,
                    token_output: None,
                })
                .unwrap();
        }

        // Keep only 3
        let deleted = store.cleanup(3, 999999).unwrap();
        assert_eq!(deleted, 2);

        let (traces, total) = store.list_traces(50, 0, None, None).unwrap();
        assert_eq!(total, 3);
        assert_eq!(traces.len(), 3);
    }

    #[test]
    fn test_cleanup_nothing_to_delete() {
        let store = setup();
        store
            .create_trace("t1", "user", "a1", "assistant")
            .unwrap();
        let deleted = store.cleanup(100, 86400).unwrap();
        assert_eq!(deleted, 0);
    }
}
