//! Session compact store — rolling summaries of evicted conversation messages.
//!
//! When messages are evicted from a session due to context overflow, they are
//! buffered here. Once the buffer reaches a threshold, an LLM compact produces
//! an incremental summary that gets injected into the system prompt so the
//! character retains awareness of earlier conversations.

use openfang_types::agent::AgentId;
use openfang_types::error::{OpenFangError, OpenFangResult};
use openfang_types::message::Message;
use rusqlite::Connection;
use std::sync::{Arc, Mutex};

/// In-memory representation of an agent's session compact state.
#[derive(Debug, Clone)]
pub struct SessionCompactState {
    pub agent_id: AgentId,
    /// Current rolling summary (empty string if no compact has run yet).
    pub summary: String,
    /// Messages evicted since the last compact, awaiting the next compact cycle.
    pub buffer: Vec<Message>,
    /// Number of messages in the buffer (matches `buffer.len()`).
    pub buffer_count: usize,
}

/// SQLite-backed store for session compact state.
#[derive(Clone)]
pub struct SessionCompactStore {
    conn: Arc<Mutex<Connection>>,
}

impl SessionCompactStore {
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Load the compact state for an agent. Returns a default empty state if
    /// no row exists yet.
    pub fn load(&self, agent_id: AgentId) -> OpenFangResult<SessionCompactState> {
        let conn = self.conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT summary, buffer, buffer_count FROM session_compacts WHERE agent_id = ?1")
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let result = stmt
            .query_row(rusqlite::params![agent_id.0.to_string()], |row| {
                let summary: String = row.get(0)?;
                let buffer_blob: Vec<u8> = row.get(1)?;
                let buffer_count: usize = row.get::<_, i64>(2)? as usize;
                Ok((summary, buffer_blob, buffer_count))
            })
            .ok();

        match result {
            Some((summary, buffer_blob, buffer_count)) => {
                let buffer: Vec<Message> = rmp_serde::from_slice(&buffer_blob)
                    .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
                Ok(SessionCompactState {
                    agent_id,
                    summary,
                    buffer,
                    buffer_count,
                })
            }
            None => Ok(SessionCompactState {
                agent_id,
                summary: String::new(),
                buffer: Vec::new(),
                buffer_count: 0,
            }),
        }
    }

    /// Append evicted messages to the buffer. Returns the new buffer_count.
    pub fn append_evicted(
        &self,
        agent_id: AgentId,
        messages: &[Message],
    ) -> OpenFangResult<usize> {
        let mut state = self.load(agent_id)?;
        state.buffer.extend(messages.iter().cloned());
        state.buffer_count = state.buffer.len();
        self.save(&state)?;
        Ok(state.buffer_count)
    }

    /// Store a new compact summary and clear the pending buffer.
    pub fn store_compact_result(
        &self,
        agent_id: AgentId,
        new_summary: &str,
    ) -> OpenFangResult<()> {
        let conn = self.conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let empty_buffer = rmp_serde::to_vec_named(&Vec::<Message>::new())
            .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO session_compacts (agent_id, summary, buffer, buffer_count, updated_at)
             VALUES (?1, ?2, ?3, 0, ?4)
             ON CONFLICT(agent_id) DO UPDATE SET summary = ?2, buffer = ?3, buffer_count = 0, updated_at = ?4",
            rusqlite::params![agent_id.0.to_string(), new_summary, empty_buffer, now],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Read just the summary for prompt injection. Returns None if no row or
    /// summary is empty.
    pub fn get_summary(&self, agent_id: AgentId) -> OpenFangResult<Option<String>> {
        let conn = self.conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let result: Option<String> = conn
            .query_row(
                "SELECT summary FROM session_compacts WHERE agent_id = ?1",
                rusqlite::params![agent_id.0.to_string()],
                |row| row.get(0),
            )
            .ok();

        match result {
            Some(s) if !s.is_empty() => Ok(Some(s)),
            _ => Ok(None),
        }
    }

    /// Delete the compact state for an agent (used during daily session reset).
    pub fn clear(&self, agent_id: AgentId) -> OpenFangResult<()> {
        let conn = self.conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM session_compacts WHERE agent_id = ?1",
            rusqlite::params![agent_id.0.to_string()],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Upsert the full state to SQLite.
    fn save(&self, state: &SessionCompactState) -> OpenFangResult<()> {
        let conn = self.conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let buffer_blob = rmp_serde::to_vec_named(&state.buffer)
            .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO session_compacts (agent_id, summary, buffer, buffer_count, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(agent_id) DO UPDATE SET summary = ?2, buffer = ?3, buffer_count = ?4, updated_at = ?5",
            rusqlite::params![
                state.agent_id.0.to_string(),
                state.summary,
                buffer_blob,
                state.buffer_count as i64,
                now,
            ],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::run_migrations;
    use openfang_types::message::MessageContent;
    use rusqlite::Connection;
    use uuid::Uuid;

    fn setup() -> (Arc<Mutex<Connection>>, SessionCompactStore) {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();
        let shared = Arc::new(Mutex::new(conn));
        let store = SessionCompactStore::new(Arc::clone(&shared));
        (shared, store)
    }

    fn agent_id() -> AgentId {
        // Use a fixed UUID for deterministic tests
        AgentId(Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap())
    }

    fn make_message(role: openfang_types::message::Role, text: &str) -> Message {
        Message {
            role,
            content: MessageContent::Text(text.to_string()),
        }
    }

    #[test]
    fn test_load_empty() {
        let (_conn, store) = setup();
        let state = store.load(agent_id()).unwrap();
        assert!(state.summary.is_empty());
        assert!(state.buffer.is_empty());
        assert_eq!(state.buffer_count, 0);
    }

    #[test]
    fn test_append_and_load() {
        let (_conn, store) = setup();
        let aid = agent_id();

        let msgs = vec![
            make_message(openfang_types::message::Role::User, "hello"),
            make_message(openfang_types::message::Role::Assistant, "hi"),
        ];
        let count = store.append_evicted(aid, &msgs).unwrap();
        assert_eq!(count, 2);

        let state = store.load(aid).unwrap();
        assert_eq!(state.buffer.len(), 2);
        assert_eq!(state.buffer_count, 2);
        assert!(state.summary.is_empty());

        // Append more
        let more = vec![make_message(openfang_types::message::Role::User, "bye")];
        let count = store.append_evicted(aid, &more).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_store_compact_clears_buffer() {
        let (_conn, store) = setup();
        let aid = agent_id();

        let msgs = vec![
            make_message(openfang_types::message::Role::User, "hello"),
        ];
        store.append_evicted(aid, &msgs).unwrap();
        store.store_compact_result(aid, "summary of conversation").unwrap();

        let state = store.load(aid).unwrap();
        assert_eq!(state.summary, "summary of conversation");
        assert!(state.buffer.is_empty());
        assert_eq!(state.buffer_count, 0);
    }

    #[test]
    fn test_get_summary() {
        let (_conn, store) = setup();
        let aid = agent_id();

        // No row yet
        assert!(store.get_summary(aid).unwrap().is_none());

        // Empty summary
        store.append_evicted(aid, &[]).unwrap();
        assert!(store.get_summary(aid).unwrap().is_none());

        // With summary
        store.store_compact_result(aid, "some summary").unwrap();
        assert_eq!(store.get_summary(aid).unwrap().unwrap(), "some summary");
    }

    #[test]
    fn test_clear() {
        let (_conn, store) = setup();
        let aid = agent_id();

        store.store_compact_result(aid, "summary").unwrap();
        store.clear(aid).unwrap();

        let state = store.load(aid).unwrap();
        assert!(state.summary.is_empty());
        assert!(state.buffer.is_empty());
    }
}
