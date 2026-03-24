//! Embedded WebChat UI served as static HTML.
//!
//! In **release** mode the dashboard is assembled at compile time from separate
//! HTML/CSS/JS files under `static/` using `include_str!()`.  This keeps
//! single-binary deployment while allowing organized source files.
//!
//! In **debug** mode the same files are read from disk at runtime so that
//! changes take effect on browser refresh — no recompilation required.

use axum::http::header;
use axum::response::IntoResponse;

// ── Release mode: compile-time embedding ────────────────────────────────

#[cfg(not(debug_assertions))]
mod release {
    use super::*;

    const ETAG: &str = concat!("\"openfang-", env!("CARGO_PKG_VERSION"), "\"");
    const LOGO_PNG: &[u8] = include_bytes!("../static/logo.png");
    const FAVICON_ICO: &[u8] = include_bytes!("../static/favicon.ico");

    pub async fn logo_png() -> impl IntoResponse {
        (
            [
                (header::CONTENT_TYPE, "image/png"),
                (header::CACHE_CONTROL, "public, max-age=86400, immutable"),
            ],
            LOGO_PNG,
        )
    }

    pub async fn favicon_ico() -> impl IntoResponse {
        (
            [
                (header::CONTENT_TYPE, "image/x-icon"),
                (header::CACHE_CONTROL, "public, max-age=86400, immutable"),
            ],
            FAVICON_ICO,
        )
    }

    pub async fn webchat_page() -> impl IntoResponse {
        (
            [
                (header::CONTENT_TYPE, "text/html; charset=utf-8"),
                (header::ETAG, ETAG),
                (
                    header::CACHE_CONTROL,
                    "public, max-age=3600, must-revalidate",
                ),
            ],
            WEBCHAT_HTML,
        )
    }

    const WEBCHAT_HTML: &str = concat!(
        include_str!("../static/index_head.html"),
        "<style>\n",
        include_str!("../static/css/theme.css"),
        "\n",
        include_str!("../static/css/layout.css"),
        "\n",
        include_str!("../static/css/components.css"),
        "\n",
        include_str!("../static/vendor/github-dark.min.css"),
        "\n</style>\n",
        include_str!("../static/index_body.html"),
        "<script>\n",
        include_str!("../static/vendor/marked.min.js"),
        "\n</script>\n",
        "<script>\n",
        include_str!("../static/vendor/highlight.min.js"),
        "\n</script>\n",
        "<script>\n",
        include_str!("../static/js/api.js"),
        "\n",
        include_str!("../static/js/app.js"),
        "\n",
        include_str!("../static/js/pages/overview.js"),
        "\n",
        include_str!("../static/js/pages/chat.js"),
        "\n",
        include_str!("../static/js/pages/agents.js"),
        "\n",
        include_str!("../static/js/pages/workflows.js"),
        "\n",
        include_str!("../static/js/pages/workflow-builder.js"),
        "\n",
        include_str!("../static/js/pages/channels.js"),
        "\n",
        include_str!("../static/js/pages/skills.js"),
        "\n",
        include_str!("../static/js/pages/hands.js"),
        "\n",
        include_str!("../static/js/pages/scheduler.js"),
        "\n",
        include_str!("../static/js/pages/settings.js"),
        "\n",
        include_str!("../static/js/pages/usage.js"),
        "\n",
        include_str!("../static/js/pages/sessions.js"),
        "\n",
        include_str!("../static/js/pages/logs.js"),
        "\n",
        include_str!("../static/js/pages/wizard.js"),
        "\n",
        include_str!("../static/js/pages/approvals.js"),
        "\n",
        include_str!("../static/js/pages/traces.js"),
        "\n",
        include_str!("../static/js/pages/characters.js"),
        "\n</script>\n",
        // Alpine.js MUST be last — it processes x-data and fires alpine:init
        "<script>\n",
        include_str!("../static/vendor/alpine.min.js"),
        "\n</script>\n",
        "</body></html>"
    );
}

#[cfg(not(debug_assertions))]
pub use release::*;

// ── Debug mode: runtime disk reads ──────────────────────────────────────

#[cfg(debug_assertions)]
mod debug {
    use super::*;
    use std::path::{Path, PathBuf};

    fn static_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("static")
    }

    fn read_text(rel: &str) -> String {
        let path = static_dir().join(rel);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| format!("/* [debug] failed to read {}: {} */", rel, e))
    }

    fn read_bytes(rel: &str) -> Vec<u8> {
        let path = static_dir().join(rel);
        std::fs::read(&path).unwrap_or_default()
    }

    pub async fn logo_png() -> impl IntoResponse {
        (
            [
                (header::CONTENT_TYPE, "image/png"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            read_bytes("logo.png"),
        )
    }

    pub async fn favicon_ico() -> impl IntoResponse {
        (
            [
                (header::CONTENT_TYPE, "image/x-icon"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            read_bytes("favicon.ico"),
        )
    }

    pub async fn webchat_page() -> impl IntoResponse {
        (
            [
                (header::CONTENT_TYPE, "text/html; charset=utf-8"),
                (header::ETAG, "\"debug\""),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            assemble_html(),
        )
    }

    fn assemble_html() -> String {
        let r = read_text;
        [
            r("index_head.html"),
            "<style>\n".into(),
            r("css/theme.css"),
            "\n".into(),
            r("css/layout.css"),
            "\n".into(),
            r("css/components.css"),
            "\n".into(),
            r("vendor/github-dark.min.css"),
            "\n</style>\n".into(),
            r("index_body.html"),
            "<script>\n".into(),
            r("vendor/marked.min.js"),
            "\n</script>\n".into(),
            "<script>\n".into(),
            r("vendor/highlight.min.js"),
            "\n</script>\n".into(),
            "<script>\n".into(),
            r("js/api.js"),
            "\n".into(),
            r("js/app.js"),
            "\n".into(),
            r("js/pages/overview.js"),
            "\n".into(),
            r("js/pages/chat.js"),
            "\n".into(),
            r("js/pages/agents.js"),
            "\n".into(),
            r("js/pages/workflows.js"),
            "\n".into(),
            r("js/pages/workflow-builder.js"),
            "\n".into(),
            r("js/pages/channels.js"),
            "\n".into(),
            r("js/pages/skills.js"),
            "\n".into(),
            r("js/pages/hands.js"),
            "\n".into(),
            r("js/pages/scheduler.js"),
            "\n".into(),
            r("js/pages/settings.js"),
            "\n".into(),
            r("js/pages/usage.js"),
            "\n".into(),
            r("js/pages/sessions.js"),
            "\n".into(),
            r("js/pages/logs.js"),
            "\n".into(),
            r("js/pages/wizard.js"),
            "\n".into(),
            r("js/pages/approvals.js"),
            "\n".into(),
            r("js/pages/traces.js"),
            "\n".into(),
            r("js/pages/characters.js"),
            "\n</script>\n".into(),
            // Alpine.js MUST be last
            "<script>\n".into(),
            r("vendor/alpine.min.js"),
            "\n</script>\n</body></html>".into(),
        ]
        .join("")
    }
}

#[cfg(debug_assertions)]
pub use debug::*;
