//! Skill registry — tracks installed skills and their tools.

use crate::bundled;
use crate::openclaw_compat;
use crate::verify::SkillVerifier;
use crate::{InstalledSkill, SkillError, SkillManifest, SkillToolDef};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tracing::{info, warn};

/// Registry of installed skills.
#[derive(Debug, Default)]
pub struct SkillRegistry {
    /// Installed skills keyed by name.
    skills: HashMap<String, InstalledSkill>,
    /// Skills directory.
    skills_dir: PathBuf,
    /// When true, no new skills can be loaded (Stable mode).
    frozen: bool,
}

impl SkillRegistry {
    fn file_modified(path: &Path) -> Option<SystemTime> {
        std::fs::metadata(path).ok()?.modified().ok()
    }

    fn should_refresh_skillmd(dir: &Path) -> bool {
        let skill_md = dir.join("SKILL.md");
        if !skill_md.exists() {
            return false;
        }

        let manifest = dir.join("skill.toml");
        let prompt_context = dir.join("prompt_context.md");

        if !manifest.exists() || !prompt_context.exists() {
            return true;
        }

        let skill_md_mtime = match Self::file_modified(&skill_md) {
            Some(time) => time,
            None => return false,
        };

        let manifest_mtime = Self::file_modified(&manifest);
        let prompt_context_mtime = Self::file_modified(&prompt_context);

        match (manifest_mtime, prompt_context_mtime) {
            (Some(manifest_mtime), Some(prompt_context_mtime)) => {
                skill_md_mtime > manifest_mtime || skill_md_mtime > prompt_context_mtime
            }
            _ => true,
        }
    }

    fn refresh_openclaw_skillmd(dir: &Path, log_prefix: &str) -> Result<(), SkillError> {
        let converted = openclaw_compat::convert_skillmd(dir)?;

        let warnings = SkillVerifier::scan_prompt_content(&converted.prompt_context);
        let has_critical = warnings
            .iter()
            .any(|w| matches!(w.severity, crate::verify::WarningSeverity::Critical));
        if has_critical {
            warn!(
                skill = %converted.manifest.skill.name,
                "BLOCKED {log_prefix}: critical prompt injection patterns"
            );
            for w in &warnings {
                warn!("  [{:?}] {}", w.severity, w.message);
            }
            return Err(SkillError::InvalidManifest(
                "critical prompt injection patterns detected".to_string(),
            ));
        }

        if !warnings.is_empty() {
            for w in &warnings {
                warn!(
                    skill = %converted.manifest.skill.name,
                    "[{:?}] {}",
                    w.severity,
                    w.message
                );
            }
        }

        info!(
            skill = %converted.manifest.skill.name,
            "Refreshing generated skill files from SKILL.md ({log_prefix})"
        );
        openclaw_compat::write_openfang_manifest(dir, &converted.manifest)?;
        openclaw_compat::write_prompt_context(dir, &converted.prompt_context)?;
        Ok(())
    }

    /// Create a new registry rooted at the given skills directory.
    pub fn new(skills_dir: PathBuf) -> Self {
        Self {
            skills: HashMap::new(),
            skills_dir,
            frozen: false,
        }
    }

    /// Create a cheap owned snapshot of this registry.
    ///
    /// Used to avoid holding `RwLockReadGuard` across `.await` points
    /// (the guard is `!Send`).
    pub fn snapshot(&self) -> SkillRegistry {
        SkillRegistry {
            skills: self.skills.clone(),
            skills_dir: self.skills_dir.clone(),
            frozen: self.frozen,
        }
    }

    /// Freeze the registry, preventing any new skills from being loaded.
    /// Used in Stable mode after initial boot.
    pub fn freeze(&mut self) {
        self.frozen = true;
        info!("Skill registry frozen — no new skills will be loaded");
    }

    /// Check if the registry is frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Load all bundled skills (compile-time embedded SKILL.md files).
    ///
    /// Called before `load_all()` so that user-installed skills with the same name
    /// can override bundled ones. Runs prompt injection scan even on bundled skills
    /// as a defense-in-depth measure.
    pub fn load_bundled(&mut self) -> usize {
        let bundled = bundled::bundled_skills();
        let mut count = 0;

        for (name, content) in &bundled {
            match bundled::parse_bundled(name, content) {
                Ok(manifest) => {
                    // Defense in depth: scan even bundled skill prompt content
                    if let Some(ref ctx) = manifest.prompt_context {
                        let warnings = SkillVerifier::scan_prompt_content(ctx);
                        let has_critical = warnings.iter().any(|w| {
                            matches!(w.severity, crate::verify::WarningSeverity::Critical)
                        });
                        if has_critical {
                            warn!(
                                skill = %manifest.skill.name,
                                "BLOCKED bundled skill: critical prompt injection patterns"
                            );
                            continue;
                        }
                    }

                    self.skills.insert(
                        manifest.skill.name.clone(),
                        InstalledSkill {
                            manifest,
                            path: PathBuf::from("<bundled>"),
                            enabled: true,
                        },
                    );
                    count += 1;
                }
                Err(e) => {
                    warn!("Failed to parse bundled skill '{name}': {e}");
                }
            }
        }

        if count > 0 {
            info!("Loaded {count} bundled skill(s)");
        }
        count
    }

    /// Load all installed skills from the skills directory.
    pub fn load_all(&mut self) -> Result<usize, SkillError> {
        if !self.skills_dir.exists() {
            return Ok(0);
        }

        let mut count = 0;
        let entries = std::fs::read_dir(&self.skills_dir)?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            if Self::should_refresh_skillmd(&path) {
                if !openclaw_compat::detect_skillmd(&path) {
                    continue;
                }
                if let Err(e) = Self::refresh_openclaw_skillmd(&path, "user skill") {
                    warn!("Failed to refresh SKILL.md at {}: {e}", path.display());
                    continue;
                }
            }

            match self.load_skill(&path) {
                Ok(_) => count += 1,
                Err(e) => {
                    warn!("Failed to load skill at {}: {e}", path.display());
                }
            }
        }

        info!("Loaded {count} skills from {}", self.skills_dir.display());
        Ok(count)
    }

    /// Load a single skill from a directory.
    pub fn load_skill(&mut self, skill_dir: &Path) -> Result<String, SkillError> {
        if self.frozen {
            return Err(SkillError::NotFound(
                "Skill registry is frozen (Stable mode)".to_string(),
            ));
        }
        let manifest_path = skill_dir.join("skill.toml");
        let toml_str = std::fs::read_to_string(&manifest_path)?;
        let manifest: SkillManifest = toml::from_str(&toml_str)?;

        let name = manifest.skill.name.clone();

        self.skills.insert(
            name.clone(),
            InstalledSkill {
                manifest,
                path: skill_dir.to_path_buf(),
                enabled: true,
            },
        );

        info!("Loaded skill: {name}");
        Ok(name)
    }

    /// Get an installed skill by name.
    pub fn get(&self, name: &str) -> Option<&InstalledSkill> {
        self.skills.get(name)
    }

    /// List all installed skills.
    pub fn list(&self) -> Vec<&InstalledSkill> {
        self.skills.values().collect()
    }

    /// Remove a skill by name.
    pub fn remove(&mut self, name: &str) -> Result<(), SkillError> {
        let skill = self
            .skills
            .remove(name)
            .ok_or_else(|| SkillError::NotFound(name.to_string()))?;

        // Remove the skill directory
        if skill.path.exists() {
            std::fs::remove_dir_all(&skill.path)?;
        }

        info!("Removed skill: {name}");
        Ok(())
    }

    /// Get all tool definitions from all enabled skills.
    pub fn all_tool_definitions(&self) -> Vec<SkillToolDef> {
        self.skills
            .values()
            .filter(|s| s.enabled)
            .flat_map(|s| s.manifest.tools.provided.iter().cloned())
            .collect()
    }

    /// Get tool definitions only from the named skills.
    pub fn tool_definitions_for_skills(&self, names: &[String]) -> Vec<SkillToolDef> {
        self.skills
            .values()
            .filter(|s| s.enabled && names.contains(&s.manifest.skill.name))
            .flat_map(|s| s.manifest.tools.provided.iter().cloned())
            .collect()
    }

    /// Return all installed skill names.
    pub fn skill_names(&self) -> Vec<String> {
        self.skills.keys().cloned().collect()
    }

    /// Find which skill provides a given tool name.
    pub fn find_tool_provider(&self, tool_name: &str) -> Option<&InstalledSkill> {
        self.skills.values().find(|s| {
            s.enabled
                && s.manifest
                    .tools
                    .provided
                    .iter()
                    .any(|t| t.name == tool_name)
        })
    }

    /// Count installed skills.
    pub fn count(&self) -> usize {
        self.skills.len()
    }

    /// Load workspace-scoped skills that override global/bundled skills.
    ///
    /// Scans subdirectories of `workspace_skills_dir` using the same loading
    /// logic as `load_all()`: auto-converts SKILL.md, runs prompt injection
    /// scan, blocks critical threats. Skills loaded here override global ones
    /// with the same name (insert semantics).
    pub fn load_workspace_skills(
        &mut self,
        workspace_skills_dir: &Path,
    ) -> Result<usize, SkillError> {
        if !workspace_skills_dir.exists() {
            return Ok(0);
        }
        if self.frozen {
            return Err(SkillError::NotFound(
                "Skill registry is frozen (Stable mode)".to_string(),
            ));
        }

        let mut count = 0;
        let entries = std::fs::read_dir(workspace_skills_dir)?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            if Self::should_refresh_skillmd(&path) {
                if !openclaw_compat::detect_skillmd(&path) {
                    continue;
                }
                if let Err(e) = Self::refresh_openclaw_skillmd(&path, "workspace skill") {
                    warn!(
                        "Failed to refresh workspace SKILL.md at {}: {e}",
                        path.display()
                    );
                    continue;
                }
            }

            match self.load_skill(&path) {
                Ok(name) => {
                    info!("Loaded workspace skill: {name}");
                    count += 1;
                }
                Err(e) => {
                    warn!("Failed to load workspace skill at {}: {e}", path.display());
                }
            }
        }

        if count > 0 {
            info!(
                "Loaded {count} workspace skill(s) from {}",
                workspace_skills_dir.display()
            );
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;
    use tempfile::TempDir;

    fn create_test_skill(dir: &Path, name: &str) {
        let skill_dir = dir.join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("skill.toml"),
            format!(
                r#"
[skill]
name = "{name}"
version = "0.1.0"
description = "Test skill"

[runtime]
type = "python"
entry = "main.py"

[[tools.provided]]
name = "{name}_tool"
description = "A test tool"
input_schema = {{ type = "object" }}
"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn test_load_all() {
        let dir = TempDir::new().unwrap();
        create_test_skill(dir.path(), "skill-a");
        create_test_skill(dir.path(), "skill-b");

        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        let count = registry.load_all().unwrap();
        assert_eq!(count, 2);
        assert_eq!(registry.count(), 2);
    }

    #[test]
    fn test_get_skill() {
        let dir = TempDir::new().unwrap();
        create_test_skill(dir.path(), "my-skill");

        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        registry.load_all().unwrap();

        let skill = registry.get("my-skill");
        assert!(skill.is_some());
        assert_eq!(skill.unwrap().manifest.skill.name, "my-skill");
    }

    #[test]
    fn test_tool_definitions() {
        let dir = TempDir::new().unwrap();
        create_test_skill(dir.path(), "alpha");
        create_test_skill(dir.path(), "beta");

        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        registry.load_all().unwrap();

        let tools = registry.all_tool_definitions();
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_find_tool_provider() {
        let dir = TempDir::new().unwrap();
        create_test_skill(dir.path(), "finder");

        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        registry.load_all().unwrap();

        assert!(registry.find_tool_provider("finder_tool").is_some());
        assert!(registry.find_tool_provider("nonexistent").is_none());
    }

    #[test]
    fn test_remove_skill() {
        let dir = TempDir::new().unwrap();
        create_test_skill(dir.path(), "removable");

        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        registry.load_all().unwrap();
        assert_eq!(registry.count(), 1);

        registry.remove("removable").unwrap();
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_empty_dir() {
        let dir = TempDir::new().unwrap();
        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        assert_eq!(registry.load_all().unwrap(), 0);
    }

    #[test]
    fn test_frozen_blocks_load() {
        let dir = TempDir::new().unwrap();
        create_test_skill(dir.path(), "blocked");

        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        registry.freeze();
        assert!(registry.is_frozen());

        // Trying to load a skill should fail
        let result = registry.load_skill(&dir.path().join("blocked"));
        assert!(result.is_err());
    }

    #[test]
    fn test_frozen_after_initial_load() {
        let dir = TempDir::new().unwrap();
        create_test_skill(dir.path(), "initial");
        create_test_skill(dir.path(), "later");

        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        // Initial load works
        registry.load_all().unwrap();
        assert_eq!(registry.count(), 2);

        // Freeze
        registry.freeze();

        // Dynamic load blocked
        create_test_skill(dir.path(), "new-skill");
        let result = registry.load_skill(&dir.path().join("new-skill"));
        assert!(result.is_err());
        // Still has the original skills
        assert_eq!(registry.count(), 2);
    }

    #[test]
    fn test_registry_auto_convert_skillmd() {
        let dir = TempDir::new().unwrap();

        // Create a SKILL.md-only skill (no skill.toml)
        let skill_dir = dir.path().join("writing-coach");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: writing-coach\ndescription: Helps improve writing\n---\n# Writing Coach\n\nHelp users write better.",
        ).unwrap();

        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        let count = registry.load_all().unwrap();
        assert_eq!(count, 1, "Should auto-convert and load the SKILL.md skill");

        let skill = registry.get("writing-coach");
        assert!(skill.is_some());
        let manifest = &skill.unwrap().manifest;
        assert_eq!(
            manifest.runtime.runtime_type,
            crate::SkillRuntime::PromptOnly
        );
        assert!(manifest.prompt_context.is_some());

        // Verify that skill.toml was written
        assert!(skill_dir.join("skill.toml").exists());
    }

    #[test]
    fn test_registry_refreshes_generated_files_when_skillmd_changes() {
        let dir = TempDir::new().unwrap();
        let skill_dir = dir.path().join("selfie-direction");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: selfie-direction\ndescription: Initial\n---\nold body",
        )
        .unwrap();

        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        registry.load_all().unwrap();

        sleep(Duration::from_secs(1));
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: selfie-direction\ndescription: Updated\n---\nnew body",
        )
        .unwrap();

        let mut refreshed = SkillRegistry::new(dir.path().to_path_buf());
        refreshed.load_all().unwrap();

        let manifest = std::fs::read_to_string(skill_dir.join("skill.toml")).unwrap();
        let prompt_context = std::fs::read_to_string(skill_dir.join("prompt_context.md")).unwrap();

        assert!(manifest.contains("description = \"Updated\""));
        assert!(prompt_context.contains("new body"));
    }
}
