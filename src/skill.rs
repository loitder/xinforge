use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// ── SkillLoader: two-layer skill injection ──
// Layer 1: skill names + descriptions in system prompt
// Layer 2: full body returned in tool_result on demand

pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
    #[allow(dead_code)]
    pub path: String,
}

pub struct SkillLoader {
    pub skills: HashMap<String, Skill>,
}

impl SkillLoader {
    pub fn new(skills_dir: &PathBuf) -> Self {
        let mut loader = Self { skills: HashMap::new() };
        loader.load_all(skills_dir);
        loader
    }

    fn load_all(&mut self, skills_dir: &PathBuf) {
        if !skills_dir.exists() {
            return;
        }
        if let Ok(entries) = fs::read_dir(skills_dir) {
            for entry in entries.flatten() {
                let skill_dir = entry.path();
                if !skill_dir.is_dir() {
                    continue;
                }
                let md_path = skill_dir.join("SKILL.md");
                if let Ok(text) = fs::read_to_string(&md_path) {
                    let (meta, body) = parse_frontmatter(&text);
                    let name = meta.get("name").cloned().unwrap_or_else(|| {
                        skill_dir.file_name().unwrap_or_default().to_string_lossy().into()
                    });
                    let description = meta.get("description").cloned().unwrap_or_else(|| "No description".into());
                    self.skills.insert(name.clone(), Skill {
                        name,
                        description,
                        body,
                        path: md_path.to_string_lossy().into(),
                    });
                }
            }
        }
    }

    /// Layer 1: short descriptions for the system prompt
    pub fn get_descriptions(&self) -> String {
        if self.skills.is_empty() {
            return "(no skills available)".into();
        }
        self.skills.values()
            .map(|s| format!("  - {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Layer 2: full skill body returned in tool_result
    pub fn get_content(&self, name: &str) -> String {
        match self.skills.get(name) {
            Some(skill) => format!("<skill name=\"{}\">\n{}\n</skill>", skill.name, skill.body),
            None => format!("Error: Unknown skill '{}'. Available: {}",
                name, self.skills.keys().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")),
        }
    }
}

/// Parse YAML frontmatter between --- delimiters (minimal, no dep needed)
fn parse_frontmatter(text: &str) -> (HashMap<String, String>, String) {
    let mut meta = HashMap::new();
    let body = if text.starts_with("---\n") {
        if let Some(rest) = text[4..].splitn(2, "\n---\n").collect::<Vec<_>>().get(1) {
            // Simple YAML parser for key: value lines
            for line in text[4..].lines() {
                if line == "---" { break; }
                if let Some((k, v)) = line.split_once(':') {
                    meta.insert(k.trim().to_string(), v.trim().to_string());
                }
            }
            rest.to_string()
        } else {
            text.to_string()
        }
    } else {
        text.to_string()
    };
    (meta, body.trim().to_string())
}
