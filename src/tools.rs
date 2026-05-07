use std::path::PathBuf;

// ── Tool definitions (for the API) ──

fn param_obj(properties: serde_json::Value, required: Vec<&str>) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

pub fn tool_defs() -> Vec<serde_json::Value> {
    vec![
        crate::llm::tool_def("bash", "Run a shell command.", param_obj(
            serde_json::json!({"command": {"type": "string", "description": "The shell command to run"}}),
            vec!["command"],
        )),
        crate::llm::tool_def("read_file", "Read file contents.", param_obj(
            serde_json::json!({"path": {"type": "string", "description": "Path to the file"}, "limit": {"type": "integer", "description": "Max lines to read"}}),
            vec!["path"],
        )),
        crate::llm::tool_def("write_file", "Write content to a file.", param_obj(
            serde_json::json!({"path": {"type": "string", "description": "Path to the file"}, "content": {"type": "string", "description": "Content to write"}}),
            vec!["path", "content"],
        )),
        crate::llm::tool_def("edit_file", "Replace exact text in a file.", param_obj(
            serde_json::json!({"path": {"type": "string", "description": "Path to the file"}, "old_text": {"type": "string", "description": "Text to find"}, "new_text": {"type": "string", "description": "Replacement text"}}),
            vec!["path", "old_text", "new_text"],
        )),
        crate::llm::tool_def("todo", "Update the task list. Track progress on multi-step tasks. Mark one in_progress before starting, completed when done.", param_obj(
            serde_json::json!({"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}),
            vec!["items"],
        )),
        crate::llm::tool_def("load_skill", "Load specialized knowledge by name before tackling unfamiliar topics.", param_obj(
            serde_json::json!({"name": {"type": "string", "description": "Skill name to load"}}),
            vec!["name"],
        )),
        crate::llm::tool_def("task", "Spawn a subagent with fresh context. It shares the filesystem but not conversation history. Returns only a summary.", param_obj(
            serde_json::json!({"prompt": {"type": "string", "description": "Task for the subagent"}, "description": {"type": "string", "description": "Short description of the task"}}),
            vec!["prompt"],
        )),
        crate::llm::tool_def("compact", "Trigger conversation compression to save context space.", param_obj(
            serde_json::json!({"focus": {"type": "string", "description": "What to preserve in the summary"}}),
            vec![],
        )),
    ]
}

// ── Tool implementations ──

pub fn safe_path(workdir: &PathBuf, p: &str) -> Result<PathBuf, String> {
    let path = workdir.join(p);
    let resolved = path.canonicalize().map_err(|e| format!("Path error: {}", e))?;
    if !resolved.starts_with(workdir) {
        return Err(format!("Path escapes workspace: {}", p));
    }
    Ok(resolved)
}

pub fn run_bash(workdir: &PathBuf, command: &str) -> String {
    let dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
    if dangerous.iter().any(|d| command.contains(d)) {
        return "Error: Dangerous command blocked".into();
    }
    match std::process::Command::new("sh")
        .arg("-c").arg(command)
        .current_dir(workdir)
        .output()
    {
        Ok(r) => {
            let out = format!("{}{}", String::from_utf8_lossy(&r.stdout), String::from_utf8_lossy(&r.stderr));
            let trimmed = out.trim().to_string();
            if trimmed.len() > 50000 { trimmed[..50000].to_string() } else if trimmed.is_empty() { "(no output)".into() } else { trimmed }
        }
        Err(e) => format!("Error: {}", e),
    }
}

pub fn run_read(workdir: &PathBuf, path: &str, limit: Option<u64>) -> String {
    match safe_path(workdir, path) {
        Ok(p) => match std::fs::read_to_string(&p) {
            Ok(text) => {
                let lines: Vec<&str> = text.lines().collect();
                if let Some(lim) = limit {
                    if (lim as usize) < lines.len() {
                        let mut result: String = lines[..lim as usize].join("\n");
                        result.push_str(&format!("\n... ({} more lines)", lines.len() - lim as usize));
                        return result;
                    }
                }
                if text.len() > 50000 { text[..50000].to_string() } else { text }
            }
            Err(e) => format!("Error: {}", e),
        },
        Err(e) => e,
    }
}

pub fn run_write(workdir: &PathBuf, path: &str, content: &str) -> String {
    match safe_path(workdir, path) {
        Ok(p) => {
            if let Some(parent) = p.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::write(&p, content) {
                Ok(_) => format!("Wrote {} bytes to {}", content.len(), path),
                Err(e) => format!("Error: {}", e),
            }
        }
        Err(e) => e,
    }
}

pub fn run_edit(workdir: &PathBuf, path: &str, old_text: &str, new_text: &str) -> String {
    match safe_path(workdir, path) {
        Ok(p) => match std::fs::read_to_string(&p) {
            Ok(content) => {
                if !content.contains(old_text) {
                    return format!("Error: Text not found in {}", path);
                }
                let new_content = content.replacen(old_text, new_text, 1);
                match std::fs::write(&p, new_content) {
                    Ok(_) => format!("Edited {}", path),
                    Err(e) => format!("Error: {}", e),
                }
            }
            Err(e) => format!("Error: {}", e),
        },
        Err(e) => e,
    }
}
