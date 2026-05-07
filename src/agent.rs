use std::collections::HashMap;
use std::path::PathBuf;

use crate::llm;
use crate::todo::TodoManager;
use crate::skill::SkillLoader;
use crate::tools;

// ── Agent: orchestrates the coding agent loop ──

const COMPACT_THRESHOLD: usize = 50000;
const KEEP_RECENT: usize = 3;
const NAG_INTERVAL: u32 = 3;

pub struct Agent {
    pub client: llm::ApiClient,
    pub model: String,
    pub workdir: PathBuf,
    pub system: String,
    pub tools: Vec<serde_json::Value>,
    pub child_tools: Vec<serde_json::Value>,
    pub todo: TodoManager,
    pub skills: SkillLoader,
}

impl Agent {
    pub fn new(client: llm::ApiClient, model: String, workdir: PathBuf) -> Self {
        let skills_dir = workdir.join("skills");
        let skills = SkillLoader::new(&skills_dir);
        let all_tools = tools::tool_defs();
        let child_tools: Vec<_> = all_tools.iter()
            .filter(|t| t["function"]["name"].as_str() != Some("task"))
            .cloned()
            .collect();

        let system = format!(
            "You are a coding agent at {}.\n\
             Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.\n\
             Use load_skill to access specialized knowledge before tackling unfamiliar topics.\n\
             Use the task tool to delegate exploration or subtasks to a subagent.\n\
             Use the compact tool when context is getting full.\n\
             Prefer tools over prose. Act, don't explain.\n\n\
             Skills available:\n{}",
            workdir.display(),
            skills.get_descriptions(),
        );

        Self { client, model, workdir, system, tools: all_tools, child_tools, todo: TodoManager::new(), skills }
    }

    pub async fn agent_loop(&mut self, messages: &mut Vec<serde_json::Value>) -> anyhow::Result<()> {
        let mut rounds_since_todo = 0u32;
        // Seed with None for pre-existing messages (pushed by main.rs before this call).
        // reasoning_tracks[i] MUST correspond to messages[i]; if we start empty, the
        // offset-1 shift for the system prompt maps reasoning to the wrong message.
        let mut reasoning_tracks: Vec<Option<String>> = vec![None; messages.len()];

        loop {
            micro_compact(messages);

            if estimate_tokens(messages) > COMPACT_THRESHOLD {
                print!("\x1b[33m[auto_compact triggered]\x1b[0m\n");
                let compacted = auto_compact(&self.client, &self.model, messages).await?;
                *messages = compacted;
                reasoning_tracks = vec![None; messages.len()];
            }

            let mut full_messages = vec![llm::msg_system(&self.system)];
            full_messages.extend_from_slice(messages);

            // Build reasoning_map offset by 1 for the system message
            let reasoning_map: HashMap<usize, Option<String>> = reasoning_tracks
                .iter()
                .enumerate()
                .map(|(i, r)| (i + 1, r.clone()))
                .collect();

            let (text, tool_calls, reasoning_content, finish_reason) = llm::chat(
                &self.client, &self.model, &full_messages, &reasoning_map, &self.tools, 8000,
            ).await?;

            // Build assistant message
            if let Some(ref tcs) = tool_calls {
                let msg = llm::msg_assistant_tool_calls(tcs);
                messages.push(msg);
                reasoning_tracks.push(reasoning_content);
            } else if let Some(ref content) = text {
                messages.push(llm::msg_assistant(content));
                reasoning_tracks.push(reasoning_content);
            } else {
                // Empty response — push a minimal assistant message
                messages.push(llm::msg_assistant(""));
                reasoning_tracks.push(reasoning_content);
            }

            if finish_reason != "tool_calls" {
                return Ok(());
            }

            // Extract tool calls from last assistant message
            let tc_list: Vec<serde_json::Value> = messages
                .last()
                .and_then(|msg| msg["tool_calls"].as_array())
                .cloned()
                .unwrap_or_default();

            let mut tool_results = Vec::new();
            let mut used_todo = false;
            let mut should_compact = false;

            for tc in &tc_list {
                let id = tc["id"].as_str().unwrap_or("unknown");
                let args: serde_json::Value = tc["function"]["arguments"]
                    .as_str()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_default();

                let tool_name = tc["function"]["name"].as_str().unwrap_or("");
                let output = match tool_name {
                    "bash" => tools::run_bash(&self.workdir, args["command"].as_str().unwrap_or("")),
                    "read_file" => tools::run_read(&self.workdir, args["path"].as_str().unwrap_or(""), args["limit"].as_u64()),
                    "write_file" => tools::run_write(&self.workdir, args["path"].as_str().unwrap_or(""), args["content"].as_str().unwrap_or("")),
                    "edit_file" => tools::run_edit(&self.workdir, args["path"].as_str().unwrap_or(""), args["old_text"].as_str().unwrap_or(""), args["new_text"].as_str().unwrap_or("")),
                    "load_skill" => self.skills.get_content(args["name"].as_str().unwrap_or("")),
                    "todo" => {
                        used_todo = true;
                        match args["items"].as_array() {
                            Some(items) => self.todo.update(items.clone()).unwrap_or_else(|e| e),
                            None => "Error: items array required".into(),
                        }
                    }
                    "task" => {
                        let prompt = args["prompt"].as_str().unwrap_or("");
                        let desc = args["description"].as_str().unwrap_or("subtask");
                        print!("\x1b[36m> task ({}): {:.80}\x1b[0m\n", desc, prompt);
                        self.run_subagent(prompt).await.unwrap_or_else(|e| format!("Subagent error: {}", e))
                    }
                    "compact" => {
                        should_compact = true;
                        "Compressing conversation...".into()
                    }
                    _ => format!("Unknown tool: {}", tool_name),
                };
                print!("\x1b[33m> {}:\x1b[0m\n", tool_name);
                print!("{}\n", &output[..output.len().min(200)]);
                tool_results.push(llm::msg_tool(id, &output));
            }

            rounds_since_todo = if used_todo { 0 } else { rounds_since_todo + 1 };
            if rounds_since_todo >= NAG_INTERVAL {
                tool_results.push(llm::msg_user("<reminder>Update your todos.</reminder>"));
            }

            let tool_count = tool_results.len();
            messages.extend(tool_results);
            reasoning_tracks.resize(reasoning_tracks.len() + tool_count, None);

            if should_compact {
                print!("\x1b[33m[manual compact]\x1b[0m\n");
                let compacted = auto_compact(&self.client, &self.model, messages).await?;
                *messages = compacted;
                reasoning_tracks = vec![None; messages.len()];
                continue;
            }
        }
    }

    async fn run_subagent(&self, prompt: &str) -> anyhow::Result<String> {
        let sub_system = format!(
            "You are a coding subagent at {}. Complete the given task, then summarize your findings.",
            self.workdir.display()
        );
        let mut sub_messages: Vec<serde_json::Value> = vec![llm::msg_user(prompt)];
        let mut last_text = String::new();

        for _ in 0..30 {
            let mut full_msgs = vec![llm::msg_system(&sub_system)];
            full_msgs.extend_from_slice(&sub_messages);
            let empty_reasoning = HashMap::new();
            let (text, tool_calls, _, _finish_reason) = llm::chat(
                &self.client, &self.model, &full_msgs, &empty_reasoning, &self.child_tools, 8000,
            ).await?;

            if let Some(tcs) = tool_calls {
                sub_messages.push(llm::msg_assistant_tool_calls(&tcs));
                for tc in &tcs {
                    let id = tc["id"].as_str().unwrap_or("unknown");
                    let args: serde_json::Value = tc["function"]["arguments"]
                        .as_str()
                        .and_then(|s| serde_json::from_str(s).ok())
                        .unwrap_or_default();
                    let output = match tc["function"]["name"].as_str().unwrap_or("") {
                        "bash" => tools::run_bash(&self.workdir, args["command"].as_str().unwrap_or("")),
                        "read_file" => tools::run_read(&self.workdir, args["path"].as_str().unwrap_or(""), args["limit"].as_u64()),
                        "write_file" => tools::run_write(&self.workdir, args["path"].as_str().unwrap_or(""), args["content"].as_str().unwrap_or("")),
                        "edit_file" => tools::run_edit(&self.workdir, args["path"].as_str().unwrap_or(""), args["old_text"].as_str().unwrap_or(""), args["new_text"].as_str().unwrap_or("")),
                        _ => format!("Unknown tool"),
                    };
                    sub_messages.push(llm::msg_tool(id, &output));
                }
            } else if let Some(content) = text {
                sub_messages.push(llm::msg_assistant(&content));
                last_text = content;
                break;
            } else {
                break;
            }
        }
        if last_text.is_empty() {
            last_text = "(no summary)".into();
        }
        Ok(last_text)
    }
}

// ── Layer 1: micro-compact ──
fn micro_compact(messages: &mut [serde_json::Value]) {
    let tool_indices: Vec<usize> = messages.iter()
        .enumerate()
        .filter(|(_, m)| m.get("role").and_then(|v| v.as_str()) == Some("tool"))
        .map(|(i, _)| i)
        .collect();

    if tool_indices.len() <= KEEP_RECENT {
        return;
    }

    let mut name_map: HashMap<String, String> = HashMap::new();
    for m in messages.iter() {
        if let Some(tcs) = m["tool_calls"].as_array() {
            for tc in tcs {
                if let (Some(id), Some(name)) = (tc["id"].as_str(), tc["function"]["name"].as_str()) {
                    name_map.insert(id.to_string(), name.to_string());
                }
            }
        }
    }

    let to_clear = tool_indices.len() - KEEP_RECENT;
    for &idx in &tool_indices[..to_clear] {
        let tool_call_id = messages[idx]["tool_call_id"].as_str().map(|s| s.to_string());
        if let Some(ref tcid) = tool_call_id {
            if let Some(content) = messages[idx].get_mut("content") {
                let text = content.as_str().unwrap_or("");
                if text.len() <= 100 { continue; }
                let tool_name = name_map.get(tcid).map(|s| s.as_str()).unwrap_or("unknown");
                if tool_name == "read_file" { continue; }
                *content = serde_json::json!(format!("[Previous: used {}]", tool_name));
            }
        }
    }
}

// ── Layer 2/3: auto/manual compact ──
async fn auto_compact(
    client: &llm::ApiClient, model: &str, messages: &[serde_json::Value],
) -> anyhow::Result<Vec<serde_json::Value>> {
    let ts_dir = std::env::current_dir()?.join(".transcripts");
    let _ = std::fs::create_dir_all(&ts_dir);
    let ts_name = format!("transcript_{}.jsonl", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs());
    let ts_path = ts_dir.join(&ts_name);
    let jsonl: String = messages.iter()
        .map(|m| serde_json::to_string(m).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    let _ = std::fs::write(&ts_path, &jsonl);
    print!("\x1b[33m[transcript saved: {}]\x1b[0m\n", ts_path.display());

    let conv_text: String = jsonl.chars().rev().take(80000).collect::<Vec<_>>().into_iter().rev().collect();
    let summary_req = vec![
        llm::msg_user(&format!(
            "Summarize this conversation for continuity. Include: \
             1) What was accomplished, 2) Current state, 3) Key decisions made. \
             Be concise but preserve critical details.\n\n{}", conv_text
        )),
    ];
    let empty_reasoning = HashMap::new();
    let (text, _, _, _) = llm::chat(client, model, &summary_req, &empty_reasoning, &[], 2000).await?;
    let summary = text.unwrap_or_else(|| "No summary generated.".into());

    Ok(vec![
        llm::msg_user(&format!(
            "[Conversation compressed. Transcript: {}]\n\n{}", ts_name, summary
        )),
    ])
}

fn estimate_tokens(messages: &[serde_json::Value]) -> usize {
    serde_json::to_string(messages).unwrap_or_default().len() / 4
}
