mod llm;
mod tools;
mod todo;
mod skill;
mod agent;

use std::env;
use crate::agent::Agent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let base_url = env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com/v1".into());
    let api_key = env::var("OPENAI_API_KEY")
        .unwrap_or_else(|_| "".into());
    let model = env::var("MODEL_ID")
        .unwrap_or_else(|_| "deepseek-v4-flash".into());

    let workdir = env::current_dir()?;
    let client = crate::llm::ApiClient::new(&base_url, &api_key);
    let mut agent = Agent::new(client, model.clone(), workdir);

    println!("\x1b[36m┌─ Xinforge coding agent ───────────────────────┐\x1b[0m");
    println!("\x1b[36m│ model: {:<40} │\x1b[0m", model);
    println!("\x1b[36m│ workdir: {:<38} │\x1b[0m", agent.workdir.display());
    println!("\x1b[36m│ type q/exit to quit                          │\x1b[0m");
    println!("\x1b[36m└──────────────────────────────────────────────┘\x1b[0m");

    let mut messages: Vec<serde_json::Value> = Vec::new();

    loop {
        let mut input = String::new();
        print!("\x1b[36m>> \x1b[0m");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        if std::io::stdin().read_line(&mut input).is_err() { break; }

        let trimmed = input.trim();
        if trimmed.is_empty() || trimmed == "q" || trimmed == "exit" {
            break;
        }

        messages.push(crate::llm::msg_user(trimmed));

        if let Err(e) = agent.agent_loop(&mut messages).await {
            eprintln!("\x1b[31mError: {}\x1b[0m", e);
        }

        // Print final assistant text
        if let Some(last) = messages.last() {
            if last["role"].as_str() == Some("assistant") {
                if let Some(text) = last["content"].as_str().filter(|s| !s.is_empty()) {
                    println!("\x1b[32m{}\x1b[0m\n", text);
                }
            }
        }
    }

    Ok(())
}
