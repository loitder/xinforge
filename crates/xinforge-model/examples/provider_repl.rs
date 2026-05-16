use std::env;
use std::io::{self, Write};

use futures_util::StreamExt;
use xinforge_model::usage::TokenUsage;
use xinforge_model::{
    message::text_from_blocks, ChatRequest, ChatStreamEvent, Message, ModelRuntime, ProfileRef,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let base_url =
        env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = env::var("OPENAI_API_KEY").unwrap_or_default();
    let model_id = env::var("MODEL_ID").unwrap_or_else(|_| "gpt-5.5".to_string());

    let runtime = ModelRuntime::from_openai_config(&base_url, &api_key, &model_id);
    let profile = ProfileRef("default".to_string());
    let mut messages = vec![Message::system(
        "You are a helpful assistant in a streaming terminal chat.",
    )];
    let mut turn = 0usize;

    println!("Xinforge model provider streaming REPL");
    println!("model: {} | base_url: {}", model_id, base_url);
    println!("commands: /exit, /quit, /clear");

    loop {
        print!("\n\nyou> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/exit" | "/quit") {
            break;
        }
        if input == "/clear" {
            messages.truncate(1);
            turn = 0;
            clear_screen()?;
            println!("history cleared");
            continue;
        }

        messages.push(Message::user_text(input));

        turn += 1;
        let history_chars = history_text_chars(&messages);
        let request = ChatRequest::new(messages.clone());
        let mut stream = runtime.chat_stream(&profile, request).await?;

        print!("assistant> ");
        io::stdout().flush()?;

        let mut assistant_text = String::new();
        let mut usage = None;

        while let Some(event) = stream.next().await {
            match event? {
                ChatStreamEvent::TextDelta(delta) => {
                    assistant_text.push_str(&delta);
                    print!("{}", delta);
                    io::stdout().flush()?;
                }
                ChatStreamEvent::Finished {
                    usage: stream_usage,
                    ..
                } => {
                    if stream_usage.is_some() {
                        usage = stream_usage;
                    }
                }
                ChatStreamEvent::Usage(stream_usage) => {
                    usage = Some(stream_usage);
                }
                ChatStreamEvent::ToolCallDelta(delta) => {
                    print!(
                        "\n[tool call delta #{}{}{}]\n",
                        delta.index,
                        delta
                            .name
                            .as_deref()
                            .map(|name| format!(" {}", name))
                            .unwrap_or_default(),
                        delta
                            .arguments_delta
                            .as_deref()
                            .map(|args| format!(" args+={}", args))
                            .unwrap_or_default(),
                    );
                    io::stdout().flush()?;
                }
                ChatStreamEvent::ResponseStarted { .. } => {}
            }
        }

        println!();
        messages.push(Message::assistant_text(assistant_text));
        print_turn_usage(turn, usage.as_ref(), messages.len(), history_chars)?;
    }

    println!("\nbye");
    Ok(())
}

fn print_turn_usage(
    turn: usize,
    usage: Option<&TokenUsage>,
    message_count: usize,
    history_chars: usize,
) -> anyhow::Result<()> {
    match usage {
        Some(usage) => println!(
            "[usage turn {}] input={} cached={} output={} reasoning={} total={} cache={} history_messages={} sent_chars={}",
            turn,
            usage_field(usage.input_tokens),
            usage_field(usage.cached_input_tokens),
            usage_field(usage.output_tokens),
            usage_field(usage.reasoning_output_tokens),
            usage_field(usage.total_tokens),
            cache_status(usage),
            message_count,
            history_chars,
        ),
        None => println!(
            "[usage turn {}] unavailable history_messages={} sent_chars={}",
            turn, message_count, history_chars
        ),
    }
    io::stdout().flush()?;
    Ok(())
}

fn usage_field(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn history_text_chars(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|message| match message {
            Message::System(message) => message.content.chars().count(),
            Message::Developer(message) => message.content.chars().count(),
            Message::User(message) => text_from_blocks(&message.content).chars().count(),
            Message::Assistant(message) => text_from_blocks(&message.content).chars().count(),
            Message::Tool(message) => text_from_blocks(&message.content).chars().count(),
        })
        .sum()
}

fn cache_status(usage: &TokenUsage) -> &'static str {
    match (usage.input_tokens, usage.cached_input_tokens) {
        (_, Some(cached)) if cached > 0 => "hit",
        (Some(input), _) if input < 1024 => "ineligible(<1024)",
        (Some(_), Some(0) | None) => "miss_or_unsupported",
        _ => "unknown",
    }
}

fn clear_screen() -> anyhow::Result<()> {
    print!("\x1b[2J\x1b[H");
    io::stdout().flush()?;
    Ok(())
}
