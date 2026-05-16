use std::env;
use std::io::Write;

use futures_util::StreamExt;
use xinforge_model::{ChatRequest, ChatStreamEvent, Message, ModelRuntime, ProfileRef};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let base_url =
        env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = env::var("OPENAI_API_KEY").unwrap_or_default();
    let model_id = env::var("MODEL_ID").unwrap_or_else(|_| "gpt-5.5".to_string());
    let prompt = env::args()
        .nth(1)
        .unwrap_or_else(|| "Say hello from Xinforge's model provider runtime.".to_string());

    let runtime = ModelRuntime::from_openai_config(&base_url, &api_key, &model_id);
    let request = ChatRequest::new(vec![
        Message::system("You are a concise assistant."),
        Message::user_text(prompt),
    ]);

    let mut stream = runtime
        .chat_stream(&ProfileRef("default".to_string()), request)
        .await?;
    let mut usage = None;

    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::TextDelta(delta) => {
                print!("{}", delta);
                std::io::stdout().flush()?;
            }
            ChatStreamEvent::Finished {
                usage: stream_usage,
                ..
            } => {
                usage = stream_usage;
            }
            ChatStreamEvent::Usage(stream_usage) => {
                usage = Some(stream_usage);
            }
            _ => {}
        }
    }
    println!();

    if let Some(usage) = usage {
        eprintln!(
            "\n[usage]\ninput_tokens: {}\ncached_input_tokens: {}\noutput_tokens: {}\nreasoning_output_tokens: {}\ntotal_tokens: {}",
            usage
                .input_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            usage
                .cached_input_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            usage
                .output_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            usage
                .reasoning_output_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            usage
                .total_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
        );
    }

    Ok(())
}
