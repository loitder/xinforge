// Direct DeepSeek API adapter using reqwest + serde_json.
// Handles reasoning_content for thinking-mode models.

use std::collections::HashMap;

pub struct ApiClient {
    pub http: reqwest::Client,
    pub base_url: String,
    pub api_key: String,
}

impl ApiClient {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
        }
    }
}

// ── Message constructors (return serde_json::Value) ──

pub fn msg_system(text: &str) -> serde_json::Value {
    serde_json::json!({
        "role": "system",
        "content": text,
    })
}

pub fn msg_user(text: &str) -> serde_json::Value {
    serde_json::json!({
        "role": "user",
        "content": text,
    })
}

pub fn msg_assistant(text: &str) -> serde_json::Value {
    serde_json::json!({
        "role": "assistant",
        "content": text,
    })
}

pub fn msg_assistant_tool_calls(tool_calls: &[serde_json::Value]) -> serde_json::Value {
    serde_json::json!({
        "role": "assistant",
        "content": null,
        "tool_calls": tool_calls,
    })
}

pub fn msg_tool(tool_call_id: &str, content: &str) -> serde_json::Value {
    serde_json::json!({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    })
}

// ── Tool definition ──

pub fn tool_def(name: &str, description: &str, parameters: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    })
}

// ── Request helper ──

/// Call DeepSeek chat/completions.
///
/// - `messages`: the conversation so far (serde_json::Value with role/content/...)
/// - `reasoning_map`: `message_index -> reasoning_content` for assistant messages
/// - `tools`: tool definitions
///
/// Returns (content, tool_calls, reasoning_content, finish_reason).
pub async fn chat(
    client: &ApiClient,
    model: &str,
    messages: &[serde_json::Value],
    reasoning_map: &HashMap<usize, Option<String>>,
    tools: &[serde_json::Value],
    max_tokens: u32,
) -> anyhow::Result<(
    Option<String>,                        // text content
    Option<Vec<serde_json::Value>>,        // tool_calls
    Option<String>,                        // reasoning_content from response
    String,                                // finish_reason
)> {
    // Build messages with reasoning_content injected
    let messages_json: Vec<serde_json::Value> = messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            let mut m = msg.clone();
            if m.get("role").and_then(|v| v.as_str()) == Some("assistant") {
                if let Some(reasoning) = reasoning_map.get(&i).and_then(|v| v.as_ref()) {
                    m["reasoning_content"] = serde_json::json!(reasoning);
                }
            }
            m
        })
        .collect();

    let mut body = serde_json::json!({
        "model": model,
        "messages": messages_json,
        "max_tokens": max_tokens,
    });

    if !tools.is_empty() {
        body["tools"] = serde_json::json!(tools);
    }

    // Enable thinking mode for DeepSeek V4 models
    body["thinking"] = serde_json::json!({"type": "enabled"});

    let url = format!("{}/chat/completions", client.base_url);

    let response = client
        .http
        .post(&url)
        .header("Authorization", format!("Bearer {}", client.api_key))
        .json(&body)
        .send()
        .await?;

    let status = response.status();
    let response_text = response.text().await?;

    if !status.is_success() {
        anyhow::bail!("DeepSeek API error (HTTP {}): {}", status.as_u16(), response_text);
    }

    let json: serde_json::Value =
        serde_json::from_str(&response_text).map_err(|e| anyhow::anyhow!("Failed to parse response JSON: {}", e))?;

    let choice = json["choices"]
        .as_array()
        .and_then(|a| a.first())
        .ok_or_else(|| anyhow::anyhow!("No choices in response: {}", response_text))?;

    let finish_reason = choice["finish_reason"]
        .as_str()
        .unwrap_or("stop")
        .to_string();

    let message = &choice["message"];

    let content = message["content"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    let tool_calls: Option<Vec<serde_json::Value>> = message["tool_calls"]
        .as_array()
        .map(|arr| arr.clone());

    let reasoning_content = message["reasoning_content"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    Ok((content, tool_calls, reasoning_content, finish_reason))
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_serialization_has_type_wrapper() {
        let tool = tool_def("bash", "run a command", serde_json::json!({
            "command": {"type": "string"}
        }));
        assert_eq!(tool["type"].as_str(), Some("function"));
        assert!(tool["function"].is_object());
        assert_eq!(tool["function"]["name"].as_str(), Some("bash"));
    }

    #[test]
    fn msg_assistant_includes_role_and_content() {
        let msg = msg_assistant("hello world");
        assert_eq!(msg["role"].as_str(), Some("assistant"));
        assert_eq!(msg["content"].as_str(), Some("hello world"));
    }

    #[test]
    fn msg_assistant_tool_calls_has_null_content() {
        let tool_call = serde_json::json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
        });
        let msg = msg_assistant_tool_calls(&[tool_call]);
        assert_eq!(msg["role"].as_str(), Some("assistant"));
        assert_eq!(msg["content"], serde_json::Value::Null);
        assert!(msg["tool_calls"].is_array());
        assert_eq!(msg["tool_calls"][0]["id"].as_str(), Some("call_1"));
    }

    #[test]
    fn msg_tool_has_tool_call_id() {
        let msg = msg_tool("call_42", "file contents here");
        assert_eq!(msg["role"].as_str(), Some("tool"));
        assert_eq!(msg["tool_call_id"].as_str(), Some("call_42"));
        assert_eq!(msg["content"].as_str(), Some("file contents here"));
    }

    #[test]
    fn reasoning_content_injected_for_assistant() {
        let mut map = HashMap::new();
        map.insert(0, Some("thinking step 1".to_string()));
        let msg = msg_assistant("hello");
        let messages = vec![msg];
        let built: Vec<serde_json::Value> = messages
            .iter()
            .enumerate()
            .map(|(i, msg)| {
                let mut m = msg.clone();
                if m.get("role").and_then(|v| v.as_str()) == Some("assistant") {
                    if let Some(reasoning) = map.get(&i).and_then(|v| v.as_ref()) {
                        m["reasoning_content"] = serde_json::json!(reasoning);
                    }
                }
                m
            })
            .collect();
        assert_eq!(built[0]["reasoning_content"].as_str(), Some("thinking step 1"));
        assert_eq!(built[0]["content"].as_str(), Some("hello"));
    }

    #[test]
    fn reasoning_content_not_injected_for_user() {
        let mut map = HashMap::new();
        map.insert(0, Some("should not appear".to_string()));
        let msg = msg_user("hello");
        let messages = vec![msg];
        let built: Vec<serde_json::Value> = messages
            .iter()
            .enumerate()
            .map(|(i, msg)| {
                let mut m = msg.clone();
                if m.get("role").and_then(|v| v.as_str()) == Some("assistant") {
                    if let Some(reasoning) = map.get(&i).and_then(|v| v.as_ref()) {
                        m["reasoning_content"] = serde_json::json!(reasoning);
                    }
                }
                m
            })
            .collect();
        assert!(built[0].get("reasoning_content").is_none());
    }
}
