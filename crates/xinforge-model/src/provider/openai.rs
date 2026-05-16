use std::pin::Pin;

use futures_util::StreamExt;

use crate::message::{text_from_blocks, AssistantMessage, ContentBlock, Message, ProviderMeta};
use crate::profile::{ModelCapabilities, ModelProfile, ReasoningSupport};
use crate::provider::{ModelProvider, ProviderId};
use crate::request::{ChatRequest, ToolChoice};
use crate::response::{ChatResponse, ChatStream, ChatStreamEvent, FinishReason, ToolCallDelta};
use crate::tool::{ToolCall, ToolCallId, ToolSpec};
use crate::usage::TokenUsage;
use crate::{ModelError, ModelResult};

pub struct OpenAiProvider {
    id: ProviderId,
    http: reqwest::Client,
}

impl OpenAiProvider {
    pub fn new() -> Self {
        Self {
            id: ProviderId("openai".to_string()),
            http: reqwest::Client::new(),
        }
    }
}

impl ModelProvider for OpenAiProvider {
    fn id(&self) -> ProviderId {
        self.id.clone()
    }

    fn capabilities(&self, _model_id: &str) -> ModelCapabilities {
        ModelCapabilities {
            streaming: true,
            tools: true,
            parallel_tool_calls: true,
            reasoning: ReasoningSupport::None,
            vision: false,
            max_context_tokens: None,
            max_output_tokens: None,
        }
    }

    fn chat<'a>(
        &'a self,
        profile: &'a ModelProfile,
        request: ChatRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = ModelResult<ChatResponse>> + Send + 'a>> {
        Box::pin(async move {
            if profile.api_key.is_empty() {
                return Err(ModelError::MissingApiKey);
            }

            let base_url = profile
                .base_url
                .as_deref()
                .unwrap_or("https://api.openai.com/v1")
                .trim_end_matches('/');
            let url = format!("{}/chat/completions", base_url);
            let body = openai_request_body(profile, &request)?;

            let response = self
                .http
                .post(&url)
                .header("Authorization", format!("Bearer {}", profile.api_key))
                .json(&body)
                .send()
                .await?;

            let status = response.status();
            let response_text = response.text().await?;
            if !status.is_success() {
                if status.as_u16() == 429 {
                    return Err(ModelError::RateLimited {
                        retry_after_ms: None,
                    });
                }
                return Err(ModelError::ProviderHttp {
                    status: status.as_u16(),
                    body: response_text,
                });
            }

            parse_openai_response(&self.id, &profile.model_id, &response_text)
        })
    }

    fn chat_stream<'a>(
        &'a self,
        profile: &'a ModelProfile,
        request: ChatRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = ModelResult<ChatStream>> + Send + 'a>> {
        Box::pin(async move {
            if profile.api_key.is_empty() {
                return Err(ModelError::MissingApiKey);
            }

            let base_url = profile
                .base_url
                .as_deref()
                .unwrap_or("https://api.openai.com/v1")
                .trim_end_matches('/');
            let url = format!("{}/chat/completions", base_url);
            let mut body = openai_request_body(profile, &request)?;
            body["stream"] = serde_json::json!(true);
            body["stream_options"] = serde_json::json!({"include_usage": true});

            let response = self
                .http
                .post(&url)
                .header("Authorization", format!("Bearer {}", profile.api_key))
                .json(&body)
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let response_text = response.text().await?;
                if status.as_u16() == 429 {
                    return Err(ModelError::RateLimited {
                        retry_after_ms: None,
                    });
                }
                return Err(ModelError::ProviderHttp {
                    status: status.as_u16(),
                    body: response_text,
                });
            }

            let provider = self.id.clone();
            let fallback_model = profile.model_id.clone();
            let mut byte_stream = response.bytes_stream();
            let stream = async_stream::try_stream! {
                let mut buffer = String::new();
                let mut started = false;

                while let Some(chunk) = byte_stream.next().await {
                    let bytes = chunk?;
                    let chunk_text = std::str::from_utf8(&bytes).map_err(|error| {
                        ModelError::StreamParse {
                            detail: format!("provider returned non-UTF8 stream data: {}", error),
                        }
                    })?;
                    buffer.push_str(chunk_text);

                    while let Some(frame) = next_sse_frame(&mut buffer) {
                        for payload in sse_data_payloads(&frame) {
                            if payload == "[DONE]" {
                                return;
                            }

                            for event in parse_openai_stream_chunk(&provider, &fallback_model, payload, &mut started)? {
                                yield event;
                            }
                        }
                    }
                }

                if !buffer.trim().is_empty() {
                    for payload in sse_data_payloads(&buffer) {
                        if payload != "[DONE]" {
                            for event in parse_openai_stream_chunk(&provider, &fallback_model, payload, &mut started)? {
                                yield event;
                            }
                        }
                    }
                }
            };

            Ok(Box::pin(stream) as ChatStream)
        })
    }
}

fn openai_request_body(
    profile: &ModelProfile,
    request: &ChatRequest,
) -> ModelResult<serde_json::Value> {
    let mut body = serde_json::json!({
        "model": profile.model_id,
        "messages": messages_to_openai(&request.messages)?,
    });

    if let Some(max_tokens) = request.max_output_tokens {
        body["max_tokens"] = serde_json::json!(max_tokens);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = serde_json::json!(temperature);
    }
    if !request.tools.is_empty() {
        body["tools"] = serde_json::json!(tools_to_openai(&request.tools));
    }
    match request.tool_choice {
        ToolChoice::Auto => {}
        ToolChoice::None => body["tool_choice"] = serde_json::json!("none"),
    }

    Ok(body)
}

pub fn messages_to_openai(messages: &[Message]) -> ModelResult<Vec<serde_json::Value>> {
    messages.iter().map(message_to_openai).collect()
}

fn message_to_openai(message: &Message) -> ModelResult<serde_json::Value> {
    match message {
        Message::System(msg) => Ok(serde_json::json!({
            "role": "system",
            "content": msg.content,
        })),
        Message::Developer(msg) => Ok(serde_json::json!({
            "role": "developer",
            "content": msg.content,
        })),
        Message::User(msg) => Ok(serde_json::json!({
            "role": "user",
            "content": text_from_blocks(&msg.content),
        })),
        Message::Assistant(msg) => {
            let value = if msg.tool_calls.is_empty() {
                serde_json::json!({
                    "role": "assistant",
                    "content": text_from_blocks(&msg.content),
                })
            } else {
                serde_json::json!({
                    "role": "assistant",
                    "content": serde_json::Value::Null,
                    "tool_calls": msg.tool_calls.iter().map(tool_call_to_openai).collect::<Vec<_>>(),
                })
            };
            Ok(value)
        }
        Message::Tool(msg) => Ok(serde_json::json!({
            "role": "tool",
            "tool_call_id": msg.tool_call_id.0,
            "content": text_from_blocks(&msg.content),
        })),
    }
}

pub fn tools_to_openai(tools: &[ToolSpec]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })
        })
        .collect()
}

pub fn tool_call_to_openai(call: &ToolCall) -> serde_json::Value {
    serde_json::json!({
        "id": call.id.0,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
        }
    })
}

pub fn parse_openai_response(
    provider: &ProviderId,
    fallback_model: &str,
    response_text: &str,
) -> ModelResult<ChatResponse> {
    let json: serde_json::Value = serde_json::from_str(response_text)?;
    let choice = json["choices"]
        .as_array()
        .and_then(|a| a.first())
        .ok_or_else(|| ModelError::Serialization {
            detail: format!("no choices in response: {}", response_text),
        })?;

    let message = &choice["message"];
    let content = message["content"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| vec![ContentBlock::Text(s.to_string())])
        .unwrap_or_default();
    let tool_calls = message["tool_calls"]
        .as_array()
        .map(|calls| {
            calls
                .iter()
                .map(parse_tool_call)
                .collect::<ModelResult<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    let finish_reason = FinishReason::from_openai(choice["finish_reason"].as_str());

    Ok(ChatResponse {
        id: json["id"].as_str().map(|s| s.to_string()),
        model: json["model"].as_str().unwrap_or(fallback_model).to_string(),
        provider: provider.clone(),
        message: AssistantMessage {
            content,
            tool_calls,
            reasoning: None,
            provider_meta: ProviderMeta {
                raw_role: message["role"].as_str().map(|s| s.to_string()),
            },
        },
        finish_reason,
        usage: parse_usage(json.get("usage")),
        warnings: Vec::new(),
    })
}

fn next_sse_frame(buffer: &mut String) -> Option<String> {
    let (frame_end, delimiter_len) = ["\r\n\r\n", "\n\n", "\r\r"]
        .iter()
        .filter_map(|delimiter| buffer.find(delimiter).map(|index| (index, delimiter.len())))
        .min_by_key(|(index, _)| *index)?;
    let frame = buffer[..frame_end].to_string();
    buffer.drain(..frame_end + delimiter_len);
    Some(frame)
}

fn sse_data_payloads(frame: &str) -> Vec<&str> {
    frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect()
}

fn parse_openai_stream_chunk(
    provider: &ProviderId,
    fallback_model: &str,
    payload: &str,
    started: &mut bool,
) -> ModelResult<Vec<ChatStreamEvent>> {
    let json: serde_json::Value = serde_json::from_str(payload)?;
    let mut events = Vec::new();

    if !*started {
        events.push(ChatStreamEvent::ResponseStarted {
            id: json["id"].as_str().map(|s| s.to_string()),
            model: json["model"].as_str().unwrap_or(fallback_model).to_string(),
            provider: provider.clone(),
        });
        *started = true;
    }

    let choice = json["choices"]
        .as_array()
        .and_then(|choices| choices.first());
    if let Some(choice) = choice {
        let delta = &choice["delta"];
        if let Some(content) = delta["content"].as_str().filter(|s| !s.is_empty()) {
            events.push(ChatStreamEvent::TextDelta(content.to_string()));
        }

        if let Some(tool_calls) = delta["tool_calls"].as_array() {
            for tool_call in tool_calls {
                let index = tool_call["index"].as_u64().unwrap_or(0) as usize;
                let id = tool_call["id"]
                    .as_str()
                    .map(|id| ToolCallId(id.to_string()));
                let function = &tool_call["function"];
                let name = function["name"].as_str().map(|name| name.to_string());
                let arguments_delta = function["arguments"].as_str().map(|args| args.to_string());
                events.push(ChatStreamEvent::ToolCallDelta(ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments_delta,
                }));
            }
        }

        if !choice["finish_reason"].is_null() {
            events.push(ChatStreamEvent::Finished {
                finish_reason: FinishReason::from_openai(choice["finish_reason"].as_str()),
                usage: parse_usage(json.get("usage")),
            });
        }
    } else if let Some(usage) = parse_usage(json.get("usage")) {
        events.push(ChatStreamEvent::Usage(usage));
    }

    Ok(events)
}

fn parse_tool_call(value: &serde_json::Value) -> ModelResult<ToolCall> {
    let name = value["function"]["name"].as_str().unwrap_or("").to_string();
    let raw_args = value["function"]["arguments"].as_str().unwrap_or("{}");
    let arguments =
        serde_json::from_str(raw_args).map_err(|_| ModelError::InvalidToolArguments {
            tool_name: name.clone(),
            raw: raw_args.to_string(),
        })?;
    Ok(ToolCall {
        id: ToolCallId(value["id"].as_str().unwrap_or("unknown").to_string()),
        name,
        arguments,
    })
}

fn parse_usage(value: Option<&serde_json::Value>) -> Option<TokenUsage> {
    let usage = value?;
    Some(TokenUsage {
        input_tokens: usage["prompt_tokens"].as_u64().map(|v| v as u32),
        cached_input_tokens: usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .map(|v| v as u32),
        output_tokens: usage["completion_tokens"].as_u64().map(|v| v as u32),
        reasoning_output_tokens: usage["completion_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .map(|v| v as u32),
        total_tokens: usage["total_tokens"].as_u64().map(|v| v as u32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_messages_to_openai_json() {
        let messages = vec![
            Message::system("system"),
            Message::user_text("hello"),
            Message::assistant_text("world"),
            Message::tool_text("call_1", "result"),
        ];

        let json = messages_to_openai(&messages).unwrap();

        assert_eq!(json[0]["role"].as_str(), Some("system"));
        assert_eq!(json[1]["content"].as_str(), Some("hello"));
        assert_eq!(json[2]["role"].as_str(), Some("assistant"));
        assert_eq!(json[3]["tool_call_id"].as_str(), Some("call_1"));
    }

    #[test]
    fn converts_tool_spec_to_openai_json() {
        let tools = tools_to_openai(&[ToolSpec::new(
            "bash",
            "Run command",
            serde_json::json!({"type": "object"}),
        )]);

        assert_eq!(tools[0]["type"].as_str(), Some("function"));
        assert_eq!(tools[0]["function"]["name"].as_str(), Some("bash"));
    }

    #[test]
    fn parses_response_with_usage() {
        let response = parse_openai_response(
            &ProviderId("openai".to_string()),
            "fallback",
            r#"{
              "id": "chat_1",
              "model": "gpt-5.5",
              "choices": [{
                "finish_reason": "stop",
                "message": {
                  "role": "assistant",
                  "content": "hello"
                }
              }],
              "usage": {
                "prompt_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens": 6,
                "completion_tokens_details": {"reasoning_tokens": 2},
                "total_tokens": 16
              }
            }"#,
        )
        .unwrap();

        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert!(response.message.reasoning.is_none());
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.cached_input_tokens, Some(4));
        assert_eq!(usage.output_tokens, Some(6));
        assert_eq!(usage.reasoning_output_tokens, Some(2));
        assert_eq!(usage.total_tokens, Some(16));
    }

    #[test]
    fn parses_tool_calls_to_provider_neutral_shape() {
        let response = parse_openai_response(
            &ProviderId("openai".to_string()),
            "fallback",
            r#"{
              "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                  "role": "assistant",
                  "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
                  }]
                }
              }]
            }"#,
        )
        .unwrap();

        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        assert_eq!(response.message.tool_calls[0].name, "bash");
        assert_eq!(response.message.tool_calls[0].arguments["command"], "ls");
    }

    #[test]
    fn splits_sse_frames_with_crlf_delimiters() {
        let mut buffer = "data: {\"a\":1}\r\n\r\ndata: {\"b\":2}\r\n\r\npartial".to_string();

        assert_eq!(
            next_sse_frame(&mut buffer),
            Some("data: {\"a\":1}".to_string())
        );
        assert_eq!(
            next_sse_frame(&mut buffer),
            Some("data: {\"b\":2}".to_string())
        );
        assert_eq!(next_sse_frame(&mut buffer), None);
        assert_eq!(buffer, "partial");
    }

    #[test]
    fn waits_for_complete_sse_frame() {
        let mut buffer = "data: {\"a\":1}".to_string();

        assert_eq!(next_sse_frame(&mut buffer), None);

        buffer.push_str("\n\n");
        assert_eq!(
            next_sse_frame(&mut buffer),
            Some("data: {\"a\":1}".to_string())
        );
        assert!(buffer.is_empty());
    }

    #[test]
    fn parses_stream_text_delta_and_finish() {
        let mut started = false;
        let events = parse_openai_stream_chunk(
            &ProviderId("openai".to_string()),
            "fallback",
            r#"{
              "id": "chatcmpl_1",
              "model": "gpt-5.5",
              "choices": [{
                "delta": {"content": "hel"},
                "finish_reason": null
              }]
            }"#,
            &mut started,
        )
        .unwrap();

        assert_eq!(
            events[0],
            ChatStreamEvent::ResponseStarted {
                id: Some("chatcmpl_1".to_string()),
                model: "gpt-5.5".to_string(),
                provider: ProviderId("openai".to_string()),
            }
        );
        assert_eq!(events[1], ChatStreamEvent::TextDelta("hel".to_string()));

        let events = parse_openai_stream_chunk(
            &ProviderId("openai".to_string()),
            "fallback",
            r#"{
              "choices": [{
                "delta": {},
                "finish_reason": "stop"
              }]
            }"#,
            &mut started,
        )
        .unwrap();

        assert_eq!(
            events,
            vec![ChatStreamEvent::Finished {
                finish_reason: FinishReason::Stop,
                usage: None,
            }]
        );
    }

    #[test]
    fn parses_stream_tool_call_delta() {
        let mut started = true;
        let events = parse_openai_stream_chunk(
            &ProviderId("openai".to_string()),
            "fallback",
            r#"{
              "choices": [{
                "delta": {
                  "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "bash", "arguments": "{\"command\":"}
                  }]
                },
                "finish_reason": null
              }]
            }"#,
            &mut started,
        )
        .unwrap();

        assert_eq!(
            events,
            vec![ChatStreamEvent::ToolCallDelta(ToolCallDelta {
                index: 0,
                id: Some(ToolCallId("call_1".to_string())),
                name: Some("bash".to_string()),
                arguments_delta: Some("{\"command\":".to_string()),
            })]
        );
    }

    #[test]
    fn parses_stream_usage_without_overriding_finish_reason() {
        let mut started = true;
        let events = parse_openai_stream_chunk(
            &ProviderId("openai".to_string()),
            "fallback",
            r#"{
              "choices": [],
              "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3
              }
            }"#,
            &mut started,
        )
        .unwrap();

        assert!(matches!(events[0], ChatStreamEvent::Usage(_)));
    }
}
