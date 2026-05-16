use crate::message::AssistantMessage;
use crate::provider::ProviderId;
use crate::tool::ToolCallId;
use crate::usage::TokenUsage;
use crate::ModelResult;
use futures_util::Stream;
use std::pin::Pin;

pub type ChatStream = Pin<Box<dyn Stream<Item = ModelResult<ChatStreamEvent>> + Send>>;

#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    pub id: Option<String>,
    pub model: String,
    pub provider: ProviderId,
    pub message: AssistantMessage,
    pub finish_reason: FinishReason,
    pub usage: Option<TokenUsage>,
    pub warnings: Vec<ModelWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Error,
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelWarning {
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatStreamEvent {
    ResponseStarted {
        id: Option<String>,
        model: String,
        provider: ProviderId,
    },
    TextDelta(String),
    ToolCallDelta(ToolCallDelta),
    Finished {
        finish_reason: FinishReason,
        usage: Option<TokenUsage>,
    },
    Usage(TokenUsage),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<ToolCallId>,
    pub name: Option<String>,
    pub arguments_delta: Option<String>,
}

impl FinishReason {
    pub fn from_openai(value: Option<&str>) -> Self {
        match value.unwrap_or("stop") {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            "error" => FinishReason::Error,
            other => FinishReason::Unknown(other.to_string()),
        }
    }

    pub fn as_openai_str(&self) -> &str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Error => "error",
            FinishReason::Unknown(value) => value.as_str(),
        }
    }
}
