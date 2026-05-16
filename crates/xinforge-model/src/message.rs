use crate::tool::{ToolCall, ToolCallId};

#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    System(SystemMessage),
    Developer(DeveloperMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolMessage),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemMessage {
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeveloperMessage {
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserMessage {
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
    pub tool_calls: Vec<ToolCall>,
    pub reasoning: Option<ReasoningTrace>,
    pub provider_meta: ProviderMeta,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolMessage {
    pub tool_call_id: ToolCallId,
    pub content: Vec<ContentBlock>,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBlock {
    Text(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningTrace {
    pub text: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderMeta {
    pub raw_role: Option<String>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self::System(SystemMessage {
            content: text.into(),
        })
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::User(UserMessage {
            content: vec![ContentBlock::Text(text.into())],
        })
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant(AssistantMessage {
            content: vec![ContentBlock::Text(text.into())],
            tool_calls: Vec::new(),
            reasoning: None,
            provider_meta: ProviderMeta::default(),
        })
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self::Assistant(AssistantMessage {
            content: Vec::new(),
            tool_calls,
            reasoning: None,
            provider_meta: ProviderMeta::default(),
        })
    }

    pub fn tool_text(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool(ToolMessage {
            tool_call_id: ToolCallId(tool_call_id.into()),
            content: vec![ContentBlock::Text(content.into())],
            is_error: false,
        })
    }
}

pub fn text_from_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) => text.as_str(),
        })
        .collect::<Vec<_>>()
        .join("")
}
