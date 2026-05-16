use crate::message::Message;
use crate::profile::ReasoningConfig;
use crate::tool::ToolSpec;

#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub tool_choice: ToolChoice,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub reasoning: ReasoningConfig,
    pub metadata: RequestMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequestMetadata {
    pub request_id: Option<String>,
}

impl ChatRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            max_output_tokens: None,
            temperature: None,
            reasoning: ReasoningConfig::Auto,
            metadata: RequestMetadata::default(),
        }
    }
}
