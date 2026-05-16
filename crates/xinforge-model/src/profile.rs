use crate::provider::ProviderId;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelProfile {
    pub name: String,
    pub provider: ProviderId,
    pub model_id: String,
    pub base_url: Option<String>,
    pub api_key: String,
    pub defaults: GenerationDefaults,
    pub capabilities: ModelCapabilities,
    pub fallback: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileRef(pub String);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationDefaults {
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub reasoning: ReasoningConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub streaming: bool,
    pub tools: bool,
    pub parallel_tool_calls: bool,
    pub reasoning: ReasoningSupport,
    pub vision: bool,
    pub max_context_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningSupport {
    None,
    Native,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningConfig {
    Disabled,
    Enabled,
    Auto,
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        Self {
            max_output_tokens: None,
            temperature: None,
            reasoning: ReasoningConfig::Auto,
        }
    }
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            streaming: false,
            tools: true,
            parallel_tool_calls: true,
            reasoning: ReasoningSupport::None,
            vision: false,
            max_context_tokens: None,
            max_output_tokens: None,
        }
    }
}
