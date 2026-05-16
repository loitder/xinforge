use std::fmt;

pub type ModelResult<T> = Result<T, ModelError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    MissingApiKey,
    UnknownProvider { provider: String },
    UnknownProfile { profile: String },
    UnsupportedCapability { capability: String, model: String },
    ProviderHttp { status: u16, body: String },
    RateLimited { retry_after_ms: Option<u64> },
    Timeout,
    StreamParse { detail: String },
    InvalidToolArguments { tool_name: String, raw: String },
    Serialization { detail: String },
    Transport { detail: String },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::MissingApiKey => write!(f, "missing API key"),
            ModelError::UnknownProvider { provider } => {
                write!(f, "unknown model provider: {}", provider)
            }
            ModelError::UnknownProfile { profile } => {
                write!(f, "unknown model profile: {}", profile)
            }
            ModelError::UnsupportedCapability { capability, model } => {
                write!(f, "model {} does not support {}", model, capability)
            }
            ModelError::ProviderHttp { status, body } => {
                write!(f, "provider HTTP error {}: {}", status, body)
            }
            ModelError::RateLimited { retry_after_ms } => match retry_after_ms {
                Some(ms) => write!(f, "provider rate limited; retry after {}ms", ms),
                None => write!(f, "provider rate limited"),
            },
            ModelError::Timeout => write!(f, "provider request timed out"),
            ModelError::StreamParse { detail } => write!(f, "stream parse error: {}", detail),
            ModelError::InvalidToolArguments { tool_name, raw } => {
                write!(f, "invalid tool arguments for {}: {}", tool_name, raw)
            }
            ModelError::Serialization { detail } => write!(f, "serialization error: {}", detail),
            ModelError::Transport { detail } => write!(f, "transport error: {}", detail),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<reqwest::Error> for ModelError {
    fn from(value: reqwest::Error) -> Self {
        if value.is_timeout() {
            ModelError::Timeout
        } else {
            ModelError::Transport {
                detail: value.to_string(),
            }
        }
    }
}

impl From<serde_json::Error> for ModelError {
    fn from(value: serde_json::Error) -> Self {
        ModelError::Serialization {
            detail: value.to_string(),
        }
    }
}
