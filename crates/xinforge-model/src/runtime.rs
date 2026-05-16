use std::collections::HashMap;

use crate::profile::{ModelCapabilities, ModelProfile, ProfileRef};
use crate::provider::{openai::OpenAiProvider, ModelProvider, ProviderId};
use crate::request::ChatRequest;
use crate::response::{ChatResponse, ChatStream};
use crate::{ModelError, ModelResult};

pub struct ModelRuntime {
    providers: HashMap<ProviderId, Box<dyn ModelProvider>>,
    profiles: HashMap<String, ModelProfile>,
}

impl ModelRuntime {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            profiles: HashMap::new(),
        }
    }

    pub fn with_default_providers() -> Self {
        let mut runtime = Self::new();
        runtime.register_provider(Box::new(OpenAiProvider::new()));
        runtime
    }

    pub fn from_openai_config(base_url: &str, api_key: &str, model_id: &str) -> Self {
        let mut runtime = Self::with_default_providers();
        let provider = ProviderId("openai".to_string());
        let capabilities = ModelCapabilities {
            streaming: true,
            tools: true,
            parallel_tool_calls: true,
            reasoning: crate::profile::ReasoningSupport::None,
            vision: false,
            max_context_tokens: None,
            max_output_tokens: None,
        };
        runtime.register_profile(ModelProfile {
            name: "default".to_string(),
            provider,
            model_id: model_id.to_string(),
            base_url: Some(base_url.to_string()),
            api_key: api_key.to_string(),
            defaults: crate::profile::GenerationDefaults {
                max_output_tokens: Some(8000),
                temperature: None,
                reasoning: crate::profile::ReasoningConfig::Auto,
            },
            capabilities,
            fallback: Vec::new(),
        });
        runtime
    }

    pub fn register_provider(&mut self, provider: Box<dyn ModelProvider>) {
        self.providers.insert(provider.id(), provider);
    }

    pub fn register_profile(&mut self, profile: ModelProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }

    pub async fn chat(
        &self,
        profile_ref: &ProfileRef,
        mut request: ChatRequest,
    ) -> ModelResult<ChatResponse> {
        let profile =
            self.profiles
                .get(&profile_ref.0)
                .ok_or_else(|| ModelError::UnknownProfile {
                    profile: profile_ref.0.clone(),
                })?;

        if !request.tools.is_empty() && !profile.capabilities.tools {
            return Err(ModelError::UnsupportedCapability {
                capability: "tools".to_string(),
                model: profile.model_id.clone(),
            });
        }

        if request.max_output_tokens.is_none() {
            request.max_output_tokens = profile.defaults.max_output_tokens;
        }
        if request.temperature.is_none() {
            request.temperature = profile.defaults.temperature;
        }
        if request.reasoning == crate::profile::ReasoningConfig::Auto {
            request.reasoning = profile.defaults.reasoning;
        }

        let provider =
            self.providers
                .get(&profile.provider)
                .ok_or_else(|| ModelError::UnknownProvider {
                    provider: profile.provider.0.clone(),
                })?;

        provider.chat(profile, request).await
    }

    pub async fn chat_stream(
        &self,
        profile_ref: &ProfileRef,
        mut request: ChatRequest,
    ) -> ModelResult<ChatStream> {
        let profile =
            self.profiles
                .get(&profile_ref.0)
                .ok_or_else(|| ModelError::UnknownProfile {
                    profile: profile_ref.0.clone(),
                })?;

        if !profile.capabilities.streaming {
            return Err(ModelError::UnsupportedCapability {
                capability: "streaming".to_string(),
                model: profile.model_id.clone(),
            });
        }
        if !request.tools.is_empty() && !profile.capabilities.tools {
            return Err(ModelError::UnsupportedCapability {
                capability: "tools".to_string(),
                model: profile.model_id.clone(),
            });
        }

        if request.max_output_tokens.is_none() {
            request.max_output_tokens = profile.defaults.max_output_tokens;
        }
        if request.temperature.is_none() {
            request.temperature = profile.defaults.temperature;
        }
        if request.reasoning == crate::profile::ReasoningConfig::Auto {
            request.reasoning = profile.defaults.reasoning;
        }

        let provider =
            self.providers
                .get(&profile.provider)
                .ok_or_else(|| ModelError::UnknownProvider {
                    provider: profile.provider.0.clone(),
                })?;

        provider.chat_stream(profile, request).await
    }
}

impl Default for ModelRuntime {
    fn default() -> Self {
        Self::with_default_providers()
    }
}
