use std::future::Future;
use std::pin::Pin;

use crate::profile::{ModelCapabilities, ModelProfile};
use crate::request::ChatRequest;
use crate::response::{ChatResponse, ChatStream, ChatStreamEvent};
use crate::ModelResult;
use futures_util::stream;

pub mod openai;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProviderId(pub String);

pub trait ModelProvider: Send + Sync {
    fn id(&self) -> ProviderId;

    fn capabilities(&self, model_id: &str) -> ModelCapabilities;

    fn chat<'a>(
        &'a self,
        profile: &'a ModelProfile,
        request: ChatRequest,
    ) -> Pin<Box<dyn Future<Output = ModelResult<ChatResponse>> + Send + 'a>>;

    fn chat_stream<'a>(
        &'a self,
        profile: &'a ModelProfile,
        request: ChatRequest,
    ) -> Pin<Box<dyn Future<Output = ModelResult<ChatStream>> + Send + 'a>> {
        Box::pin(async move {
            let response = self.chat(profile, request).await?;
            let mut events = Vec::new();
            events.push(Ok(ChatStreamEvent::ResponseStarted {
                id: response.id.clone(),
                model: response.model.clone(),
                provider: response.provider.clone(),
            }));
            for block in &response.message.content {
                let crate::message::ContentBlock::Text(text) = block;
                events.push(Ok(ChatStreamEvent::TextDelta(text.clone())));
            }
            events.push(Ok(ChatStreamEvent::Finished {
                finish_reason: response.finish_reason,
                usage: response.usage,
            }));
            Ok(Box::pin(stream::iter(events)) as ChatStream)
        })
    }
}
