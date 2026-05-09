from typing import Literal, TypedDict, Unpack

from langchain.chat_models import init_chat_model

OPENAI_RESPONSES_WS_BASE_URL = "wss://api.openai.com/v1"


OpenAIReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]


class OpenAIReasoning(TypedDict, total=False):
    effort: OpenAIReasoningEffort


class ModelKwargs(TypedDict, total=False):
    max_tokens: int | None
    reasoning: OpenAIReasoning | None
    temperature: float | None


def make_model(model_id: str, **kwargs: Unpack[ModelKwargs]):
    model_kwargs: dict[str, object] = kwargs.copy()

    if model_id.startswith("nvidia:"):
        import os
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        
        nvidia_model = model_id.replace("nvidia:", "")
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        
        # Configure max_tokens, temperature according to project
        temp = model_kwargs.get("temperature", 0.2)
        max_tokens = model_kwargs.get("max_tokens", 2048)
        
        return ChatNVIDIA(
            model=nvidia_model,
            api_key=api_key,
            temperature=temp,
            max_tokens=max_tokens
        )

    if model_id.startswith("openai:"):
        model_kwargs["base_url"] = OPENAI_RESPONSES_WS_BASE_URL
        model_kwargs["use_responses_api"] = True

    return init_chat_model(model=model_id, **model_kwargs)
