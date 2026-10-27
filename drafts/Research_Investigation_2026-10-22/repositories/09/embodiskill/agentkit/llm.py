import os
import time
import math
import re

from typing import (
    Protocol,
    Literal,
    Optional,
    List,
    Any,
)
from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .utils import load_config
from .azure import azure_openai_model


# model configs
CONFIG: dict = load_config("configs/configs.yaml")
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)  
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)

if os.environ.get("EMBODISKILL_MAX_TOKEN", "").strip():
    print(f"# EMBODISKILL_MAX_TOKEN override applied: {MAX_TOKEN}")

# Only read OpenAI env vars if they exist (for Azure compatibility)
URL = os.environ.get("OPENAI_BASE_URL", "")
KEY = os.environ.get("OPENAI_API_KEY", "")
if URL or KEY:
    print('# api url: ', URL)
    print('# api key: ', KEY)


completion_tokens, prompt_tokens = 0, 0

REQUEST_TIMEOUT = float(os.environ.get("OPENAI_TIMEOUT", "120"))
MAX_RETRIES = max(1, int(os.environ.get("OPENAI_MAX_RETRIES", "20")))
INITIAL_RETRY_WAIT = float(os.environ.get("OPENAI_RETRY_WAIT", "5"))
MAX_RETRY_WAIT = float(os.environ.get("OPENAI_MAX_RETRY_WAIT", "60"))


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _env_optional_int(name: str) -> Optional[int]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    return int(value)


def _build_gemini_thinking_config(genai_types: Any, model_name: str) -> Optional[Any]:
    if not _env_flag("EMBODISKILL_GEMINI_DISABLE_THINKING", False):
        return None

    thinking_level = os.environ.get("EMBODISKILL_GEMINI_THINKING_LEVEL", "").strip()
    thinking_budget = _env_optional_int("EMBODISKILL_GEMINI_THINKING_BUDGET")
    thinking_kwargs: dict[str, Any] = {}

    if thinking_budget is not None:
        thinking_kwargs["thinking_budget"] = thinking_budget
    elif thinking_level:
        thinking_kwargs["thinking_level"] = thinking_level
    else:
        model_name_lower = model_name.lower()
        if "gemini-3" in model_name_lower:
            # Gemini 3 Flash does not expose a full "off" switch, so use the lowest reasoning level.
            thinking_kwargs["thinking_level"] = "minimal"
        elif "gemini-2.5-flash" in model_name_lower:
            thinking_kwargs["thinking_budget"] = 0
        else:
            print(
                f"# Gemini thinking disable requested for {model_name}, "
                "but no safe default is configured for this model. Using provider default."
            )
            return None

    print(f"# Gemini thinking config for {model_name}: {thinking_kwargs}")
    return genai_types.ThinkingConfig(**thinking_kwargs)


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class LLM(ABC):
    
    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass


class LLMRequestError(RuntimeError):
    """Raised when an LLM request still fails after exhausting retries."""

    pass


def _format_error_details(error: Exception) -> str:
    details = [f"{type(error).__name__}: {error}"]
    cause = getattr(error, "__cause__", None)
    if cause is not None:
        details.append(f"cause={type(cause).__name__}: {cause}")
    context = getattr(error, "__context__", None)
    if context is not None and context is not cause:
        details.append(f"context={type(context).__name__}: {context}")
    return " | ".join(details)


def _estimate_text_tokens(text: Any) -> int:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return 0

    ascii_chars = sum(1 for ch in normalized if ord(ch) < 128)
    non_ascii_chars = len(normalized) - ascii_chars
    word_count = len(normalized.split(" "))

    # Conservative heuristic that works reasonably for English-heavy prompts with JSON.
    char_estimate = math.ceil(ascii_chars / 4) + non_ascii_chars
    word_estimate = math.ceil(word_count * 1.35)
    return max(char_estimate, word_estimate)


def _message_role_and_content(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        role = str(message.get("role", "") or "")
        content = message.get("content", "")
    else:
        role = str(getattr(message, "role", "") or "")
        content = getattr(message, "content", "")

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None) or ""
            if text:
                parts.append(str(text))
        content = "".join(parts)

    return role, str(content or "")


def _estimate_messages_tokens(messages: list[Any]) -> int:
    total = 3
    for message in messages:
        role, content = _message_role_and_content(message)
        total += 4 + _estimate_text_tokens(role) + _estimate_text_tokens(content)
    return total


def _enforce_input_token_limit(
    messages: list[Any],
    *,
    model_name: str,
    max_tokens: int,
) -> None:
    max_input_tokens = _env_optional_int("EMBODISKILL_MAX_INPUT_TOKENS")
    if max_input_tokens is None or max_input_tokens <= 0:
        return

    estimated_input_tokens = _estimate_messages_tokens(messages)
    if estimated_input_tokens <= max_input_tokens:
        return

    raise LLMRequestError(
        f"Estimated input tokens {estimated_input_tokens} exceed "
        f"EMBODISKILL_MAX_INPUT_TOKENS={max_input_tokens} for model {model_name} "
        f"(requested max_tokens={max_tokens}). Refusing request before send."
    )


def _is_retryable_error(error: Exception) -> bool:
    retryable_types = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "RateLimitError",
        "InternalServerError",
    }
    if type(error).__name__ in retryable_types:
        return True

    error_message = str(error).lower()
    retryable_keywords = [
        "rate limit",
        "429",
        "timeout",
        "timed out",
        "connection error",
        "connection refused",
        "connection reset",
        "server disconnected",
        "temporarily unavailable",
        "502",
        "503",
        "504",
    ]
    return any(keyword in error_message for keyword in retryable_keywords)


def _is_fatal_quota_error(error: Exception) -> bool:
    """Stop the run when the paid GPT endpoint cannot serve more requests."""
    error_details = _format_error_details(error).lower()
    error_type = type(error).__name__.lower()

    fatal_quota_keywords = [
        "insufficient_quota",
        "local:insufficient_quota",
        "user quota is not enough",
        "exceeded your current quota",
        "check your plan and billing",
    ]
    if any(keyword in error_details for keyword in fatal_quota_keywords):
        return True

    if error_type == "permissiondeniederror" and (
        "403" in error_details or "forbidden" in error_details
    ):
        return True

    return (
        "permissiondeniederror" in error_details
        and ("403" in error_details or "forbidden" in error_details)
    )


def _extract_text_and_usage(response: Any) -> tuple[str, int, int]:
    if isinstance(response, str):
        return response, 0, 0

    if isinstance(response, dict):
        usage = response.get("usage", {}) or {}
        choices = response.get("choices", []) or []
        if not choices:
            raise ValueError("LLM response dict did not contain choices.")
        content = choices[0].get("message", {}).get("content")
        return content or "", usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0

    usage = getattr(response, "usage", None)
    prompt_token_count = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_token_count = getattr(usage, "completion_tokens", 0) if usage else 0

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("LLM response choice did not contain a message.")
        content = getattr(message, "content", None)
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)
                if text:
                    text_parts.append(text)
            content = "".join(text_parts)
        return content or "", prompt_token_count or 0, completion_token_count or 0

    raise TypeError(
        f"Unsupported LLM response type: {type(response).__name__}. "
        f"Response preview: {repr(response)[:300]}"
    )


def _create_chat_completion(
    client: Any,
    model_name: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    stop_strs: Optional[List[str]],
    num_comps: int,
    extra_body: Optional[dict[str, Any]] = None,
):
    if "gpt-5" in model_name.lower():
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            n=num_comps,
            extra_body=extra_body,
        )

    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=num_comps,
        stop=stop_strs,
        extra_body=extra_body,
    )


def _call_with_retries(
    client: Any,
    model_name: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    stop_strs: Optional[List[str]],
    num_comps: int,
    provider_name: str = "API",
    extra_body: Optional[dict[str, Any]] = None,
) -> tuple[str, int, int]:
    wait_time = INITIAL_RETRY_WAIT
    last_error: Optional[Exception] = None

    _enforce_input_token_limit(
        messages,
        model_name=model_name,
        max_tokens=max_tokens,
    )

    if "qwen3" in model_name.lower():
        extra_body = {}
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _create_chat_completion(
                client=client,
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_strs=stop_strs,
                num_comps=num_comps,
                extra_body=extra_body,
            )
            answer, prompt_token_count, completion_token_count = _extract_text_and_usage(response)

            if answer is None or (isinstance(answer, str) and not answer.strip()):
                raise ValueError("LLM returned empty content.")

            return answer, prompt_token_count, completion_token_count

        except Exception as error:
            last_error = error
            error_details = _format_error_details(error)

            if _is_fatal_quota_error(error):
                raise LLMRequestError(
                    f"Fatal {provider_name} quota/permission error for model {model_name}: {error_details}"
                ) from error

            is_retryable = _is_retryable_error(error)
            if isinstance(error, ValueError) and "empty content" in str(error).lower():
                is_retryable = True
            has_attempts_left = attempt < MAX_RETRIES

            if is_retryable and has_attempts_left:
                print(
                    f"Retrying {provider_name} call "
                    f"(attempt {attempt}/{MAX_RETRIES}) after error: {error_details}. "
                    f"Sleeping {wait_time:.1f}s."
                )
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, MAX_RETRY_WAIT)
                continue

            print(f"Error during {provider_name} call: {error_details}")
            break

    if last_error is None:
        raise LLMRequestError(
            f"{provider_name} call failed for model {model_name} after {MAX_RETRIES} attempts: unknown error."
        )

    raise LLMRequestError(
        f"{provider_name} call failed for model {model_name} after {MAX_RETRIES} attempts: "
        f"{_format_error_details(last_error)}"
    ) from last_error

class GPTChat(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.client = OpenAI(
            base_url=URL,
            api_key=KEY,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        global prompt_tokens, completion_tokens

        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Ensure max_tokens is never None
        if max_tokens is None:
            max_tokens = MAX_TOKEN

        answer, prompt_token_count, completion_token_count = _call_with_retries(
            client=self.client,
            model_name=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_strs=stop_strs,
            num_comps=num_comps,
            provider_name="API",
        )
        prompt_tokens += prompt_token_count
        completion_tokens += completion_token_count
        return answer


class LocalLLM(LLM):
    """Local LLM via vLLM OpenAI-compatible API"""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
    ):
        super().__init__(model_name=model_name)
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",  # vLLM doesn't require a real key
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        global prompt_tokens, completion_tokens

        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Ensure max_tokens is never None
        if max_tokens is None:
            max_tokens = MAX_TOKEN

        answer, prompt_token_count, completion_token_count = _call_with_retries(
            client=self.client,
            model_name=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_strs=stop_strs,
            num_comps=num_comps,
            provider_name="API",
        )
        prompt_tokens += prompt_token_count
        completion_tokens += completion_token_count
        return answer


class AzureGPTChat(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.client = azure_openai_model(model_name)

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        global prompt_tokens, completion_tokens

        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Ensure max_tokens is never None
        if max_tokens is None:
            max_tokens = MAX_TOKEN

        answer, prompt_token_count, completion_token_count = _call_with_retries(
            client=self.client,
            model_name=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_strs=stop_strs,
            num_comps=num_comps,
            provider_name="API",
        )
        prompt_tokens += prompt_token_count
        completion_tokens += completion_token_count
        return answer


class GeminiChat(LLM):
    """Gemini LLM via google-genai SDK."""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        api_key = os.environ.get("GEMINI_API_KEY", "")
        base_url = os.environ.get("GOOGLE_GEMINI_BASE_URL", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as error:
            raise ImportError(
                "Gemini support requires the `google-genai` package. "
                "Install it with `pip install google-genai`."
            ) from error

        self._genai_types = genai_types
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["http_options"] = {"base_url": base_url}
            print(f"# Gemini base_url: {base_url}")
        self.client = genai.Client(**client_kwargs)

        print(f"# Gemini model: {model_name}")
        print("# Gemini sdk: google-genai")

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        global prompt_tokens, completion_tokens

        system_parts = []
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "user":
                contents.append(self._genai_types.Content(
                    role="user", parts=[self._genai_types.Part(text=msg.content)]
                ))
            elif msg.role == "assistant":
                contents.append(self._genai_types.Content(
                    role="model", parts=[self._genai_types.Part(text=msg.content)]
                ))

        if max_tokens is None:
            max_tokens = MAX_TOKEN

        _enforce_input_token_limit(
            messages,
            model_name=self.model_name,
            max_tokens=max_tokens,
        )

        wait_time = INITIAL_RETRY_WAIT
        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                config_kwargs = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                thinking_config = _build_gemini_thinking_config(
                    genai_types=self._genai_types,
                    model_name=self.model_name,
                )
                if thinking_config is not None:
                    config_kwargs["thinking_config"] = thinking_config
                if system_parts:
                    config_kwargs["system_instruction"] = "\n".join(system_parts)
                if stop_strs:
                    config_kwargs["stop_sequences"] = stop_strs

                config = self._genai_types.GenerateContentConfig(**config_kwargs)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                answer = response.text
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    prompt_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    completion_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

                if answer is None or not str(answer).strip():
                    raise ValueError("Gemini returned empty content.")
                # import pdb; pdb.set_trace()
                return answer

            except Exception as error:
                last_error = error
                error_details = _format_error_details(error)
                is_retryable = _is_retryable_error(error)
                if isinstance(error, ValueError) and "empty content" in str(error).lower():
                    is_retryable = True

                if is_retryable and attempt < MAX_RETRIES:
                    print(
                        f"Retrying Gemini call "
                        f"(attempt {attempt}/{MAX_RETRIES}) after error: {error_details}. "
                        f"Sleeping {wait_time:.1f}s."
                    )
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, MAX_RETRY_WAIT)
                    continue

                print(f"Error during Gemini API call: {error_details}")
                break

        if last_error is None:
            raise LLMRequestError(
                f"Gemini call failed for model {self.model_name} after {MAX_RETRIES} attempts: unknown error."
            )

        raise LLMRequestError(
            f"Gemini call failed for model {self.model_name} after {MAX_RETRIES} attempts: "
            f"{_format_error_details(last_error)}"
        ) from last_error


def get_price():
    global completion_tokens, prompt_tokens
    return completion_tokens, prompt_tokens, completion_tokens*60/1000000+prompt_tokens*30/1000000
