"""
LLM Provider 추상화 — Claude / OpenAI / Ollama 교체 가능
"""
from abc import ABC, abstractmethod
from typing import Any
import asyncio
import time
import json
from loguru import logger
from ..core.config import settings


class LLMMessage:
    def __init__(self, role: str, content: str | list):
        self.role = role
        self.content = content


class LLMResponse:
    def __init__(self, content: str, tool_calls: list[dict] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []

    @property
    def has_tool_call(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        system: str,
        tools: list[dict] | None = None,
        image_data: bytes | None = None,
        image_media_type: str = "image/jpeg",
    ) -> LLMResponse:
        pass


class ClaudeProvider(BaseLLMProvider):
    def __init__(self):
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.LLM_MODEL

    async def complete(
        self,
        messages: list[LLMMessage],
        system: str,
        tools: list[dict] | None = None,
        image_data: bytes | None = None,
        image_media_type: str = "image/jpeg",
    ) -> LLMResponse:
        import anthropic
        import base64

        formatted = []
        for i, msg in enumerate(messages):
            # 첫 번째 유저 메시지에 이미지 첨부
            if msg.role == "user" and i == 0 and image_data:
                b64 = base64.standard_b64encode(image_data).decode("utf-8")
                formatted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": msg.content if isinstance(msg.content, str) else str(msg.content)},
                    ],
                })
            else:
                formatted.append({"role": msg.role, "content": msg.content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": formatted,
        }

        if tools:
            claude_tools = []
            for t in tools:
                claude_tools.append({
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                })
            kwargs["tools"] = claude_tools

        response = await self.client.messages.create(**kwargs)

        content_text = ""
        tool_calls = []

        for block in response.content:
            if hasattr(block, "text"):
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return LLMResponse(content=content_text, tool_calls=tool_calls)


class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.LLM_MODEL or "gpt-4o"

    async def complete(
        self,
        messages: list[LLMMessage],
        system: str,
        tools: list[dict] | None = None,
        image_data: bytes | None = None,
        image_media_type: str = "image/jpeg",
    ) -> LLMResponse:
        import base64

        formatted = [{"role": "system", "content": system}]
        for i, msg in enumerate(messages):
            if msg.role == "user" and i == 0 and image_data:
                b64 = base64.standard_b64encode(image_data).decode("utf-8")
                formatted.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg.content if isinstance(msg.content, str) else ""},
                        {"type": "image_url", "image_url": {"url": f"data:{image_media_type};base64,{b64}"}},
                    ],
                })
            else:
                formatted.append({"role": msg.role, "content": msg.content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted,
            "max_tokens": 4096,
        }

        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

        return LLMResponse(content=msg.content or "", tool_calls=tool_calls)


class OllamaProvider(BaseLLMProvider):
    def __init__(self):
        import ollama
        self.client = ollama.AsyncClient(host=settings.OLLAMA_BASE_URL)
        self.model = settings.OLLAMA_MODEL

    async def complete(
        self,
        messages: list[LLMMessage],
        system: str,
        tools: list[dict] | None = None,
        image_data: bytes | None = None,
        image_media_type: str = "image/jpeg",
    ) -> LLMResponse:
        import base64

        # qwen2.5vl 등 vision 모델은 tool API 미지원 → 도구 목록을 시스템 메시지에 텍스트로 주입
        effective_system = system
        if tools:
            tool_lines = []
            for t in tools:
                params = list(t.get("parameters", {}).get("properties", {}).keys())
                param_str = ", ".join(params) if params else ""
                tool_lines.append(f"- {t['name']}({param_str}): {t['description'][:80]}")
            tool_block = "\n".join(tool_lines)
            effective_system = (
                system
                + f"\n\n[사용 가능한 도구 목록]\n{tool_block}\n"
                + "도구 호출: ACTION: 도구명({\"key\": \"value\"}) 형식으로 정확히 작성하세요."
            )

        formatted = [{"role": "system", "content": effective_system}]
        for i, msg in enumerate(messages):
            m: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if i == 0 and image_data:
                m["images"] = [base64.standard_b64encode(image_data).decode("utf-8")]
            formatted.append(m)

        # tools 파라미터는 전달하지 않음 (vision 모델 400 오류 방지)
        response = await self.client.chat(model=self.model, messages=formatted)

        # ollama 라이브러리 버전에 따라 dict 또는 객체 반환
        if hasattr(response, "message"):
            msg = response.message
            content = msg.content if hasattr(msg, "content") else (msg.get("content", "") if isinstance(msg, dict) else "")
            raw_tool_calls = msg.tool_calls if hasattr(msg, "tool_calls") else (msg.get("tool_calls") if isinstance(msg, dict) else None)
        else:
            msg = response.get("message", {})
            content = msg.get("content", "")
            raw_tool_calls = msg.get("tool_calls")

        tool_calls = []
        if raw_tool_calls:
            for tc in raw_tool_calls:
                if hasattr(tc, "function"):
                    fn = tc.function
                    tool_calls.append({
                        "id": str(id(tc)),
                        "name": fn.name if hasattr(fn, "name") else fn["name"],
                        "arguments": fn.arguments if hasattr(fn, "arguments") else fn["arguments"],
                    })
                elif isinstance(tc, dict) and "function" in tc:
                    tool_calls.append({
                        "id": str(id(tc)),
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    })

        return LLMResponse(content=content or "", tool_calls=tool_calls)


class GroqProvider(BaseLLMProvider):
    # 클래스 공유 레이트리미터 — Groq 무료 티어 30 RPM 기준 20 RPM로 선제 제한
    _lock: asyncio.Lock | None = None
    _last_call: float = 0.0
    _MIN_INTERVAL: float = 2.0   # 초 (= ~30 RPM 한계에 근접, 8스텝 × 2s = 16s)

    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            max_retries=3,
            timeout=60.0,
        )
        self.model = settings.GROQ_MODEL or "llama-3.1-8b-instant"

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    async def _rate_limit(self) -> None:
        """Groq API 호출 전 선제 레이트리밋 — 429 발생 최소화"""
        async with self._get_lock():
            now = time.monotonic()
            wait = self._MIN_INTERVAL - (now - self.__class__._last_call)
            if wait > 0:
                logger.debug(f"[Groq RateLimit] {wait:.1f}s 대기")
                await asyncio.sleep(wait)
            self.__class__._last_call = time.monotonic()

    async def complete(
        self,
        messages: list[LLMMessage],
        system: str,
        tools: list[dict] | None = None,
        image_data: bytes | None = None,
        image_media_type: str = "image/jpeg",
    ) -> LLMResponse:
        # Groq: tool descriptions injected as text (avoids tool_use_failed errors
        # when model outputs free text like THINK/HYPOTHESIS/CONCLUDE format).
        effective_system = system
        if tools:
            tool_lines = []
            for t in tools:
                params = list(t.get("parameters", {}).get("properties", {}).keys())
                param_str = ", ".join(params) if params else ""
                tool_lines.append(f"- {t['name']}({param_str}): {t['description'][:100]}")
            tool_block = "\n".join(tool_lines)
            effective_system = (
                system
                + f"\n\n[사용 가능한 도구 목록]\n{tool_block}\n"
                + "도구 호출: ACTION: 도구명({\"key\": \"value\"}) 형식으로 정확히 작성하세요.\n"
                + "결론 시: ACTION: CONCLUDE"
            )

        formatted = [{"role": "system", "content": effective_system}]
        for i, msg in enumerate(messages):
            content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
            # 첫 유저 메시지에 이미지 첨부 (Groq: 이미지는 반드시 첫 번째 user 메시지에)
            if msg.role == "user" and i == 0 and image_data:
                import base64
                b64 = base64.standard_b64encode(image_data).decode("utf-8")
                formatted.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content_str},
                        {"type": "image_url", "image_url": {"url": f"data:{image_media_type};base64,{b64}"}},
                    ],
                })
            else:
                formatted.append({"role": msg.role, "content": content_str})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted,
            "max_tokens": 1500,
            "temperature": 0.1,
        }

        await self._rate_limit()
        # Rate limit 폴백: scout → 70b → 8b 자동 전환
        FALLBACK_MODELS = [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]
        last_exc = None
        for attempt_model in ([kwargs["model"]] + [m for m in FALLBACK_MODELS if m != kwargs["model"]]):
            try:
                kwargs["model"] = attempt_model
                response = await self.client.chat.completions.create(**kwargs)
                if attempt_model != self.model:
                    logger.info(f"[Groq] Using fallback model: {attempt_model}")
                msg = response.choices[0].message
                return LLMResponse(content=msg.content or "", tool_calls=[])
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower() or "token" in err.lower():
                    logger.warning(f"[Groq] Rate limit on {attempt_model}, trying next fallback")
                    last_exc = e
                    await asyncio.sleep(2)
                    continue
                raise
        raise last_exc


def get_llm_provider() -> BaseLLMProvider:
    provider = settings.LLM_PROVIDER
    if provider == "claude":
        return ClaudeProvider()
    elif provider == "openai":
        return OpenAIProvider()
    elif provider == "ollama":
        return OllamaProvider()
    elif provider == "groq":
        return GroqProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

