import base64
from typing import Awaitable, Callable, TypedDict
from io import BytesIO
import time
from openai import AsyncOpenAI as AsyncClient
from openai.types import CompletionUsage
from PIL.Image import Image


class ChatTextDetails(TypedDict):
    prompt: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    content: str
    total_time: float
    tps: float
    ttft: float


class ImageDetails(TypedDict):
    image_format: str
    image_size: tuple[int, int]
    image_bytes: int


class ChatImageDetails(ChatTextDetails, ImageDetails):
    pass


class ChatImageListDetails(ChatTextDetails):
    image_list: list[ImageDetails]


def image_to_base64(image: Image, default_fmt="JPEG") -> dict:
    buffer = BytesIO()
    fmt = image.format or default_fmt
    image.save(buffer, format=fmt)
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")

    return {
        "base64": f"data:image/{fmt.lower()};base64,{base64_str}",
        "size": image.size,  # (width, height)
        "bytes": len(img_bytes),  # 字节数
        "format": fmt,
    }


class TokenInfo(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLM:
    client: AsyncClient
    model: str
    temperature: float
    token_dict: dict[str, TokenInfo]

    def __init__(
        self,
        api_key: str,
        base_url: str,
        temperature: float,
        model: str,
    ) -> None:
        self.temperature = temperature
        self.client = AsyncClient(api_key=api_key, base_url=base_url)
        self.model = model
        self.token_dict = {}

    def update_token_dict(self, model: str, usage: CompletionUsage):
        if model not in self.token_dict:
            self.token_dict[model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        self.token_dict[model]["completion_tokens"] += usage.completion_tokens
        self.token_dict[model]["prompt_tokens"] += usage.prompt_tokens
        self.token_dict[model]["total_tokens"] += usage.total_tokens

    async def chat_with_text_detail(
        self, prompt: str, hook: Callable[[str], Awaitable[None]] | None = None, **kwargs
    ) -> ChatTextDetails:
        params = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "include_obfuscation": False,
            },
            **kwargs,
        }
        if self.model.startswith("qwen"):
            params["extra_body"] = {"enable_thinking": False}

        start_time = time.time()
        first_token_time = None
        full_content = ""
        usage = None

        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices == [] and chunk.usage is not None:
                    usage = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_content += delta.content
                    if hook is not None:
                        await hook(full_content)

            assert first_token_time is not None
            assert usage is not None
        except Exception as e:
            raise Exception(f"LLM async request failed: {e}")

        end_time = time.time()

        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        self.update_token_dict(self.model, usage)

        return {
            "prompt": prompt,
            "model": self.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "content": full_content,
            "total_time": end_time - start_time,
            "ttft": first_token_time - start_time,
            "tps": completion_tokens / (end_time - first_token_time),
        }

    async def chat_with_image_detail(
        self,
        prompt: str,
        image: Image,
        default_fmt="JPEG",
        hook: Callable[[str], Awaitable[None]] | None = None,
        **kwargs,
    ) -> ChatImageDetails:
        base64_res = image_to_base64(image, default_fmt)

        params = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": base64_res["base64"]}},
                    ],
                }
            ],
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "include_obfuscation": False,
            },
            **kwargs,
        }
        if self.model.startswith("qwen"):
            params["extra_body"] = {"enable_thinking": False}

        start_time = time.time()
        first_token_time = None
        full_content = ""
        usage = None

        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices == [] and chunk.usage is not None:
                    usage = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_content += delta.content
                    if hook is not None:
                        await hook(full_content)

            assert first_token_time is not None
            assert usage is not None
        except Exception as e:
            raise Exception(f"LLM async request failed: {e}")

        end_time = time.time()

        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        self.update_token_dict(self.model, usage)

        return {
            "prompt": prompt,
            "model": self.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "content": full_content,
            "total_time": end_time - start_time,
            "ttft": first_token_time - start_time,
            "tps": completion_tokens / (end_time - first_token_time),
            "image_format": base64_res["format"],
            "image_size": base64_res["size"],
            "image_bytes": base64_res["bytes"],
        }

    async def chat_with_image_list_detail(
        self,
        prompt: str,
        image_list: list[Image],
        default_fmt="JPEG",
        hook: Callable[[str], Awaitable[None]] | None = None,
        **kwargs,
    ) -> ChatImageListDetails:
        base64_res_list = [image_to_base64(image, default_fmt) for image in image_list]

        params = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[{"type": "image_url", "image_url": {"url": b["base64"]}} for b in base64_res_list],
                    ],
                }
            ],
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "include_obfuscation": False,
            },
            **kwargs,
        }
        if self.model.startswith("qwen"):
            params["extra_body"] = {"enable_thinking": False}

        start_time = time.time()
        first_token_time = None
        full_content = ""
        usage = None

        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices == [] and chunk.usage is not None:
                    usage = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_content += delta.content
                    if hook is not None:
                        await hook(full_content)

            assert first_token_time is not None
            assert usage is not None
        except Exception as e:
            raise Exception(f"LLM async request failed: {e}")

        end_time = time.time()

        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        self.update_token_dict(self.model, usage)

        return {
            "prompt": prompt,
            "model": self.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "content": full_content,
            "total_time": end_time - start_time,
            "ttft": first_token_time - start_time,
            "tps": completion_tokens / (end_time - first_token_time),
            "image_list": [
                {
                    "image_format": b["format"],
                    "image_size": b["size"],
                    "image_bytes": b["bytes"],
                }
                for b in base64_res_list
            ],
        }


class LLMs:
    vlm_primary: LLM
    llm_primary: LLM

    vlm_secondary: LLM
    llm_secondary: LLM

    @classmethod
    def init(
        cls,
        vlm_primary_config: dict,
        llm_primary_config: dict,
        vlm_secondary_config: dict,
        llm_secondary_config: dict,
    ) -> None:
        cls.vlm_primary = LLM(**vlm_primary_config)
        cls.llm_primary = LLM(**llm_primary_config)
        cls.vlm_secondary = LLM(**vlm_secondary_config)
        cls.llm_secondary = LLM(**llm_secondary_config)
