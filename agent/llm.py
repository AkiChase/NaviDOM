import base64
from typing import TypedDict
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


class PrimaryLLM:
    client: AsyncClient
    text_model: str
    image_model: str
    temperature: float
    tokenDict: dict[str, TokenInfo]

    @classmethod
    def init(
        cls,
        api_key: str,
        base_url: str,
        temperature: float,
        text_model: str,
        image_model: str,
    ) -> None:
        cls.temperature = temperature
        cls.client = AsyncClient(api_key=api_key, base_url=base_url)
        cls.text_model = text_model
        cls.image_model = image_model
        cls.tokenDict = {}

    @classmethod
    def update_token_dict(cls, model: str, usage: CompletionUsage):
        if model not in cls.tokenDict:
            cls.tokenDict[model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        cls.tokenDict[model]["completion_tokens"] += usage.completion_tokens
        cls.tokenDict[model]["prompt_tokens"] += usage.prompt_tokens
        cls.tokenDict[model]["total_tokens"] += usage.total_tokens

    @classmethod
    async def chat_with_text_detail(cls, prompt: str, **kwargs) -> ChatTextDetails:
        params = {
            "model": cls.text_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": cls.temperature,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "include_obfuscation": False,
            },
            **kwargs,
        }

        start_time = time.time()
        first_token_time = None
        full_content = []
        usage = None

        try:
            stream = await cls.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices == [] and chunk.usage is not None:
                    usage = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_content.append(delta.content)

            assert first_token_time is not None
            assert usage is not None
        except Exception as e:
            raise Exception(f"LLM async request failed: {e}")

        end_time = time.time()

        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        cls.update_token_dict(cls.text_model, usage)

        return {
            "prompt": prompt,
            "model": cls.text_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "content": "".join(full_content),
            "total_time": end_time - start_time,
            "ttft": first_token_time - start_time,
            "tps": completion_tokens / (end_time - first_token_time),
        }

    @classmethod
    async def chat_with_image_detail(cls, prompt: str, image: Image, default_fmt="JPEG", **kwargs) -> ChatImageDetails:
        base64_res = image_to_base64(image, default_fmt)

        params = {
            "model": cls.image_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": base64_res["base64"]}},
                    ],
                }
            ],
            "temperature": cls.temperature,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "include_obfuscation": False,
            },
            **kwargs,
        }

        start_time = time.time()
        first_token_time = None
        full_content = []
        usage = None

        try:
            stream = await cls.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices == [] and chunk.usage is not None:
                    usage = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_content.append(delta.content)

            assert first_token_time is not None
            assert usage is not None
        except Exception as e:
            raise Exception(f"LLM async request failed: {e}")

        end_time = time.time()

        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        cls.update_token_dict(cls.text_model, usage)

        return {
            "prompt": prompt,
            "model": cls.image_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "content": "".join(full_content),
            "total_time": end_time - start_time,
            "ttft": first_token_time - start_time,
            "tps": completion_tokens / (end_time - first_token_time),
            "image_format": base64_res["format"],
            "image_size": base64_res["size"],
            "image_bytes": base64_res["bytes"],
        }

    @classmethod
    async def chat_with_image_list_detail(
        cls, prompt: str, image_list: list[Image], default_fmt="JPEG", **kwargs
    ) -> ChatImageListDetails:
        base64_res_list = [image_to_base64(image, default_fmt) for image in image_list]

        params = {
            "model": cls.image_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[{"type": "image_url", "image_url": {"url": b["base64"]}} for b in base64_res_list],
                    ],
                }
            ],
            "temperature": cls.temperature,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "include_obfuscation": False,
            },
            **kwargs,
        }

        start_time = time.time()
        first_token_time = None
        full_content = []
        usage = None

        try:
            stream = await cls.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices == [] and chunk.usage is not None:
                    usage = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_content.append(delta.content)

            assert first_token_time is not None
            assert usage is not None
        except Exception as e:
            raise Exception(f"LLM async request failed: {e}")

        end_time = time.time()

        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        cls.update_token_dict(cls.text_model, usage)

        return {
            "prompt": prompt,
            "model": cls.image_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "content": "".join(full_content),
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


class SecondaryLLM(PrimaryLLM):
    pass
