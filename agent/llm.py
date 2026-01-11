import base64
from typing import TypedDict
from io import BytesIO
import time
from openai import AsyncOpenAI as AsyncClient
from PIL.Image import Image


class ChatTextDetails(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    content: str
    time_cost: float


class ChatImageDetails(ChatTextDetails):
    image_format: str
    image_size: tuple[int, int]
    image_bytes: int


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


class PrimaryLLM:
    client: AsyncClient
    text_model: str
    image_model: str
    temperature: float
    tokenDict: dict

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
    async def chat_detail(cls, prompt: str, **kwargs) -> ChatTextDetails:
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
            **kwargs,
        }

        try:
            start_time = time.time()
            completion = await cls.client.chat.completions.create(**params)
            time_cost = time.time() - start_time
        except Exception as e:
            raise Exception(f"LLM async request failed: {e}")

        usage = completion.usage
        res = completion.choices[0].message.content
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "content": res,
            "time_cost": time_cost,
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
            **kwargs,
        }

        try:
            start_time = time.time()
            completion = await cls.client.chat.completions.create(**params)
            time_cost = time.time() - start_time
        except Exception as e:
            raise Exception(f"LLM async request failed: {e}")

        if not completion.choices:
            raise Exception("LLM async response nothing")

        usage = completion.usage
        res = completion.choices[0].message.content

        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "content": res,
            "time_cost": time_cost,
            "image_format": base64_res["format"],
            "image_size": base64_res["size"],
            "image_bytes": base64_res["bytes"],
        }


class SecondaryLLM(PrimaryLLM):
    pass
