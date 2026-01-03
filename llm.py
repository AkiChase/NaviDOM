import base64
import math
from io import BytesIO
import time
from openai import AsyncOpenAI as AsyncClient
from PIL.Image import Image


class LLM:
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

    @staticmethod
    def _image_to_base64(image: Image, default_fmt="JPEG") -> dict:
        buffer = BytesIO()
        fmt = image.format or default_fmt
        image.save(buffer, format=fmt)
        img_bytes = buffer.getvalue()
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "base64": f"data:image/{fmt.lower()};base64,{base64_str}",
            "size": image.size,  # (width, height)
            "bytes": len(img_bytes),  # 字节数
            "format": fmt,
        }

    @classmethod
    async def chat_detail(cls, prompt: str, **kwargs):
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
    async def chat_with_image_detail(cls, prompt: str, image: Image, default_fmt="JPEG", **kwargs):
        base64_res = LLM._image_to_base64(image, default_fmt)

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


class LocalLLM(LLM):
    pass


def estimate_image_tokens(width: int, height: int, model_id: str, detail: str = "high"
                          ) -> tuple[int, tuple[float, float]]:
    """估算图片 token 消耗"""
    # GPT-4.1-mini / GPT-4.1-nano / o4-mini
    patch_models = {
        "gpt-5-mini": 1.62,
        "gpt-5-nano": 2.46,
        "gpt-4.1-mini": 1.62,
        "gpt-4.1-nano": 2.46,
        "gpt-o4-mini": 1.72,
    }
    if model_id in patch_models:
        # 计算 patch
        patches_w = math.ceil(width / 32)
        patches_h = math.ceil(height / 32)
        total_patches = patches_w * patches_h

        # 超过 1536 patch，按比例缩放
        if total_patches > 1536:
            r = math.sqrt(32 ** 2 * 1536 / (width * height))
            width_r, height_r = width * r, height * r
            patches_w = math.floor(width_r / 32)
            patches_h = math.floor(height_r / 32)
            total_patches = patches_w * patches_h

        return math.ceil(total_patches * patch_models[model_id]), (width, height)

    # GPT-4o / GPT-4.1 / o1 系列
    tile_models = {
        "gpt-5": (70, 140),
        "gpt-5-chat-latest": (70, 140),
        "gpt-4o": (85, 170),
        "gpt-4.1": (85, 170),
        "gpt-4.5": (85, 170),
        "gpt-4o-mini": (2833, 5667),
        "gpt-o1": (75, 150),
        "o1-pro": (75, 150),
        "gpt-o3": (75, 150),
    }
    if model_id in tile_models:
        base_tokens, tile_tokens = tile_models[model_id]

        if detail == "low":
            return base_tokens

        # 最长边 <= 2048
        width_s, height_s = width, height
        if max(width, height) > 2048:
            scale1 = 2048 / max(width, height)
            width_s = width * scale1
            height_s = height * scale1

        # 最短边 <= 768
        if min(width_s, height_s) > 768:
            scale2 = 768 / min(width_s, height_s)
            width_s *= scale2
            height_s *= scale2

        # 计算 512x512 tile 数量
        tiles_w = math.ceil(width_s / 512)
        tiles_h = math.ceil(height_s / 512)
        total_tiles = tiles_w * tiles_h

        return base_tokens + total_tiles * tile_tokens, (width_s, height_s)

    raise ValueError("Unknown model")
