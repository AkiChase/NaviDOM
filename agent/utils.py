from datetime import datetime, timedelta
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from typing import Union, Tuple
import uuid
from PIL import ImageDraw, ImageFont, Image
from loguru import logger
from playwright.async_api import Page
from urllib.parse import quote


class SpecialException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]


Color = Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]

bg_colors = [
    "#FF6B6B",  # 暖红（珊瑚红）— 比纯红更柔和，但足够深
    "#4ECDC4",  # 青蓝（明亮但不过亮）— 白字清晰
    "#45B7D1",  # 天蓝（稍深）— 安全
    "#9C89B8",  # 灰紫（莫兰迪紫）— 优雅且对比度好
    "#FFA07A",  # 浅鲑红（替代橙色，比纯橙更深）
    "#6A0DAD",  # 深紫（经典紫）— 白字非常清晰
    "#FF6F61",  # 珊瑚橙（比 yellow/orange 更深，适合白字）
]


async def tab_screenshot(tab: Page):
    screenshot_bytes = await tab.screenshot(full_page=False, type="jpeg")
    return Image.open(BytesIO(screenshot_bytes))


def draw_text_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    text_color: Color = (255, 255, 255, 255),
    bg_color: Color = (0, 0, 0, 160),
    padding: int = 6,
):
    x, y = position

    # Text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    box_w = text_w + padding * 2
    box_h = text_h + padding * 2

    box = (x, y, x + box_w, y + box_h)

    # Background
    draw.rectangle(
        box,
        fill=bg_color,
    )

    # Text
    box_center_x = x + box_w / 2
    box_center_y = y + box_h / 2
    draw.text((box_center_x, box_center_y), text, fill=text_color, font=font, anchor="mm")


def load_default_font(font_size: int):
    system_name = platform.system()
    if system_name == "Windows":
        font_path = "C:\\Windows\\Fonts\\arial.ttf"
    elif system_name == "Darwin":  # macOS
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    else:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if not os.path.exists(font_path):
        logger.warning("Font not found: {}. Use default font instead.", font_path)
        return ImageFont.load_default(size=font_size)
    return ImageFont.truetype(font_path, size=font_size)


css_ident_re = re.compile(r"^-?[_a-zA-Z][_a-zA-Z0-9-]*$")


def css_escape(value: str) -> str:
    """
    Escape a string so it can be safely used in a CSS selector.
    This is a simplified but practical implementation.
    """
    if css_ident_re.match(value):
        return value

    result = []
    for i, ch in enumerate(value):
        code = ord(ch)

        # NULL
        if ch == "\0":
            result.append("\ufffd")
            continue

        # control chars or non-ASCII
        if code < 0x20 or code > 0x7E:
            result.append(f"\\{code:X} ")
            continue

        # special CSS chars
        if ch in r' !"#$%&\'()*+,./:;<=>?@[\]^`{|}~':
            result.append(f"\\{ch}")
            continue

        # identifier starting with digit
        if i == 0 and ch.isdigit():
            result.append(f"\\{code:X} ")
            continue

        result.append(ch)

    return "".join(result)


def gen_uid():
    return uuid.uuid4().hex


def time_stamp(now: datetime | None = None):
    # YYYY/MM/DD-HH:MM:SS
    if not now:
        now = datetime.now()
    return now.strftime("%Y/%m/%d-%H:%M:%S")


def format_time_delta(start: datetime | float, end: datetime | float, with_ms: bool = True) -> str:
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        delta_seconds = end - start
        delta = timedelta(seconds=delta_seconds)
    elif isinstance(start, datetime) and isinstance(end, datetime):
        delta = end - start
    else:
        raise TypeError("start and end must be both datetime or both float")

    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if with_ms:
        milliseconds = int(delta.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_seconds(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def google_search_url(keywords: str) -> str:
    encoded = quote(keywords, safe="")
    return f"https://www.google.com/search?q={encoded}"


def bing_search_url(keywords: str) -> str:
    encoded = quote(keywords, safe="")
    return f"https://www.bing.com/search?q={encoded}"


def load_prompts():
    prompt_dir = Path(__file__).parent / "prompts"
    prompts = {}
    for prompt_file in prompt_dir.glob("*.md"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts[prompt_file.stem] = f.read().strip()
    logger.info(f"Loaded {len(prompts)} prompts: {prompts.keys()}")
    return prompts
