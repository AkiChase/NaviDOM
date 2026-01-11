from io import BytesIO
import os
import platform
from typing import Union, Tuple
import uuid
from PIL import ImageDraw, ImageFont, Image
from loguru import logger
from playwright.async_api import Page


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


async def page_screenshot(page: Page):
    screenshot_bytes = await page.screenshot(full_page=False, type="jpeg")
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


def gen_uid():
    return uuid.uuid4().hex
