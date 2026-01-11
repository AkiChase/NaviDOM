import json
from pathlib import Path
import sys
from loguru import logger

from agent.llm import PrimaryLLM, SecondaryLLM


class Config:
    debug: bool
    browser_headless: bool
    browser_executable_path: str | None
    browser_viewport_w: int
    browser_viewport_h: int

    @classmethod
    def init(cls, env_path: str):
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )

        env: dict = json.loads(Path(env_path).read_text())

        cls.debug = env.get("debug", False)
        browser_config = env.get("browser", {})
        cls.browser_headless = browser_config.get("headless", False)
        cls.browser_executable_path = browser_config.get("executable_path", None)
        viewport = browser_config.get("viewport", {})
        cls.browser_viewport_w = viewport.get("width", 1280)
        cls.browser_viewport_h = viewport.get("height", 720)

        primary_llm_service = env["primary_llm_service"]
        secondary_llm_service = env["secondary_llm_service"]
        primary_llm_config = env[primary_llm_service]
        secondary_llm_config = env[secondary_llm_service]

        PrimaryLLM.init(
            api_key=primary_llm_config["api_key"],
            base_url=primary_llm_config["base_url"],
            temperature=0.0,
            text_model=primary_llm_config["text_model"],
            image_model=primary_llm_config["image_model"],
        )
        SecondaryLLM.init(
            api_key=secondary_llm_config["api_key"],
            base_url=secondary_llm_config["base_url"],
            temperature=0.0,
            text_model=secondary_llm_config["text_model"],
            image_model=secondary_llm_config["image_model"],
        )
