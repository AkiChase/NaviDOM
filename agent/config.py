import json
from pathlib import Path
import shutil
import sys
from loguru import logger

from agent.llm import PrimaryLLM, SecondaryLLM


class Config:
    debug: bool
    browser_headless: bool
    browser_executable_path: str | None
    browser_viewport_w: int
    browser_viewport_h: int
    max_act_retry_times: int
    max_iteration_times: int

    @classmethod
    def init(cls, env_path: str, out_dir: Path):
        if out_dir.exists():
            shutil.rmtree(out_dir)

        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )
        logger.add(
            out_dir / "log.log",
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
        cls.max_act_retry_times = env.get("max_act_retry_times", 3)
        cls.max_iteration_times = env.get("max_iteration_times", 10)

        primary_llm_service = env["primary_llm_service"]
        secondary_llm_service = env["secondary_llm_service"]
        primary_llm_config = env[primary_llm_service]
        secondary_llm_config = env[secondary_llm_service]

        PrimaryLLM.init(
            api_key=primary_llm_config["api_key"],
            base_url=primary_llm_config["base_url"],
            temperature=primary_llm_config["temperature"],
            text_model=primary_llm_config["text_model"],
            image_model=primary_llm_config["image_model"],
        )
        SecondaryLLM.init(
            api_key=secondary_llm_config["api_key"],
            base_url=secondary_llm_config["base_url"],
            temperature=primary_llm_config["temperature"],
            text_model=secondary_llm_config["text_model"],
            image_model=secondary_llm_config["image_model"],
        )
