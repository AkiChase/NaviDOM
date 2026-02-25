import json
from pathlib import Path
import shutil
import sys
from loguru import logger

from agent.llm import LLMs


class Config:
    debug: bool
    browser_viewport_w: int
    browser_viewport_h: int
    max_iteration_times: int

    @classmethod
    def init(cls, env_path: str, out_dir: Path):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

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
        viewport = browser_config.get("viewport", {})
        cls.browser_viewport_w = viewport.get("width", 1280)
        cls.browser_viewport_h = viewport.get("height", 720)
        cls.max_iteration_times = env.get("max_iteration_times", 10)

        LLMs.init(
            vlm_primary_config=env[env["vlm_primary_service"]],
            llm_primary_config=env[env["llm_primary_service"]],
            vlm_secondary_config=env[env["vlm_secondary_service"]],
            llm_secondary_config=env[env["llm_secondary_service"]],
        )