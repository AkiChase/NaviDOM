import argparse
import asyncio
from pathlib import Path
from loguru import logger
from playwright.async_api import async_playwright

from agent.config import Config
from agent.agent import Agent


async def run_agent(
    out_dir: Path,
    task: str,
    start_url: str | None = None,
):
    Config.init("env.json", out_dir)
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="",
            headless=False,
            args=[
                f"--disable-extensions-except=local/I-Still-Dont-Care-About-Cookies",
                f"--load-extension=local/I-Still-Dont-Care-About-Cookies",
            ],
            viewport={
                "width": Config.browser_viewport_w,
                "height": Config.browser_viewport_h,
            },
        )
        await context.clear_cookies()
        await context.add_init_script(path="agent/scripts/cssSelectorGenerator.js")

        out_dir.mkdir(parents=True, exist_ok=True)
        agent = Agent(
            out_dir=out_dir,
            context=context,
            user_request=task,
        )
        if start_url:
            await agent.run(start_url=start_url)
        else:
            await agent.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run browser agent with a natural language task")
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for logs, traces, and report",
    )
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        help="User task in natural language",
    )
    parser.add_argument(
        "--start-url",
        type=str,
        default=None,
        help="Optional start URL for the browser agent",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        asyncio.run(
            run_agent(
                out_dir=args.out_dir,
                task=args.task,
                start_url=args.start_url,
            )
        )
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
