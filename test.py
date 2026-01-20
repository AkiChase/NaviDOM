import asyncio
from playwright.async_api import async_playwright
from pathlib import Path

from agent.agent import Agent
from agent.config import Config


async def run_agent(
    out_dir: Path,
    task: str,
    start_url: str | None = None,
):
    Config.init("env.json", out_dir)
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=Config.browser_executable_path,
            headless=Config.browser_headless,
        )
        context = await browser.new_context(
            viewport={
                "width": Config.browser_viewport_w,
                "height": Config.browser_viewport_h,
            }
        )
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


if __name__ == "__main__":
    asyncio.run(
        run_agent(
            out_dir=Path("output/test"),
            task="Find the store location and Business Hours of the closest Trader Joe's to zip code 90028 and set it as my home store. You must find the Business Hours of it",
            start_url="https://www.traderjoes.com/",
        )
    )
