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


if __name__ == "__main__":
    asyncio.run(
        run_agent(
            out_dir=Path("output/test"),
            task="Find the store location and Business Hours of the closest Trader Joe's to zip code 90028 and set it as my home store.",
            start_url="https://www.traderjoes.com/",
        )
    )
