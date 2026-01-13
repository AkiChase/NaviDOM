import asyncio
from pathlib import Path
import shutil
from playwright.async_api import async_playwright

from agent.config import Config
from agent.agent import Agent


async def main():
    Config.init("env.json")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=Config.browser_executable_path,
            headless=Config.browser_headless,
        )
        context = await browser.new_context(
            viewport={"width": Config.browser_viewport_w, "height": Config.browser_viewport_h}
        )

        out_dir = Path("output/test")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        agent = Agent(out_dir=out_dir)
        await agent.run(context, "visit https://www.google.com/ and search for doubao")
        pass


if __name__ == "__main__":
    asyncio.run(main())
