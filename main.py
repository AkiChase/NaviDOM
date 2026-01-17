import asyncio
from pathlib import Path
import shutil
from playwright.async_api import async_playwright

from agent.config import Config
from agent.agent import Agent


async def main():
    out_dir = Path("output/test")
    Config.init("env.json", out_dir)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=Config.browser_executable_path,
            headless=Config.browser_headless,
        )
        context = await browser.new_context(
            viewport={"width": Config.browser_viewport_w, "height": Config.browser_viewport_h}
        )

        if out_dir.exists():
            shutil.rmtree(out_dir)
        task = "Go to https://www.traderjoes.com/. Find the store location and hours of the closest Trader Joe's to zip code 90028 and set it as my home store."
        # task = "在github网站上查找浏览器自动化相关的star数量最高的项目，给出url"
        agent = Agent(
            out_dir=out_dir,
            context=context,
            user_request=task,
        )
        await agent.run()
        pass


if __name__ == "__main__":
    asyncio.run(main())
