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
            # args=[
            #     f"--disable-extensions-except=local/I-Still-Dont-Care-About-Cookies",
            #     f"--load-extension=local/I-Still-Dont-Care-About-Cookies",
            # ],
            viewport={
                "width": Config.browser_viewport_w,
                "height": Config.browser_viewport_h,
            },
        )
        await context.clear_cookies()
        await context.add_init_script(path="agent/scripts/cssSelectorGenerator.js")

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
            task="在B站找到一个关于如何在笔记本电脑部署Qwen大模型的教程视频",
            start_url="https://www.bilibili.com/",
        )
    )
