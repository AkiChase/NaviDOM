import asyncio
from urllib.parse import unquote
from loguru import logger
from playwright.async_api import BrowserContext, Page


class TabManager:
    context: BrowserContext
    tab_dict: dict[int, Page]
    _next_tab_id: int
    cur_tab_id: int | None
    pre_tab_id: int | None

    def __init__(self, context: BrowserContext) -> None:
        self.context = context
        self.tab_dict = {}
        self._next_tab_id = 1
        self.cur_tab_id = None
        self.pre_tab_id = None
        self.context.on("page", self._on_new_tab)

    def __del__(self):
        self.context.remove_listener("page", self._on_new_tab)

    async def _on_new_tab(self, tab: Page):
        tab_id = self._next_tab_id
        self._next_tab_id += 1
        self.tab_dict[tab_id] = tab
        self.pre_tab_id = self.cur_tab_id
        self.cur_tab_id = tab_id
        logger.debug(f"[TabManager] Tab opened, id={tab_id}, url={tab.url}")
        tab.on("close", lambda p: self._on_tab_closed(p, tab_id))
        await tab.add_init_script("Object.defineProperties(navigator, {webdriver:{get:()=>undefined}});")
        await tab.evaluate("()=>{Object.defineProperty(navigator, 'webdriver', {get:()=>undefined});}")
        await tab.bring_to_front()

    def _on_tab_closed(self, tab: Page, tab_id: int):
        if tab_id in self.tab_dict:
            tab = self.tab_dict.pop(tab_id)
            logger.debug(f"[TabManager] Tab closed, id={tab_id}, url={tab.url}")
            if tab_id == self.cur_tab_id:
                self.cur_tab_id = self.pre_tab_id
                self.pre_tab_id = None
            else:
                if tab_id == self.pre_tab_id:
                    self.pre_tab_id = None

    async def reset_context_tabs(self):
        for tab in reversed(self.context.pages):
            await tab.close()
        self.tab_dict = {}
        self._next_tab_id = 1
        self.cur_tab_id = None
        self.pre_tab_id = None

        _ = await self.context.new_page()
        for _ in range(20):
            await asyncio.sleep(0.1)
            if self.pre_tab_id is not None:
                break

    async def get_tabs_info(self) -> str:

        tabs_info = [
            f"{id}. **{'Visible' if self.cur_tab_id == id else 'Background'}** {await tab.title()} {unquote(tab.url)[:100]}"
            for id, tab in self.tab_dict.items()
        ]
        if self.pre_tab_id is not None:
            tabs_info.append(f"Previous Visible tab ID: {self.pre_tab_id}")
        return "\n".join(tabs_info)

    async def get_cur_tab_info(self) -> dict:
        for _ in range(10):
            try:
                front_tab = self.front_tab
                await front_tab.wait_for_load_state("load")
                return {
                    "tab_id": self.cur_tab_id,
                    "title": await front_tab.title(),
                    "url": front_tab.url,
                }
            except Exception as e:
                logger.warning(f"[TabManager] Failed to get tab info, error: {e}")
                await asyncio.sleep(0.5)
        raise RuntimeError("Failed to get tab info after 10 retries.")

    @staticmethod
    def compare_tab_info(new_tab_info: dict, old_tab_info: dict) -> str | None:
        if new_tab_info["tab_id"] != old_tab_info["tab_id"]:
            return "Tab switched: " f"tab_id {old_tab_info['tab_id']} → {new_tab_info['tab_id']}."

        old_title, new_title = old_tab_info["title"], new_tab_info["title"]
        old_url, new_url = old_tab_info["url"], new_tab_info["url"]

        messages = []
        if old_url != new_url:
            messages.append(
                f"Tab URL changed from '{old_url}' to '{new_url}', indicating navigation within the same tab."
            )

        if old_title != new_title:
            messages.append(
                f"Tab title changed from '{old_title}' to '{new_title}', indicating a possible change in page content or state."
            )

        if not messages:
            return None

        return "Tab not switched." + " ".join(messages)

    @property
    def front_tab(self) -> Page:
        assert self.cur_tab_id is not None
        return self.tab_dict[self.cur_tab_id]
