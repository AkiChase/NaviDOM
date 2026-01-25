import asyncio
from urllib.parse import unquote
from loguru import logger
from playwright.async_api import BrowserContext, Page


class TabManager:
    context: BrowserContext
    tab_dict: dict[int, Page]
    _next_tab_id: int
    latest_tab_id: int | None
    last_tab_id: int | None

    def __init__(self, context: BrowserContext) -> None:
        self.context = context
        self.tab_dict = {}
        self._next_tab_id = 1
        self.latest_tab_id = None
        self.last_tab_id = None
        self.context.on("page", self._on_new_tab)

    def __del__(self):
        self.context.remove_listener("page", self._on_new_tab)

    async def _on_new_tab(self, tab: Page):
        tab_id = self._next_tab_id
        self._next_tab_id += 1
        self.tab_dict[tab_id] = tab
        self.last_tab_id = self.latest_tab_id
        self.latest_tab_id = tab_id
        logger.debug(f"[TabManager] Tab opened, id={tab_id}, url={tab.url}")
        tab.on("close", lambda p: self._on_tab_closed(p, tab_id))
        await tab.add_init_script("Object.defineProperties(navigator, {webdriver:{get:()=>undefined}});")
        await tab.evaluate("()=>{Object.defineProperty(navigator, 'webdriver', {get:()=>undefined});}")
        await tab.bring_to_front()

    def _on_tab_closed(self, tab: Page, tab_id: int):
        if tab_id in self.tab_dict:
            tab = self.tab_dict.pop(tab_id)
            logger.debug(f"[TabManager] Tab closed, id={tab_id}, url={tab.url}")
            if tab_id == self.latest_tab_id:
                self.latest_tab_id = self.last_tab_id
                self.last_tab_id = None
            else:
                if tab_id == self.last_tab_id:
                    self.last_tab_id = None

    async def reset_context_tabs(self):
        for tab in reversed(self.context.pages):
            await tab.close()
        self.tab_dict = {}
        self._next_tab_id = 1
        self.latest_tab_id = None
        self.last_tab_id = None

        _ = await self.context.new_page()
        for _ in range(20):
            await asyncio.sleep(0.1)
            if self.last_tab_id is not None:
                break

    async def get_tabs_info(self) -> str:
        tabs_info = [f"{id}. {await tab.title()} {unquote(tab.url)}" for id, tab in self.tab_dict.items()]
        return "\n".join(tabs_info)

    def get_tab_id_info(self) -> str:
        return f"Current Tab ID: {self.latest_tab_id}{f', previous tab ID: {self.last_tab_id}' if self.last_tab_id is not None else ''}"

    @property
    def front_tab(self) -> Page:
        assert self.latest_tab_id is not None
        return self.tab_dict[self.latest_tab_id]
