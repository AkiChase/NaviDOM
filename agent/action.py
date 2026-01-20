import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict
import typing

from agent.config import Config
from agent.dom_utils import DomNode
from playwright.async_api import Page, CDPSession

from agent.llm import ChatImageDetails
from agent.utils import SpecialException, bing_search_url

if typing.TYPE_CHECKING:
    from agent.agent import TabManager


class ActionParseException(SpecialException):
    pass


class ActionExecuteException(SpecialException):
    pass


class ActionType(Enum):
    Click = "CLICK"
    Input = "INPUT"
    Scroll = "SCROLL"
    Press = "PRESS"
    Navigate = "NAVIGATE"
    Search = "SEARCH"
    TabSwitch = "TAB_SWITCH"
    TabClose = "TAB_CLOSE"


action_format_prompt = {
    ActionType.Click: "CLICK, <backend_node_id>",
    ActionType.Input: "INPUT, <backend_node_id>, <clear:true|false>, <text>\t// In <text>, only \\n has special meaning for line breaks; no other escaping is required",
    ActionType.Scroll: "SCROLL, <direction>, <pages>\t// Scroll by <pages:float> viewport-height pages in the given <direction:up|down>",
    ActionType.Press: "PRESS, <key>\t// Must follow KeyboardEvent.key (e.g. a, Enter, Control+o)",
    ActionType.Navigate: "NAVIGATE, <url>\t// Navigate to the specified URL",
    ActionType.Search: "SEARCH, <keywords>\t// Navigate to Bing search results for the given keywords",
    ActionType.TabSwitch: "TAB_SWITCH, <tab_id>\t// Switch to and activate the specified tab",
    ActionType.TabClose: "TAB_CLOSE, <tab_id>\t// Close the specified tab",
}
all_action_types = [
    ActionType.Click,
    ActionType.Input,
    ActionType.Scroll,
    ActionType.Press,
    ActionType.Navigate,
    ActionType.Search,
    ActionType.TabSwitch,
    ActionType.TabClose,
]


class Action:
    uid: str
    type: ActionType
    target: DomNode | None
    extra: dict

    def __init__(self, uid: str, action_type: ActionType, target: DomNode | None, extra: dict):
        self.uid = uid
        self.type = action_type
        self.target = target
        self.extra = extra

    def get_description(self) -> str:
        out = [self.type.value]
        if self.target:
            out.append(f"target={self.target.get_description()}")
        for k, v in self.extra.items():
            out.append(f"{k}={v}")
        return ", ".join(out)

    @staticmethod
    def get_format_prompt(include_types: list[ActionType] | None = None, exclude_types: list[ActionType] | None = None):
        if include_types is None:
            include_types = all_action_types
        if exclude_types is None:
            exclude_types = []

        return "\n".join([f"- {action_format_prompt[t]}" for t in include_types if t not in exclude_types])

    @staticmethod
    def from_raw_action(uid: str, raw_action: str, dom_nodes: list[DomNode], tab_manager: "TabManager") -> "Action":
        raw_action = raw_action.strip()
        assert raw_action, "Empty action line"

        parts = [p.strip() for p in raw_action.split(",", maxsplit=1)]
        action_type = ActionType(parts[0].upper())

        def find_target(backend_node_id: str) -> DomNode:
            target_id = int(backend_node_id.strip("[]<>"))
            for node in dom_nodes:
                target = node.find_nodes_by_backend_node_ids([target_id])[0]
                if target is not None:
                    return target
            raise ActionParseException(f"backend_node_id [{target_id}] not found")

        if action_type == ActionType.Click:
            # CLICK, <backend_node_id>
            assert len(parts) == 2, ActionParseException(f"Invalid CLICK format: {raw_action}")
            target = find_target(parts[1])
            return Action(uid, action_type, target, {})
        elif action_type == ActionType.Input:
            # INPUT, <backend_node_id>, <clear>, <text>
            assert len(parts) == 2, ActionParseException(f"Invalid INPUT format: {raw_action}")
            sub_parts = [p.strip() for p in parts[1].split(",", maxsplit=2)]
            assert len(sub_parts) == 3, ActionParseException(f"Invalid INPUT format: {raw_action}")
            clear_raw = sub_parts[1].lower()
            assert clear_raw in ("true", "false"), ActionParseException(f"Invalid clear value: {clear_raw}")
            clear = clear_raw == "true"
            text = sub_parts[2].replace("\\n", "\n")
            target = find_target(sub_parts[0])
            return Action(
                uid,
                action_type,
                target,
                {
                    "clear": clear,
                    "text": text,
                },
            )
        elif action_type == ActionType.Scroll:
            # SCROLL, <direction>, <pages>
            assert len(parts) == 2, ActionParseException(f"Invalid SCROLL format: {raw_action}")
            sub_parts = [p.strip() for p in parts[1].split(",", maxsplit=1)]
            assert len(sub_parts) == 2, ActionParseException(f"Invalid SCROLL format: {raw_action}")
            direction = sub_parts[0].lower()
            assert direction in ("up", "down"), ActionParseException(f"Invalid scroll direction: {direction}")
            pages = float(sub_parts[1])
            return Action(uid, action_type, None, {"direction": direction, "pages": pages})
        elif action_type == ActionType.Press:
            # PRESS, <KeyboardEvent.key>
            assert len(parts) == 2, ActionParseException(f"Invalid PRESS format: {raw_action}")
            key = parts[1]
            return Action(uid, action_type, None, {"key": key})
        elif action_type == ActionType.Navigate:
            # NAVIGATE, <url>
            assert len(parts) == 2, ActionParseException(f"Invalid NAVIGATE format: {raw_action}")
            url = parts[1]
            return Action(uid, action_type, None, {"url": url})
        elif action_type == ActionType.Search:
            # SEARCH, <keywords>
            assert len(parts) == 2, ActionParseException(f"Invalid SEARCH format: {raw_action}")
            keywords = parts[1]
            return Action(uid, action_type, None, {"keywords": keywords})
        elif action_type == ActionType.TabSwitch:
            # TAB_SWITCH, <tab_id>
            assert len(parts) == 2, ActionParseException(f"Invalid TAB_SWITCH format: {raw_action}")
            tab_id = int(parts[1])
            if tab_id not in tab_manager.tab_dict:
                raise ActionParseException(f"Tab ID {tab_id} not found")
            return Action(uid, action_type, None, {"tab_id": tab_id})
        elif action_type == ActionType.TabClose:
            # TAB_CLOSE, <tab_id>
            assert len(parts) == 2, ActionParseException(f"Invalid TAB_CLOSE format: {raw_action}")
            tab_id = int(parts[1])
            if tab_id not in tab_manager.tab_dict:
                raise ActionParseException(f"Tab ID {tab_id} not found")
            return Action(uid, action_type, None, {"tab_id": tab_id})
        else:
            raise ActionParseException(f"Unsupported action type: {action_type}")

    async def execute(self, tab: Page, cdp_session: CDPSession, tab_manager: "TabManager"):
        # 简单实现
        node = None
        if self.type in [ActionType.Click, ActionType.Input]:
            assert self.target is not None

            if self.type == ActionType.Click:
                loc = await self.target.find_node_in_tab(tab)
                if loc is not None:
                    try:
                        await loc.click(timeout=300)
                    except Exception as e:
                        raise ActionExecuteException(f"Failed to click: {e}")
                else:
                    node = await cdp_session.send("DOM.resolveNode", {"backendNodeId": self.target.backend_node_id})
                    if "object" not in node or "objectId" not in node["object"]:
                        raise ActionExecuteException("Failed to get object ID for element")
                    object_id = node["object"]["objectId"]
                    await cdp_session.send(
                        "Runtime.callFunctionOn",
                        {
                            "objectId": object_id,
                            "functionDeclaration": """function() {
                                this.click();
                            }
                            """,
                        },
                    )
                await asyncio.sleep(1.5)
            elif self.type == ActionType.Input:
                if not self.target.editable:
                    raise ActionExecuteException(f"Target {self.target} is not editable")
                
                clear = self.extra["clear"]
                text = self.extra["text"]
                loc = await self.target.find_node_in_tab(tab)
                if loc is not None:
                    try:
                        if clear:
                            await loc.fill(text, timeout=300)
                        else:
                            await loc.type(text, timeout=300)
                    except Exception as e:
                        raise ActionExecuteException(f"Failed to click: {e}")
                else:
                    node = await cdp_session.send("DOM.resolveNode", {"backendNodeId": self.target.backend_node_id})
                    if "object" not in node or "objectId" not in node["object"]:
                        raise ActionExecuteException("Failed to get object ID for element")
                    object_id = node["object"]["objectId"]
                    await cdp_session.send("DOM.focus", {"objectId": object_id})
                    if clear:
                        value_stmt = "try { this.select(); } catch (e) {}\n"
                        value_stmt += f'this.value = "{text}";'
                    else:
                        value_stmt = f'this.value += "{text}";'

                    await cdp_session.send(
                        "Runtime.callFunctionOn",
                        {
                            "objectId": object_id,
                            "functionDeclaration": """function() {
                            <Mask>
                            this.dispatchEvent(new Event("input", { bubbles: true }));
                            this.dispatchEvent(new Event("change", { bubbles: true }));
                        }""".replace(
                                "<Mask>", value_stmt
                            ),
                        },
                    )
                await asyncio.sleep(1.5)
        elif self.type == ActionType.Scroll:
            pages = self.extra["pages"]
            dy = Config.browser_viewport_h * pages
            direction = self.extra["direction"]
            if direction == "up":
                dy = -dy
            await tab.evaluate(f"window.scrollBy(0, {dy});")
            await asyncio.sleep(1)
        elif self.type == ActionType.Press:
            key = self.extra["key"]
            await tab.keyboard.press(key)
            await asyncio.sleep(1)
        elif self.type == ActionType.Navigate:
            url = self.extra["url"]
            await tab.goto(url)
            await asyncio.sleep(1)
        elif self.type == ActionType.Search:
            keywords = self.extra["keywords"]
            await tab.goto(bing_search_url(keywords))
            await asyncio.sleep(1)
        elif self.type == ActionType.TabSwitch:
            tab_id = self.extra["tab_id"]
            if tab_id not in tab_manager.tab_dict:
                raise ActionExecuteException(f"Tab ID {tab_id} not found")
            tab = tab_manager.tab_dict[tab_id]
            tab_manager.latest_tab_id = tab_id
            await tab.bring_to_front()
            await asyncio.sleep(0.5)
        elif self.type == ActionType.TabClose:
            tab_id = self.extra["tab_id"]
            if tab_id not in tab_manager.tab_dict:
                raise ActionExecuteException(f"Tab ID {tab_id} not found")
            tab = tab_manager.tab_dict[tab_id]
            await tab.close()
            await asyncio.sleep(0.5)
        else:
            raise NotImplementedError(f"Action type {self.type} is not implemented")


class ActionExecuteResult(TypedDict):
    success: bool
    additional: ChatImageDetails | None | str  # Extract/Other/Error


@dataclass
class ActionDetails:
    action: Action | None
    raw_action: str
    execute_result: ActionExecuteResult
    action_screenshot_path: Path
    result_screenshot_path: Path

    def to_dict(self) -> dict:
        return {
            "raw_action": self.raw_action,
            "execute_result": self.execute_result,
            "action_screenshot_path": str(self.action_screenshot_path),
            "result_screenshot_path": str(self.result_screenshot_path),
        }
