import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict

from agent.config import Config
from agent.dom import DomNode
from playwright.async_api import Page

from agent.utils import SpecialException, bing_search_url
from agent.tab import TabManager


class ActionParseException(SpecialException):
    pass


class ActionExecuteException(SpecialException):
    pass


class ActionType(Enum):
    Click = "CLICK"
    Input = "INPUT"
    Scroll = "SCROLL"
    SelectOption = "SELECT_OPTION"
    # Press = "PRESS"
    TabSwitch = "TAB_SWITCH"
    TabClose = "TAB_CLOSE"
    WAIT = "WAIT"
    Navigate = "NAVIGATE"
    Search = "SEARCH"


all_action_types = list(ActionType)

action_format_prompt = {
    ActionType.Click: "CLICK, <node_id>",
    ActionType.Input: "INPUT, <node_id>, <clear:true|false>, <text>\t// Focus and input <text> into the specified input or textarea node. In <text>, only \\n has special meaning for line breaks; no other escaping is required",
    ActionType.Scroll: "SCROLL, <direction>, <pages>\t// Scroll by <pages:float> viewport-height pages in the given <direction:up|down>, based on **Current Visible Tab scroll info**. Minimum scroll increment is 0.1 pages",
    ActionType.SelectOption: "SELECT_OPTION, <select_node_id>, <option_node_id>[, ...]\t// Select option(s) in a <select> HTML element (Exclusive to <select> nodes only; invalid for other node types). Single-select: exactly one <option_node_id>. Multi-select: multiple <option_node_id>s separated by commas",
    # ActionType.Press: "PRESS, <key>\t// Must follow KeyboardEvent.key (e.g. a, Enter, Control+o)",
    ActionType.TabSwitch: "TAB_SWITCH, <tab_id>\t// Switch to the specified tab making it the active and visible tab",
    ActionType.TabClose: "TAB_CLOSE, <tab_id>\t// Close the specified tab",
    ActionType.WAIT: "WAIT, <seconds>\t// Wait for <seconds:float> seconds",
    ActionType.Navigate: "NAVIGATE, <url>\t// Navigate to the specified URL",
    ActionType.Search: "SEARCH, <keywords>\t// Navigate to Bing search results for the given keywords",
    # ActionType.Memory: "MEMORY, <content>\t// If the user request requires certain information to be output, explicitly record <content> (\\n for line breaks) using this action.",
    # ActionType.ExtractAndMemory: "EXTRACT_AND_MEMORY, <subject>\t// The text in provided Interactive Nodes may be omitted. Use this action as an enhanced but more expensive version of MEMORY to extract content related to <subject> from the full text and record it.",
}


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
    def get_available_actions_prompt(
        include_types: list[ActionType] | None = None, exclude_types: list[ActionType] | None = None
    ):
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
        if len(parts) == 1:
            raise ActionParseException(f"Invalid action format: {raw_action}")

        try:
            action_type = ActionType(parts[0].upper())
        except:
            raise ActionParseException(f"Unknown action type: {parts[0]}")

        def find_target(local_id: str) -> DomNode:
            target_id = int(local_id.strip("[]<>"))
            for node in dom_nodes:
                target = node.find_children_by_local_ids([target_id])[0]
                if target is not None:
                    if target.tag == "":
                        target = target.parent
                        assert target is not None, f"Target text node[{target_id}] has no parent"
                    return target
            raise ActionParseException(f"Node[{target_id}] not found")

        if action_type == ActionType.Click:
            # CLICK, <local_id>
            assert len(parts) == 2, ActionParseException(f"Invalid CLICK format: {raw_action}")
            target = find_target(parts[1])
            if not target.clickable:
                raise ActionParseException(f"Target node {target.get_description(full=False)} is not clickable")
            return Action(uid, action_type, target, {})
        elif action_type == ActionType.Input:
            # INPUT, <local_id>, <clear>, <text>
            assert len(parts) == 2, ActionParseException(f"Invalid INPUT format: {raw_action}")
            sub_parts = [p.strip() for p in parts[1].split(",", maxsplit=2)]
            assert len(sub_parts) == 3, ActionParseException(f"Invalid INPUT format: {raw_action}")
            clear_raw = sub_parts[1].lower()
            assert clear_raw in ("true", "false"), ActionParseException(f"Invalid clear value: {clear_raw}")
            clear = clear_raw == "true"
            text = sub_parts[2].replace("\\n", "\n")
            if text and text[0] == '"' and text[-1] == '"':
                text = text[1:-1]
            target = find_target(sub_parts[0])
            if not target.editable:
                raise ActionParseException(
                    f"Target node {target.get_description(full=False)} is not editable, INPUT action must target an input or textarea element",
                    extra=target,
                )
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
        # elif action_type == ActionType.Press:
        #     # PRESS, <KeyboardEvent.key>
        #     assert len(parts) == 2, ActionParseException(f"Invalid PRESS format: {raw_action}")
        #     key = parts[1]
        #     return Action(uid, action_type, None, {"key": key})
        elif action_type == ActionType.SelectOption:
            # SELECT_OPTION, <select_node_id>, <option_node_id>[, <option_node_id>, ...]
            assert len(parts) == 2, ActionParseException(f"Invalid SELECT_OPTION format: {raw_action}")
            sub_parts = [p.strip() for p in parts[1].split(",")]

            option_node_ids = [int(id.strip("[]<>")) for id in sub_parts[1:]]
            target = find_target(sub_parts[0])
            if target.tag != "select":
                raise ActionParseException(
                    f"Target node {target.get_description(full=False)} is not a select node", extra=target
                )
            options = target.find_children_by_local_ids(option_node_ids)
            errors = []
            values = []
            for local_id, opt in zip(option_node_ids, options):
                if opt is None:
                    errors.append(f"Option node[{local_id}] not found")
                elif opt.tag != "option":
                    errors.append(f"The node[{local_id}] is not an option node")
                elif opt.attributes.get("value") is None:
                    errors.append(f"Option node[{local_id}] has no value attribute")
                else:
                    values.append(opt.attributes["value"])
            if errors:
                raise ActionParseException(f"SELECT_OPTION errors: {'; '.join(errors)}")
            return Action(uid, action_type, target, {"options": values})
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
        elif action_type == ActionType.WAIT:
            # WAIT, <seconds>
            assert len(parts) == 2, ActionParseException(f"Invalid WAIT format: {raw_action}")
            seconds = float(parts[1])
            return Action(uid, action_type, None, {"seconds": seconds})
        # elif action_type == ActionType.Memory:
        #     # MEMORY, <content>
        #     assert len(parts) == 2, ActionParseException(f"Invalid MEMORY format: {raw_action}")
        #     content = parts[1].replace("\\n", "\n")
        #     return Action(uid, action_type, None, {"content": content})
        # elif action_type == ActionType.ExtractAndMemory:
        #     # EXTRACT_AND_MEMORY, <subject>
        #     assert len(parts) == 2, ActionParseException(f"Invalid EXTRACT_AND_MEMORY format: {raw_action}")
        #     subject = parts[1]
        #     return Action(uid, action_type, None, {"subject": subject})
        else:
            raise ActionParseException(f"Unsupported action type: {action_type}")

    async def execute(self, tab: Page, tab_manager: "TabManager"):
        result = None
        if self.type in [ActionType.Click, ActionType.Input, ActionType.SelectOption]:
            # find target in tab
            assert self.target is not None
            frame = await self.target.find_owner_frame(tab)
            if frame is None:
                raise ActionExecuteException(f"Target {self.target.get_description(full=False)} owner frame not found")
            loc = await self.target.find_locator_in_frame(frame)
            if loc is None:
                raise ActionExecuteException(
                    f"Target {self.target.get_description(full=False)} locator not found in frame"
                )

            # execute click/input action
            if self.type == ActionType.Click:
                try:
                    await loc.click(timeout=5000, force=True)
                    if self.target.tag == "select":
                        result = f"Warning! Target node is a select element, CLICK did not work, you should only use SELECT_OPTION instead."
                except Exception as e:
                    raise ActionExecuteException(f"Failed to click: {e}")
            elif self.type == ActionType.Input:
                clear = self.extra["clear"]
                text = self.extra["text"]
                try:
                    if clear:
                        await loc.fill(text, timeout=5000)
                    else:
                        await loc.type(text, timeout=5000)
                    try:
                        await loc.press("Enter", timeout=1000)
                    except:
                        pass
                except Exception as e:
                    raise ActionExecuteException(f"Failed to click: {e}")
            elif self.type == ActionType.SelectOption:
                options: list[str] = self.extra["options"]
                try:
                    if len(options) == 1:
                        await loc.select_option(value=options[0], timeout=5000)
                    else:
                        await loc.select_option(value=options, timeout=5000)
                except Exception as e:
                    raise ActionExecuteException(f"Failed to select option: {e}")
        elif self.type == ActionType.Scroll:
            pages = self.extra["pages"]
            dy = Config.browser_viewport_h * pages
            direction = self.extra["direction"]
            if direction == "up":
                dy = -dy
            await tab.evaluate(f"window.scrollBy(0, {dy});")
            await asyncio.sleep(1)
        # elif self.type == ActionType.Press:
        #     key = self.extra["key"]
        #     await tab.keyboard.press(key)
        #     await asyncio.sleep(1)
        elif self.type == ActionType.Navigate:
            url = self.extra["url"]
            await tab.goto(url)
        elif self.type == ActionType.Search:
            keywords = self.extra["keywords"]
            await tab.goto(bing_search_url(keywords))
        elif self.type == ActionType.TabSwitch:
            tab_id = self.extra["tab_id"]
            if tab_id == tab_manager.cur_tab_id:
                raise ActionExecuteException(
                    f"Tab ID {tab_id} is already the current tab, you may want to scroll the tab page"
                )
            if tab_id not in tab_manager.tab_dict:
                raise ActionExecuteException(f"Tab ID {tab_id} not found in available tabs")
            tab = tab_manager.tab_dict[tab_id]
            tab_manager.cur_tab_id = tab_id
            await tab.bring_to_front()
        elif self.type == ActionType.TabClose:
            tab_id = self.extra["tab_id"]
            if tab_id not in tab_manager.tab_dict:
                raise ActionExecuteException(f"Tab ID {tab_id} not found")
            tab = tab_manager.tab_dict[tab_id]
            await tab.close()
        elif self.type == ActionType.WAIT:
            seconds = self.extra["seconds"]
            await asyncio.sleep(seconds)
        # elif self.type == ActionType.Memory:
        #     content = self.extra["content"]
        #     memory.append(content)
        # elif self.type == ActionType.ExtractAndMemory:
        #     if dom is None:
        #         raise ActionExecuteException("DOM is None, cannot extract info")
        #     subject = self.extra["subject"]
        #     text_content = dom.convert_to_repr_node(skip_omit=True).get_human_tree_repr()
        #     prompt = f"Extract the content strictly related to {subject} from the following HTML, ignoring HTML tags, and present it as a concise paragraph:\n{text_content}"
        #     llm_detail = await LLMs.llm_primary.chat_with_text_detail(prompt)
        #     memory.append(llm_detail["content"])
        #     return llm_detail
        else:
            raise NotImplementedError(f"Action type {self.type} is not implemented")
        return result


class ActionExecuteResult(TypedDict):
    success: bool
    result: None | str  # Other/Error
    tab_changed_info: str | None


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
