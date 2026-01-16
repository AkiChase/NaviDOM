import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict

from agent import dom_utils
from agent.config import Config
from agent.dom_utils import DomNode
from playwright.async_api import Page, CDPSession

from agent.llm import ChatImageDetails, SecondaryLLM
from agent.pruning import Pruning
from agent.utils import SpecialException, google_search_url, page_screenshot, time_stamp


class ActionParseException(SpecialException):
    pass


class ActionExecuteException(SpecialException):
    pass


class ActionType(Enum):
    Click = "CLICK"
    Input = "INPUT"
    Scroll = "SCROLL"
    Press = "PRESS"
    Memory = "MEMORY"
    Extract = "EXTRACT"
    Navigate = "NAVIGATE"
    Search = "SEARCH"


action_format_prompt = {
    ActionType.Click: "CLICK, <backend_node_id>",
    ActionType.Input: "INPUT, <backend_node_id>, <clear:true|false>, <text>\t// In <text>, only \\n has special meaning for line breaks; no other escaping is required",
    ActionType.Scroll: "SCROLL, <up|down>\t// Move exactly one page (down +1, up -1)",
    ActionType.Press: "PRESS, <key>\t// Must follow KeyboardEvent.key (e.g. a, Enter, Control+o)",
    ActionType.Memory: "MEMORY, <label>, <content>\t// If the user request requires certain information to be output, store <content> under a semantic <label> using this action. After task completion, all stored MEMORY contents will be refined and returned to the user",
    ActionType.Extract: "EXTRACT, <label>, <instruction>\t// Extract information from the current UI page according to the natural-language instruction. The extracted result will be stored in MEMORY under the semantic <label>",
    ActionType.Navigate: "NAVIGATE, <url>\t// Navigate to the specified URL",
    ActionType.Search: "SEARCH, <keywords>\t// Navigate to Google search results for the given keywords",
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
            out.append(self.target.get_description())
        for k, v in self.extra.items():
            out.append(f"{k}={v}")
        return ", ".join(out)

    @staticmethod
    def get_format_prompt(types: list[ActionType] | None = None):
        if types is None:
            types = [
                ActionType.Click,
                ActionType.Input,
                ActionType.Scroll,
                ActionType.Press,
                ActionType.Memory,
                ActionType.Extract,
                ActionType.Navigate,
                ActionType.Search,
            ]

        return "\n".join([action_format_prompt[t] for t in types])

    @staticmethod
    def from_raw_action(uid: str, raw_action: str, dom_nodes: list[DomNode]) -> "Action":
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
            assert len(parts) == 2, f"Invalid CLICK format: {raw_action}"
            target = find_target(parts[1])
            return Action(uid, action_type, target, {})
        elif action_type == ActionType.Input:
            # INPUT, <backend_node_id>, <clear>, <text>
            assert len(parts) == 2, f"Invalid INPUT format: {raw_action}"
            sub_parts = [p.strip() for p in parts[1].split(",", maxsplit=2)]
            assert len(sub_parts) == 3, f"Invalid INPUT format: {raw_action}"
            clear_raw = sub_parts[1].lower()
            assert clear_raw in ("true", "false"), f"Invalid clear value: {clear_raw}"
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
            # SCROLL, <up|down>
            assert len(parts) == 2, f"Invalid SCROLL format: {raw_action}"
            direction = parts[1].lower()
            assert direction in ("up", "down"), f"Invalid scroll direction: {direction}"
            return Action(uid, action_type, None, {"direction": direction})
        elif action_type == ActionType.Press:
            # PRESS, <KeyboardEvent.key>
            assert len(parts) == 2, f"Invalid PRESS format: {raw_action}"
            key = parts[1]
            return Action(uid, action_type, None, {"key": key})
        elif action_type == ActionType.Memory:
            # MEMORY, <label>, <content>
            assert len(parts) == 2, f"Invalid MEMORY format: {raw_action}"
            sub_parts = [p.strip() for p in parts[1].split(",", maxsplit=1)]
            assert len(sub_parts) == 2, f"Invalid MEMORY format: {raw_action}"
            label = sub_parts[0]
            content = sub_parts[1]
            return Action(uid, action_type, None, {"label": label, "content": content})
        elif action_type == ActionType.Extract:
            # EXTRACT, <label>, <instruction>
            assert len(parts) == 2, f"Invalid EXTRACT format: {raw_action}"
            sub_parts = [p.strip() for p in parts[1].split(",", maxsplit=1)]
            assert len(sub_parts) == 2, f"Invalid EXTRACT format: {raw_action}"
            label = sub_parts[0]
            instruction = sub_parts[1]
            return Action(uid, action_type, None, {"label": label, "instruction": instruction})
        elif action_type == ActionType.Navigate:
            # NAVIGATE, <url>
            assert len(parts) == 2, f"Invalid NAVIGATE format: {raw_action}"
            url = parts[1]
            return Action(uid, action_type, None, {"url": url})
        elif action_type == ActionType.Search:
            # SEARCH, <keywords>
            assert len(parts) == 2, f"Invalid SEARCH format: {raw_action}"
            keywords = parts[1]
            return Action(uid, action_type, None, {"keywords": keywords})
        else:
            raise ActionParseException(f"Unsupported action type: {action_type}")

    async def execute(self, page: Page, cdp_session: CDPSession, memory: list[tuple[str, str]]):
        # 简单实现
        node = None
        if self.type in [ActionType.Click, ActionType.Input]:
            assert self.target is not None
            node = await cdp_session.send("DOM.resolveNode", {"backendNodeId": self.target.backend_node_id})
            if "object" not in node or "objectId" not in node["object"]:
                raise ActionExecuteException("Failed to get object ID for element")
            object_id = node["object"]["objectId"]

            if self.type == ActionType.Click:
                await cdp_session.send(
                    "Runtime.callFunctionOn",
                    {
                        "objectId": object_id,
                        "functionDeclaration": "function() {this.click();}",
                    },
                )
                await asyncio.sleep(1)
            elif self.type == ActionType.Input:
                # TODO 完全照抄吧，目前这样不会触发修改
                clear = self.extra["clear"]
                text = self.extra["text"]
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
                await asyncio.sleep(1)
        elif self.type == ActionType.Scroll:
            dy = Config.browser_viewport_h
            direction = self.extra["direction"]
            if direction == "up":
                dy = -dy
            await page.evaluate(f"window.scrollBy(0, {dy});")
            await asyncio.sleep(1)
        elif self.type == ActionType.Press:
            key = self.extra["key"]
            await page.keyboard.press(key)
            await asyncio.sleep(1)
        elif self.type == ActionType.Memory:
            label = self.extra["label"]
            content = self.extra["content"]
            memory.append((label, content))
        elif self.type == ActionType.Extract:
            instruction = self.extra["instruction"]
            dom = await cdp_session.send("DOM.getDocument", {"depth": -1})
            snapshot = await cdp_session.send(
                "DOMSnapshot.captureSnapshot",
                {
                    "computedStyles": ["display", "visibility", "opacity"],
                    "includeDOMRects": True,
                },
            )
            viewport = dom_utils.Viewport(await cdp_session.send("Page.getLayoutMetrics"))
            tree = dom_utils.parse_dom(dom, snapshot)
            Pruning.trim_dom_tree_by_visibility(tree, viewport)
            tree_text = Pruning.extract_dom_tree_text(tree)

            prompt = f"""
You are an AI assistant specialized in UI understanding and information extraction.
Your task is to extract only the information that is explicitly requested in the instruction from the UI page.

## Instruction
{instruction}

## UI Text
{tree_text}

## Requirements
- Use the UI text and the screenshot as information sources.
- Do NOT infer or assume information that is not directly observable.
- If the requested information is not present, output `NOT_FOUND`.
- Output only the extracted information. Do not include additional explanations.
            """.strip()
            screenshot = await page_screenshot(page)
            llm_details = await SecondaryLLM.chat_with_image_detail(prompt, screenshot)
            memory.append((self.extra["label"], llm_details["content"]))
            return llm_details
        elif self.type == ActionType.Navigate:
            url = self.extra["url"]
            await page.goto(url)
            await asyncio.sleep(1)
        elif self.type == ActionType.Search:
            keywords = self.extra["keywords"]
            await page.goto(google_search_url(keywords))
            await asyncio.sleep(1)
        else:
            raise NotImplementedError(f"Action type {self.type} is not implemented")


class ActionExecuteResult(TypedDict):
    success: bool
    additional: ChatImageDetails | None | str


@dataclass
class ActionDetails:
    action: Action
    raw_action: str
    execute_time: datetime
    execute_result: ActionExecuteResult
    action_screenshot_path: Path
    result_screenshot_path: Path

    def to_dict(self) -> dict:
        return {
            "raw_action": self.raw_action,
            "execute_time": time_stamp(self.execute_time),
            "execute_result": self.execute_result,
            "action_screenshot_path": str(self.action_screenshot_path),
            "result_screenshot_path": str(self.result_screenshot_path),
        }
