import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from agent.config import Config
from agent.dom_utils import DomNode
from playwright.async_api import Page, CDPSession

from agent.llm import ChatImageDetails, ChatTextDetails
from agent.utils import time_stamp


class ActionType(Enum):
    # Click({backend_node_id})
    Click = "CLICK"
    # Input({backend_node_id, text, clear}) // clear should be a boolean value.
    Input = "INPUT"
    # Scroll({direction}) // Direction should be either up or down.
    Scroll = "SCROLL"
    # Press({key}) // The key should follow the KeyboardEvent.key specification and can be a single character or a supported function/modifier key, e.g. "a", "Enter", "Control+o".
    Press = "PRESS"


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

    @staticmethod
    def get_format_prompt():
        prompt = [
            "CLICK, <backend_node_id>",
            "INPUT, <backend_node_id>, <clear:true|false>, <text>\t// In <text>, only \\n has special meaning for line breaks; no other escaping is required.",
            "SCROLL, <up|down>\t// Move exactly one page (down +1, up -1)",
            "PRESS, <key>\t// Must follow KeyboardEvent.key (e.g. a, Enter, Control+o)",
        ]
        return "\n".join(prompt)

    @staticmethod
    def from_raw_action(uid: str, csv_line: str, dom_nodes: list[DomNode]) -> "Action":
        csv_line = csv_line.strip()
        assert csv_line, "Empty action line"

        parts = [p.strip() for p in csv_line.split(",", maxsplit=1)]
        action_type = ActionType(parts[0].upper())

        def find_target(backend_node_id: str) -> DomNode:
            target_id = int(backend_node_id.strip("[]<>"))
            for node in dom_nodes:
                target = node.find_nodes_by_backend_node_ids([target_id])[0]
                if target is not None:
                    return target
            raise AssertionError(f"backend_node_id {target_id} not found in dom_nodes")

        if action_type == ActionType.Click:
            # CLICK, <backend_node_id>
            assert len(parts) == 2, f"Invalid CLICK format: {csv_line}"
            target = find_target(parts[1])
            return Action(uid, action_type, target, {})
        elif action_type == ActionType.Input:
            # INPUT, <backend_node_id>, <clear>, <text>
            assert len(parts) == 2, f"Invalid INPUT format: {csv_line}"
            sub_parts = [p.strip() for p in parts[1].split(",", maxsplit=2)]
            assert len(sub_parts) == 3, f"Invalid INPUT format: {csv_line}"
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
            assert len(parts) == 2, f"Invalid SCROLL format: {csv_line}"
            direction = parts[1].lower()
            assert direction in ("up", "down"), f"Invalid scroll direction: {direction}"
            return Action(uid, action_type, None, {"direction": direction})
        elif action_type == ActionType.Press:
            # PRESS, <KeyboardEvent.key>
            assert len(parts) == 2, f"Invalid PRESS format: {csv_line}"
            key = parts[1]
            return Action(uid, action_type, None, {"key": key})

        else:
            raise AssertionError(f"Unsupported action type: {action_type}")

    async def execute(self, page: Page, cdp_session: CDPSession) -> None:
        # 简单实现
        node = None
        if self.type in [ActionType.Click, ActionType.Input]:
            assert self.target is not None
            node = await cdp_session.send("DOM.resolveNode", {"backendNodeId": self.target.backend_node_id})
            if "object" not in node or "objectId" not in node["object"]:
                raise RuntimeError("Failed to get object ID for element")
            object_id = node["object"]["objectId"]

            if self.type == ActionType.Click:
                await cdp_session.send(
                    "Runtime.callFunctionOn",
                    {
                        "objectId": object_id,
                        "functionDeclaration": "function() {this.click();}",
                    },
                )
            elif self.type == ActionType.Input:
                clear = self.extra["clear"]
                text = self.extra["text"]
                await cdp_session.send("DOM.focus", {"objectId": object_id})

                value_stmt = f'this.value = "{text}";' if clear else f'this.value += "{text}";'
                await cdp_session.send(
                    "Runtime.callFunctionOn",
                    {
                        "objectId": object_id,
                        "functionDeclaration": """function() {
                        try { this.select(); } catch (e) {}
                        <Mask>
                        this.dispatchEvent(new Event("input", { bubbles: true }));
                        this.dispatchEvent(new Event("change", { bubbles: true }));
                    }""".replace(
                            "<Mask>", value_stmt
                        ),
                    },
                )
        elif self.type == ActionType.Scroll:
            dy = Config.browser_viewport_h
            direction = self.extra["direction"]
            if direction == "up":
                dy = -dy
            await page.evaluate(f"window.scrollBy(0, {dy});")
        elif self.type == ActionType.Press:
            key = self.extra["key"]
            await page.keyboard.press(key)


@dataclass
class ActionDetails:
    action: Action
    raw_action: str
    execute_time: datetime
    action_screenshot_path: Path
    result_screenshot_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "raw_action": self.raw_action,
            "execute_time": time_stamp(self.execute_time),
            "action_screenshot_path": str(self.action_screenshot_path),
            "result_screenshot_path": str(self.result_screenshot_path),
        }
