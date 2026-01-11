import asyncio
from enum import Enum
from typing import Any

from agent.config import Config
from agent.dom_utils import DomNode
from playwright.async_api import Page, CDPSession


class ActionType(Enum):
    # Click({backend_node_id})
    Click = "CLICK"
    # Input({backend_node_id, text, clear}) // clear should be a boolean value.
    Input = "INPUT"
    # Scroll({direction}) // Direction should be either up or down.
    Scroll = "SCROLL"
    # Press({key}) // The key should follow the KeyboardEvent.key specification and can be a single character or a supported function/modifier key, e.g. "a", "Enter", "Control+o".
    Press = "PRESS"


class ActionRecord:
    # TODO 记录action的操作和当时的DOM、截图等等
    pass


class Action:
    uid: str
    type: ActionType
    target: DomNode | None
    extra: Any

    def __init__(self, uid: str, action_type: ActionType, target: DomNode | None, extra: Any):
        self.uid = uid
        self.type = action_type
        self.target = target
        self.extra = extra

    @staticmethod
    def from_json(uid: str, json_obj: dict, dom_nodes: list[DomNode]) -> "Action":
        action_type = ActionType(json_obj["type"])

        target = None
        if action_type in [ActionType.Click, ActionType.Input]:
            target_id = json_obj["backend_node_id"]
            for node in dom_nodes:
                target = node.find_nodes_by_backend_node_ids([target_id])[0]
                if target is not None:
                    break
            assert target is not None

        if action_type == ActionType.Click:
            return Action(uid, action_type, target, None)
        elif action_type == ActionType.Input:
            clear = json_obj["clear"]
            if not isinstance(clear, bool):
                assert isinstance(clear, str)
                assert clear.lower() in ["true", "false"]
                clear = clear.lower() == "true"

            return Action(
                uid,
                action_type,
                target,
                {
                    "clear": clear,
                    "text": json_obj["text"],
                },
            )
        elif action_type == ActionType.Scroll:
            direction = json_obj["direction"]
            assert isinstance(direction, str)
            assert direction.lower() in ["up", "down"]
            direction = "down" if direction.lower() == "down" else "up"
            return Action(uid, action_type, target, direction)
        elif action_type == ActionType.Press:
            key = json_obj["key"]
            return Action(uid, action_type, target, key)

        raise NotImplementedError

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
            if self.extra == "up":
                dy = -dy
            await page.evaluate(f"window.scrollBy(0, {dy});")
        elif self.type == ActionType.Press:
            await page.keyboard.press(self.extra)
