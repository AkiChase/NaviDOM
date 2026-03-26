from collections import defaultdict, deque
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
from typing import Any
import numpy as np
from playwright.async_api import Page, Locator, Frame
from PIL import ImageDraw, ImageFont, Image
from loguru import logger
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from agent.llm import LLMs



class Viewport:
    width: float
    height: float
    scroll_y: float
    scroll_x: float
    scroll_h: float
    scroll_w: float

    @staticmethod
    async def from_tab(tab: Page):
        return Viewport(
            await tab.evaluate(
                """() => {
                    return {
                        "width": window.innerWidth,
                        "height": window.innerHeight,
                        "scrollY": window.scrollY,
                        "scrollX": window.scrollX,
                        "scrollH": document.body.scrollHeight,
                        "scrollW": document.body.scrollWidth,
                    }
                }"""
            )
        )

    def __init__(self, viewport: dict[str, float]):
        self.width = viewport["width"]
        self.height = viewport["height"]
        self.scroll_y = viewport["scrollY"]
        self.scroll_x = viewport["scrollX"]
        self.scroll_h = viewport["scrollH"]
        self.scroll_w = viewport["scrollW"]

    @property
    def remaining_up_pages(self) -> float:
        return max(self.scroll_y / self.height, 0.0)

    @property
    def remaining_down_pages(self) -> float:
        remaining_height = self.scroll_h - (self.scroll_y + self.height)
        return max(remaining_height / self.height, 0.0)

    def get_viewport_scroll_info(self) -> str:
        if self.remaining_up_pages >= 0.1:
            up_info = f"can scroll up 0.1 to {self.remaining_up_pages:.1f} pages"
        else:
            up_info = "cannot scroll up"
        if self.remaining_down_pages >= 0.1:
            down_info = f"can scroll down 0.1 to {self.remaining_down_pages:.1f} pages"
        else:
            down_info = "cannot scroll down"

        return f"Current Visible Tab scroll info: {up_info}, {down_info}"


default_font = ImageFont.load_default()


@dataclass
class Bounds:
    x: float
    y: float
    width: float
    height: float

    def __repr__(self):
        return f"Bounds(x={self.x:.1f}, y={self.y:.1f}, width={self.width:.1f}, height={self.height:.1f})"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, (Bounds, IframeBounds)):
            return False

        return (
            abs(self.x - value.x) < 0.2
            and abs(self.y - value.y) < 0.2
            and abs(self.width - value.width) < 0.2
            and abs(self.height - value.height) < 0.2
        )

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def to_xyxy(self):
        return self.x, self.y, self.x + self.width, self.y + self.height

    def to_xy(self, offset: tuple[float, float] | None = None):
        if offset is None:
            offset = (0, 0)
        return self.x + offset[0], self.y + offset[1]


@dataclass
class IframeBounds(Bounds):
    offset_x: float
    offset_y: float

    def __repr__(self):
        return f"IframeBounds(x={self.x:.1f}, y={self.y:.1f}, width={self.width:.1f}, height={self.height:.1f}, offset_x={self.offset_x:.1f}, offset_y={self.offset_y:.1f})"

    def __eq__(self, value: object) -> bool:
        return super().__eq__(value)


class DomNode:
    frame_node_id: int
    local_id: int
    level: int
    index: int | None
    tag: str
    text: str
    attributes: dict[str, str]
    children: list["DomNode"]
    parent: "DomNode | None"
    bounds: Bounds | IframeBounds | None
    is_covered: bool
    selector: str | None

    @staticmethod
    async def from_raw(raw: dict[str, Any], start_local_id: int = 0) -> list["DomNode"]:

        node_list = []
        cur_id = start_local_id

        def recur(node: DomNode, raw_node: dict[str, Any], parent: DomNode | None):
            nonlocal cur_id
            node.local_id = cur_id
            cur_id += 1
            node_list.append(node)

            node.index = raw_node["index"]
            node.parent = parent
            node.level = node.parent.level + 1 if node.parent is not None else 0
            node.tag = raw_node["tag"]
            node.text = raw_node["text"]
            node.attributes = raw_node["attrs"]
            node.selector = raw_node["selector"]
            if raw_node["bounds"]:
                bounds_params = {
                    "x": raw_node["bounds"]["x"],
                    "y": raw_node["bounds"]["y"],
                    "width": raw_node["bounds"]["width"],
                    "height": raw_node["bounds"]["height"],
                }
                node.is_covered = raw_node["bounds"]["isCovered"]

                if node.tag == "iframe":
                    node.bounds = IframeBounds(
                        **bounds_params,
                        offset_x=raw_node["bounds"]["offsetX"],
                        offset_y=raw_node["bounds"]["offsetY"],
                    )
                else:
                    node.bounds = Bounds(**bounds_params)
            else:
                node.bounds = None
                node.is_covered = True

            node.children = []
            for raw_child in raw_node["children"]:
                child_node = DomNode()
                node.children.append(child_node)
                recur(child_node, raw_child, node)

        root_node = DomNode()
        recur(root_node, raw, None)

        return node_list

    def __repr__(self):
        return f"<DomNode tag={self.tag} id={self.local_id} text={self.text}>"

    @property
    def editable(self) -> bool:
        if self.tag in ["input", "textarea"]:
            return True
        if self.attributes.get("contenteditable") == "true":
            return True
        return False

    @property
    def clickable(self) -> bool:
        if self.tag == "iframe":
            return False
        return True

    def max_bounds(self) -> Bounds | None:
        bounds_list: list[Bounds] = []

        def collect(node: "DomNode"):
            if node.bounds is not None:
                bounds_list.append(node.bounds)

            for child in node.children:
                collect(child)

        collect(self)

        if not bounds_list:
            return None

        x1 = min(b.x for b in bounds_list)
        y1 = min(b.y for b in bounds_list)
        x2 = max(b.x + b.width for b in bounds_list)
        y2 = max(b.y + b.height for b in bounds_list)

        return Bounds(x1, y1, x2 - x1, y2 - y1)

    def draw_bounds(
        self,
        draw: ImageDraw.ImageDraw,
        outline: str = "red",
        width=1,
        draw_id=False,
        recursive=True,
        max_bounds=False,
    ):
        bounds = self.max_bounds() if max_bounds else self.bounds
        if bounds:
            draw.rectangle(bounds.to_xyxy(), outline=outline, width=width)
            if draw_id:
                draw.text((bounds.to_xy((width, width))), str(self.local_id), fill=outline, font=default_font)

        if recursive:
            for child in getattr(self, "children", []):
                child.draw_bounds(
                    draw=draw,
                    outline=outline,
                    width=width,
                    draw_id=draw_id,
                    recursive=recursive,
                    max_bounds=max_bounds,
                )

    def find_children_by_local_ids(self, target_ids: list[int]) -> list["DomNode | None"]:
        node_map = {}
        stack: list["DomNode"] = [self]
        target_set = set(target_ids)

        while stack and target_set:
            node = stack.pop()
            if node.local_id in target_set:
                node_map[node.local_id] = node
                target_set.remove(node.local_id)

            stack.extend(reversed(node.children))

        return [node_map.get(i) for i in target_ids]

    def get_description(self, full=True) -> str:

        def collect_text(node: "DomNode") -> list[str]:
            out: list[str] = []
            if node.tag == "":
                if node.text:
                    if full or len(node.text) <= 20:
                        out.append(node.text)
                    else:
                        out.append(node.text[:20] + "...")

            for child in node.children:
                out.extend(collect_text(child))
            return out

        text = collect_text(self)
        content = f'({" ".join(text)})' if text else ""
        return f"{self.tag}{content}"

    def get_simple_selector(self, max_depth: int = 8) -> str:
        # 结构型 selector：tag + nth-child
        cur = self
        parts = []
        while cur is not None and cur.tag != "iframe" and len(parts) < max_depth:
            if cur.index is not None:
                parts.append(f"{cur.tag}:nth-child({cur.index})")
            else:
                parts.append(cur.tag)
            cur = cur.parent
        selector = " > ".join(reversed(parts))
        return selector

    async def find_owner_frame(self, tab: Page) -> Frame | None:
        if self.frame_node_id == -1:
            return tab.main_frame

        frame_node_list = []
        cur = self.parent
        while cur is not None:
            if cur.tag == "iframe":
                frame_node_list.append(cur)
            cur = cur.parent

        cur_frame = tab.main_frame
        for frame_node in reversed(frame_node_list):
            locator = await frame_node.find_locator_in_frame(cur_frame)
            if locator is None:
                logger.warning(f"iframe {frame_node.local_id} locator not found")
                return None
            cur_frame = await (await locator.element_handle(timeout=100)).content_frame()
            if cur_frame is None:
                logger.warning(f"iframe {frame_node.local_id} content frame not found")
                return None

        return cur_frame

    async def find_locator_in_frame(self, frame: Frame) -> Locator | None:
        # no search in sub-iframe
        if self.selector is not None:
            locator = frame.locator(self.selector)
            if await locator.count() == 1:
                return locator

        if len(self.children) == 1 and self.children[0].tag == "":
            locator = frame.get_by_text(text=self.children[0].text, exact=True)
            if await locator.count() == 1:
                return locator

        locator = frame.locator(self.get_simple_selector())
        if await locator.count() == 1:
            return locator
        return None

    _ancestor_set: set["DomNode"] | None = None

    def get_ancestor_set(self) -> set["DomNode"]:
        if self._ancestor_set is None:
            s = set()
            node = self
            while node:
                s.add(node)
                node = node.parent
            self._ancestor_set = s
            return s
        else:
            return self._ancestor_set

    def convert_to_repr_node(self, skip_filter=False, skip_promote=False, skip_omit=False) -> "ReprDomNode":
        return ReprDomNode.from_dom_tree(self, skip_filter, skip_promote, skip_omit)

    @staticmethod
    def extract_interactive_nodes(root: "DomNode") -> list["DomNode"]:
        def dfs(node: DomNode):
            if node.tag in FORCE_REMOVE_ELEMENTS:
                return []

            output = []
            for child in node.children:
                output.extend(dfs(child))

            if node.bounds is not None and (node.tag == "" or node.tag in SUPPORT_ELEMENTS or len(node.children) > 0):
                if not output:
                    output.append(node)
                else:
                    if node.tag in GROUP_ELEMENTS or node.attributes.get("role") in GROUP_ELEMENT_ROLES:
                        output = [node]

            return output

        return dfs(root)


# Interactive Elements
GROUP_ELEMENTS = {"a", "button", "input", "textarea", "select", "checkbox", "dialog"}
GROUP_ELEMENT_ROLES = {"button", "checkbox", "radio", "switch", "textbox", "combobox", "listbox", "dialog"}
FORCE_REMOVE_ELEMENTS = {"script", "style", "meta", "link", "title"}
SUPPORT_ELEMENTS = {
    # ========= 文档 / 布局语义 =========
    "html",
    "body",
    "main",
    "header",
    "footer",
    "nav",
    "section",
    "article",
    "aside",
    # ========= 通用容器 =========
    "div",
    "span",
    # ========= 标题 =========
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    # ========= 文本 / 内容 =========
    "p",
    "label",
    "strong",
    "em",
    "b",
    "i",
    "small",
    "mark",
    "code",
    "pre",
    "blockquote",
    # ========= 列表 =========
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    # ========= 链接与媒体 =========
    "a",
    "img",
    "svg",
    "canvas",
    "video",
    "audio",
    "iframe",
    # ========= 表单（交互关键） =========
    "form",
    "input",
    "textarea",
    "button",
    "select",
    "option",
    "fieldset",
    "legend",
    # ========= 表格（常见信息密集区） =========
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "th",
    "td",
    "caption",
    # ========= HTML5 交互 / 状态 =========
    "details",
    "summary",
    "dialog",
    "progress",
    "meter",
}

IMPORTANT_ATTRS = {
    "id",
    "class",
    "src",
    "href",
    "alt",
    "role",
    "title",
    "for",
    "disabled",
    "readonly",
    "required",
    "placeholder",
    "value",
    "type",
}


class ReprDomNode:
    original_node: DomNode
    level: int
    local_id: int
    tag: str
    repr_text: str
    repr_attributes: dict[str, str]
    children: list["ReprDomNode"]
    parent: "ReprDomNode | None"

    @staticmethod
    def from_dom_tree(root: DomNode, skip_filter=False, skip_promote=False, skip_omit=False) -> "ReprDomNode":

        # build repr dom tree
        def build_recur(node: DomNode, parent: "ReprDomNode | None"):
            if node.tag not in FORCE_REMOVE_ELEMENTS:
                repr_node = ReprDomNode()
                repr_node.original_node = node
                repr_node.level = node.level
                repr_node.local_id = node.local_id
                repr_node.tag = node.tag
                repr_node.parent = parent
                repr_node.children = [build_recur(child, repr_node) for child in node.children]
            return repr_node

        # filter not support elements
        def filter_recur(node: ReprDomNode):
            retained_children = []
            for child in node.children:
                filter_recur(child)
                if child.tag == "" or child.tag in SUPPORT_ELEMENTS or len(child.children) > 0:
                    retained_children.append(child)
            node.children = retained_children

        # promote node
        def promote_recur(node: ReprDomNode):

            def is_promotable(node: ReprDomNode):
                if node.parent is None or node.tag == "":
                    return False
                if node.parent.tag in GROUP_ELEMENTS:
                    return False
                if (
                    node.original_node.bounds is not None
                    and node.original_node.bounds == node.parent.original_node.bounds
                ):
                    return True
                return False

            def update_level(node: ReprDomNode, base_level: int):
                node.level = base_level
                for c in node.children:
                    update_level(c, base_level + 1)

            def promote_self(node: ReprDomNode):
                if node.parent is None:
                    return
                parent = node.parent
                grandparent = parent.parent
                if grandparent is None:
                    return  # 不提升根节点

                # 把 node及兄弟节点 插入到 grandparent下，替换 parent
                idx = grandparent.children.index(parent)
                siblings = list(parent.children)  # 包括 node 和它的兄弟
                grandparent.children[idx : idx + 1] = siblings
                for sib in siblings:
                    sib.parent = grandparent
                    update_level(sib, parent.level)
                parent.children.clear()

            if is_promotable(node):
                promote_self(node)

            for child in list(node.children):
                promote_recur(child)

        # convert attributes
        def omit_recur(node: ReprDomNode, skip_omit):
            node.repr_attributes = {}
            if skip_omit:
                node.repr_text = node.original_node.text
            else:
                node.repr_text = (
                    node.original_node.text[:100] + "..."
                    if len(node.original_node.text) > 100
                    else node.original_node.text
                )
            node.repr_attributes = {}
            for name, value in node.original_node.attributes.items():
                if name in IMPORTANT_ATTRS or name.startswith("aria-"):
                    value = value.strip()
                    if not skip_omit:
                        if name == "class":
                            value = " ".join(value.split()[:2])
                        elif len(value) > 20:
                            value = value[:20] + "..."
                    node.repr_attributes[name] = value

            for child in node.children:
                omit_recur(child, skip_omit)

        repr_root = build_recur(root, None)
        if not skip_filter:
            filter_recur(repr_root)
        if not skip_promote:
            promote_recur(repr_root)

        omit_recur(repr_root, skip_omit)
        return repr_root

    def _get_tree_repr(self, indent: int = 0, no_end=False, no_id=False):
        out = []
        prefix = "\t" * indent
        attr_str = ""
        # attributes
        if self.repr_attributes:
            attr_pairs = [f'{k}="{v}"' if v else k for k, v in self.repr_attributes.items()]
            attr_str = " " + " ".join(attr_pairs)
        # if self.original_node.is_covered:
        #     attr_str = f" covered{attr_str}"

        # leaf text
        if not self.children:
            if self.repr_text:
                out.append(f"{prefix}{self.repr_text}")
            else:
                out.append(f"{prefix}[{self.local_id}]<{self.tag}{attr_str} />")
            return out
        # all text child → inline
        if all(c.tag == "" for c in self.children):
            children_text = "".join([c.repr_text for c in self.children])
            end_text = "" if no_end else f"</{self.tag}>"
            out.append(f"{prefix}[{self.local_id}]<{self.tag}{attr_str}>{children_text}{end_text}")
            return out
        # normal expanded node
        out.append(f"{prefix}[{self.local_id}]<{self.tag}{attr_str}>")
        for child in self.children:
            # text node → append to current line
            if child.tag == "" and child.repr_text:
                out[-1] += child.repr_text
            else:
                out.extend(child._get_tree_repr(indent + 1, no_end=no_end, no_id=no_id))
        if not no_end:
            out.append(f"{prefix}{' ' * len(f'[{self.local_id}]')}</{self.tag}>")

        return out

    def get_human_tree_repr(self, indent: int = 0, no_end=False, no_id=False) -> str:
        return "\n".join(self._get_tree_repr(indent=indent, no_end=no_end, no_id=no_id))


@dataclass
class DomState:
    viewport: Viewport
    dom: DomNode

    @staticmethod
    async def load_dom_state(tab: Page) -> "DomState":
        load_script = (Path(__file__).parent / "scripts/loadDomState.js").read_text(encoding="utf-8")
        res = await tab.evaluate(load_script)
        viewport = Viewport(res["viewport"])
        dom_node_list = await DomNode.from_raw(res["dom"])
        dom_root = dom_node_list[0]
        next_local_id = dom_node_list[-1].local_id + 1

        async def recur(root: DomNode, parent_frame: Frame, parent_frame_id: int):
            nonlocal next_local_id

            root.frame_node_id = parent_frame_id
            if root.tag == "iframe":
                frame_loc = await root.find_locator_in_frame(parent_frame)
                if frame_loc is None:
                    logger.warning(f"iframe {root.local_id} locator not found")
                else:
                    target_frame = await (await frame_loc.element_handle(timeout=100)).content_frame()
                    if target_frame is None:
                        logger.warning(f"iframe {root.local_id} content frame not found")
                    else:
                        if not isinstance(root.bounds, IframeBounds):
                            logger.warning(f"iframe node {root.local_id} bounds is not IframeBounds")
                        else:
                            frame_viewport = {
                                "offsetX": root.bounds.x + root.bounds.offset_x,
                                "offsetY": root.bounds.y + root.bounds.offset_y,
                                "left": max(0, root.bounds.x + root.bounds.offset_x),
                                "top": max(0, root.bounds.y + root.bounds.offset_y),
                                "bottom": min(
                                    viewport.height, root.bounds.y + root.bounds.height + root.bounds.offset_y
                                ),
                                "right": min(viewport.width, root.bounds.x + root.bounds.width + root.bounds.offset_x),
                            }
                            new_res = await target_frame.evaluate(load_script, frame_viewport)
                            if new_res["dom"] is not None:
                                new_dom_node_list = await DomNode.from_raw(
                                    new_res["dom"],
                                    start_local_id=next_local_id,
                                )
                                next_local_id = new_dom_node_list[-1].local_id + 1
                                new_dom_root = new_dom_node_list[0]
                                root.children = [new_dom_root]
                                new_dom_root.parent = root
                                await recur(new_dom_root, target_frame, root.local_id)
            else:
                for child in root.children:
                    await recur(child, parent_frame, parent_frame_id)

        await recur(dom_root, tab.main_frame, -1)

        return DomState(viewport, dom_root)


class DomCluster:
    @staticmethod
    def cluster_construct(
        nodes: list[DomNode], alpha: float = 0.5, max_clusters: int = 5
    ) -> list[list[DomNode]]:
        """
        - nodes: 可交互节点列表
        - alpha: 空间距离权重 (0~1) DOM树距离权重 (1-alpha)
        - max_clusters: 最大聚类数
        """

        n = len(nodes)
        if not n:
            return []
        if n == 1:
            return [nodes]
        centers = []
        for node in nodes:
            if node.bounds is not None:
                centers.append(node.bounds.center)
            else:
                centers.append((0.0, 0.0))
        centers = np.array(centers)
        # 空间距离 (欧氏距离)
        spatial_dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        # 归一化
        if spatial_dist.max() > 0:
            spatial_dist /= spatial_dist.max()

        # DOM树距离
        def dom_distance(a: DomNode, b: DomNode) -> int:
            set_a = a.get_ancestor_set()
            set_b = b.get_ancestor_set()

            common_nodes = set_a & set_b
            assert common_nodes
            lca = max(common_nodes, key=lambda t: t.level)
            return a.level + b.level - 2 * lca.level

        dom_dist = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = dom_distance(nodes[i], nodes[j])
                dom_dist[i, j] = dom_dist[j, i] = d
        if dom_dist.max() > 0:
            dom_dist /= dom_dist.max()
        # 综合距离矩阵
        combined_dist = alpha * spatial_dist + (1 - alpha) * dom_dist
        np.fill_diagonal(combined_dist, 0.0)
        combined_dist = np.maximum(combined_dist, 0.0)
        # 层次聚类
        z = linkage(squareform(combined_dist), method="average")
        labels: np.ndarray = fcluster(z, t=max_clusters, criterion="maxclust")

        clusters: defaultdict[int, list[DomNode]] = defaultdict(list)
        for node, label in zip(nodes, labels):
            clusters[label].append(node)

        return list(clusters.values())

    @staticmethod
    async def determine_cluster_related(nodes: list[DomNode], image: Image.Image, task: str):
        prompt = f"""
You are an experienced UI automation testing developer.
You need to determine whether the UI elements in the current region screenshot could be related to "{task}".
You should make this judgment based on the DOM elements list provided below and the corresponding UI screenshot (the relevant elements are highlighted with red boxes).
If there is any potential relationship, output Yes; if they are completely unrelated, output No. Do not output any other text.

DOM elements:
{'\n'.join([node.convert_to_repr_node().get_human_tree_repr(no_end=True, no_id=True) for node in nodes])}
    """.strip()

        # resized_image = dom_image.copy().resize((round(dom_image.size[0] * 0.75), round(dom_image.size[1] * 0.75)))
        res = await LLMs.vlm_secondary.chat_with_image_detail(
            prompt=prompt,
            image=image,
        )

        content: str = res["content"].strip()
        assert content.lower() in {"yes", "no"}
        flag = content.lower() == "yes"
        return flag, res

    @staticmethod
    def cluster_covered_xyxy(nodes: list[DomNode], viewport: Viewport) -> tuple[float, float, float, float]:
        max_bounds = [node.max_bounds() for node in nodes]
        valid_bounds = [b.to_xyxy() for b in max_bounds if b is not None]
        assert valid_bounds, "cluster contains no bounds"

        x1 = min(b[0] for b in valid_bounds)
        y1 = min(b[1] for b in valid_bounds)
        x2 = max(b[2] for b in valid_bounds)
        y2 = max(b[3] for b in valid_bounds)
        # ensure within viewport
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, viewport.width)
        y2 = min(y2, viewport.height)

        return x1, y1, x2, y2

    @staticmethod
    def cluster_merge_overlapped(
        cluster_rects: list[tuple[list[DomNode], tuple[int, int, int, int]]],
    ) -> list[tuple[list[DomNode], tuple[int, int, int, int]]]:
        merged = []

        def overlaps(r1, r2):
            _, (x11, y11, x12, y12) = r1
            _, (x21, y21, x22, y22) = r2
            return not (x12 < x21 or x22 < x11 or y12 < y21 or y22 < y11)

        def union(r1, r2):
            n1, (x11, y11, x12, y12) = r1
            n2, (x21, y21, x22, y22) = r2
            return n1 + n2, (min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22))

        rects = cluster_rects.copy()
        while rects:
            current = rects.pop(0)
            i = 0
            while i < len(rects):
                if overlaps(current, rects[i]):
                    current = union(current, rects.pop(i))
                    i = 0  # 重新检查所有矩形，因为union后的current可能与其他矩形重叠
                else:
                    i += 1
            merged.append(current)
        return merged

    @staticmethod
    def cluster_image_layout_compaction(
        screenshot: Image.Image,
        cluster_rects: list[tuple[list[DomNode], tuple[int, int, int, int]]],
        default_gap=5,
    ):

        def build_constraints(rects: list[dict], axis="x"):
            # axis: 'x' 或 'y'
            constraints = []  # list of (before_id, after_id)
            # 两两排列构建顺序规则
            for a, b in itertools.combinations(rects, 2):
                if axis == "x":
                    a1, a2 = a["orig"][0], a["orig"][2]  # x1, x2
                    b1, b2 = b["orig"][0], b["orig"][2]
                else:
                    a1, a2 = a["orig"][1], a["orig"][3]  # y1, y2
                    b1, b2 = b["orig"][1], b["orig"][3]

                # 判断是否在该轴上分离（无重叠）
                if a2 <= b1:
                    constraints.append((a["id"], b["id"]))  # a 在 b 之前
                elif b2 <= a1:
                    constraints.append((b["id"], a["id"]))  # b 在 a 之前
            return constraints

        def layout_1d(rects: list[dict], constraints: list[tuple[int, int]], axis="x", default_gap=5) -> dict[int, int]:
            # rects: list of rect with 'id', 'size'
            # constraints: list of (before, after)
            # return: start_pos in dict with 'id' as key

            size_key = 0 if axis == "x" else 1  # width or height

            # 构建图和入度
            graph = defaultdict(list)
            in_degree = {r["id"]: 0 for r in rects}
            for u, v in constraints:
                graph[u].append(v)
                in_degree[v] += 1

            # 拓扑排序（Kahn 算法）
            queue = deque([node for node in in_degree if in_degree[node] == 0])
            order = []
            while queue:
                u = queue.popleft()
                order.append(u)
                for v in graph[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

            if len(order) != len(rects):
                raise RuntimeError("Cycle detected in constraints!")

            # 贪心布局：从左到右（或从上到下），满足约束
            pos = {}
            end_pos = {}  # 记录每个节点的结束位置（用于后续约束）

            for r in rects:
                pos[r["id"]] = 0
                end_pos[r["id"]] = 0

            # 按拓扑序放置，并加上默认间距
            for node_id in order:
                # 找出所有前驱的最大结束位置
                min_start = 0
                for u, v in constraints:
                    if v == node_id:
                        min_start = round(max(min_start, end_pos[u] + default_gap))  # 添加默认间距

                # 放置当前节点
                pos[node_id] = min_start
                w_or_h = round(next(r["size"][size_key] for r in rects if r["id"] == node_id))
                end_pos[node_id] = min_start + w_or_h

            return pos

        crops = []
        for cluster, rect in cluster_rects:
            crop = screenshot.crop(rect)
            draw = ImageDraw.Draw(crop)
            for node in cluster:
                node_bounds = node.max_bounds()
                if node_bounds is None:
                    continue
                node_bounds = node_bounds.to_xyxy()
                node_bounds = (
                    node_bounds[0] - rect[0],
                    node_bounds[1] - rect[1],
                    node_bounds[2] - rect[0],
                    node_bounds[3] - rect[1],
                )
                draw.rectangle(node_bounds, outline="red", width=1)
                # draw.text((node_bounds[0] + 1, node_bounds[1] + 1), text=f"{node.backend_node_id}", fill="red", font=font)
            crops.append(crop)

        # 构建 rects 结构
        rect_data = []
        for i, (_, (x1, y1, x2, y2)) in enumerate(cluster_rects):
            rect_data.append({"id": i, "orig": (x1, y1, x2, y2), "size": (x2 - x1, y2 - y1)})

        # 构建约束
        x_constraints = build_constraints(rect_data, "x")
        y_constraints = build_constraints(rect_data, "y")

        # 布局
        x_pos = layout_1d(rect_data, x_constraints, "x", default_gap)
        y_pos = layout_1d(rect_data, y_constraints, "y", default_gap)

        # 计算画布大小
        max_w = math.ceil(max(x_pos[r["id"]] + r["size"][0] for r in rect_data))
        max_h = math.ceil(max(y_pos[r["id"]] + r["size"][1] for r in rect_data))

        # 创建新图像
        output_img = Image.new("RGB", (max_w, max_h), (0, 0, 0, 0))
        for r in rect_data:
            idx = r["id"]
            output_img.paste(crops[idx], (x_pos[idx], y_pos[idx]))

        return output_img
