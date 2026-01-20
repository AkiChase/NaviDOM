import itertools
import math
from collections import defaultdict, deque
from enum import IntEnum
from typing import Any, Optional
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from playwright.async_api import Page, CDPSession

import tiktoken
from PIL import ImageFont, ImageDraw, Image

from agent.utils import css_escape

tiktoken_enc = tiktoken.get_encoding("o200k_base")
default_font = ImageFont.load_default()


class NodeType(IntEnum):
    ELEMENT_NODE = 1  # <div>, <button>, <span> ...
    ATTRIBUTE_NODE = 2  # class="a" (rarely exposed)
    TEXT_NODE = 3  # #text
    CDATA_SECTION_NODE = 4  # <![CDATA[ ... ]]>
    PROCESSING_INSTRUCTION_NODE = 7  # <?xml ... ?>
    COMMENT_NODE = 8  # <!-- comment -->
    DOCUMENT_NODE = 9  # document
    DOCUMENT_TYPE_NODE = 10  # <!DOCTYPE html>
    DOCUMENT_FRAGMENT_NODE = 11  # ShadowRoot
    NOTATION_NODE = 12  # deprecated


class Viewport:
    page_x: float
    page_y: float
    client_w: float
    client_h: float
    total_height: float
    total_width: float

    def __init__(self, viewport: dict[str, Any]):
        layout = viewport["cssLayoutViewport"]
        content = viewport["cssContentSize"]

        self.page_x = float(layout["pageX"])
        self.page_y = float(layout["pageY"])
        self.client_w = float(layout["clientWidth"])
        self.client_h = float(layout["clientHeight"])

        self.total_height = float(content["height"])
        self.total_width = float(content["width"])

    @staticmethod
    async def from_cdp_session(cdp_session: CDPSession):
        return Viewport(await cdp_session.send("Page.getLayoutMetrics"))

    @property
    def remaining_up_pages(self) -> float:
        return max(self.page_y / self.client_h, 0.0)

    @property
    def remaining_down_pages(self) -> float:
        remaining_height = self.total_height - (self.page_y + self.client_h)
        return max(remaining_height / self.client_h, 0.0)

    def get_tab_page_info(self) -> str:
        up = self.remaining_up_pages
        down = self.remaining_down_pages
        parts: list[str] = []
        if up > 0:
            parts.append(f"Scrollable up: {up:.1f} pages")
        if down > 0:
            parts.append(f"Scrollable down: {down:.1f} pages")

        return ", ".join(parts)


class DomNode:
    backend_node_id: int
    node_type: NodeType
    local_name: str
    node_value: str
    repr_value: str
    attributes: dict[str, str]
    repr_attributes: dict[str, str]
    children: list["DomNode"]
    parent: "DomNode | None"
    bounds: tuple[float, float, float, float] | None  # (x1, y1, x2, y2)
    style_visible: bool
    level: int  # depth

    def __init__(
        self,
        node: dict[str, Any],
        parent: "DomNode | None",
        bounds_dict: dict[int, tuple[float, float, float, float]],
        visible_dict: dict[int, bool],
    ):
        self.backend_node_id = node.get("backendNodeId", -1)
        self.node_type = NodeType(node["nodeType"])
        self.local_name = node.get("localName", "")
        self.node_value = node.get("nodeValue", "").strip()
        self.repr_value = self.node_value

        attributes = node.get("attributes", [])
        self.attributes = {}
        for i in range(0, len(attributes), 2):
            self.attributes[attributes[i]] = attributes[i + 1]
        self.repr_attributes = self.attributes.copy()

        if self.backend_node_id in bounds_dict:
            x, y, w, h = bounds_dict[self.backend_node_id]
            self.bounds = (x, y, x + w, y + h)
        else:
            self.bounds = None

        if self.backend_node_id in visible_dict:
            self.style_visible = visible_dict[self.backend_node_id]
        else:
            self.style_visible = False

        self.children: list[DomNode] = []
        self.parent = parent
        if self.parent is None:
            self.level = 0
        else:
            self.level = self.parent.level + 1

        for child in node.get("children", []):
            self.children.append(DomNode(child, self, bounds_dict=bounds_dict, visible_dict=visible_dict))

    def __repr__(self):
        return f"<DomNode {self.local_name} backend_node_id={self.backend_node_id}>"

    @property
    def editable(self) -> bool:
        if self.local_name in ["input", "textarea"]:
            return True
        if self.attributes.get("contenteditable") == "true":
            return True
        return False

    def _build_selector_candidates(self) -> list[str]:
        selectors = []

        attrs = self.attributes or {}
        tag = self.local_name

        base_selector = tag
        if "id" in attrs:
            base_selector += f'#{css_escape(attrs["id"])}'

        # data-xxx
        for key in attrs:
            if key.startswith("data-"):
                selectors.append(f'{base_selector}[{key}="{css_escape(attrs[key])}"]')
        # name
        if "name" in attrs:
            selectors.append(f'{base_selector}[name="{css_escape(attrs["name"])}"]')
        # class
        if "class" in attrs:
            classes = attrs["class"].split()
            selectors.append(base_selector + "".join(f".{css_escape(c)}" for c in classes[:3]))
        return selectors

    async def find_node_in_tab(self, tab: Page):
        if len(self.children) == 1 and self.children[0].node_type == NodeType.TEXT_NODE:
            target_text = self.children[0].node_value.strip()
            if target_text:
                loc = tab.get_by_text(target_text, exact=True)
                if await loc.count() == 1:
                    return loc

        target = self
        if self.node_type == NodeType.TEXT_NODE:
            target = self.parent
            assert target is not None

        # CSS selector
        for selector in target._build_selector_candidates():
            try:
                loc = tab.locator(selector)
                if await loc.count() == 1:
                    return loc
            except:
                pass

        # 祖先节点构造 css selector
        ancestors: list["DomNode"] = []
        node = target
        while node.parent is not None and node.node_type == NodeType.ELEMENT_NODE and len(ancestors) < 5:
            ancestors.append(node)
            node = node.parent

        selector = []
        for node in reversed(ancestors):
            cur_selector = node.local_name
            if "id" in node.attributes:
                cur_selector += f'#{css_escape(node.attributes["id"])}'
            selector.append(cur_selector)
        selector = " > ".join(selector)
        loc = tab.locator(selector)
        if await loc.count() == 1:
            return loc

        return None

    def get_description(self) -> str:

        def collect_text(node: "DomNode") -> list[str]:
            out: list[str] = []
            if node.node_type == NodeType.TEXT_NODE:
                if node.node_value.strip():
                    if len(node.node_value) < 20:
                        out.append(node.node_value)
                    else:
                        out.append(f"{node.node_value[:20]}...")

            for child in node.children:
                out.extend(collect_text(child))
            return out

        text = collect_text(self)
        content = f'({" ".join(text)})' if text else ""
        return f"{self.local_name}{content}"

    def max_bounds(self) -> tuple[float, float, float, float] | None:
        bounds_list: list[tuple[float, float, float, float]] = []

        def collect(node: "DomNode"):
            if node.style_visible and node.bounds is not None:
                bounds_list.append(node.bounds)

            for child in node.children:
                collect(child)

        collect(self)

        if not bounds_list:
            return None

        x1 = min(b[0] for b in bounds_list)
        y1 = min(b[1] for b in bounds_list)
        x2 = max(b[2] for b in bounds_list)
        y2 = max(b[3] for b in bounds_list)

        return x1, y1, x2, y2

    def draw_bounds(
        self,
        draw: ImageDraw.ImageDraw,
        viewport: Viewport,
        outline: str = "red",
        width=1,
        draw_id=False,
        recursive=True,
        max_bounds=False,
    ):
        bounds = self.max_bounds() if max_bounds else self.bounds
        if bounds:
            rect = map_bounds_to_viewport(bounds, viewport)
            if rect:
                draw.rectangle(rect, outline=outline, width=width)
                if draw_id:
                    draw.text(
                        (rect[0] + width, rect[1] + width), str(self.backend_node_id), fill=outline, font=default_font
                    )

        if recursive:
            for child in getattr(self, "children", []):
                child.draw_bounds(
                    draw=draw,
                    viewport=viewport,
                    outline=outline,
                    width=width,
                    draw_id=draw_id,
                    recursive=recursive,
                    max_bounds=max_bounds,
                )

    def _get_tree_repr(self, indent: int = 0, no_end=False, no_id=False):
        out = []

        prefix = "  " * indent
        attr_str = ""
        if self.repr_attributes:
            attr_pairs = [f'{k}="{v}"' if v else k for k, v in self.repr_attributes.items()]
            attr_str = " " + " ".join(attr_pairs)

        node_text = prefix
        if self.children:
            node_text += f"[{self.backend_node_id}]<{self.local_name}{attr_str}>"
        else:
            if self.node_type == NodeType.TEXT_NODE:
                node_text += f"{self.node_value}"
            else:
                node_text += f"[{self.backend_node_id}]<{self.local_name}{attr_str} />"

        out.append(node_text)
        for child in self.children:
            out.extend(child._get_tree_repr(indent=indent + 1, no_end=no_end, no_id=no_id))

        if self.children and not no_end:
            out.append(f"{prefix}{' ' * len(f'[{self.backend_node_id}]')}</{self.local_name}>")

        return out

    def get_human_tree_repr(self, indent: int = 0, no_end=False, no_id=False) -> str:
        return "\n".join(self._get_tree_repr(indent=indent, no_end=no_end, no_id=no_id))

    def is_overlap_viewport(self, viewport: Viewport) -> bool:
        if self.bounds is None or (self.bounds[0] == self.bounds[2] or self.bounds[1] == self.bounds[3]):
            return False

        ex1, ey1, ex2, ey2 = self.bounds
        vx1, vy1 = viewport.page_x, viewport.page_y
        vx2, vy2 = viewport.page_x + viewport.client_w, viewport.page_y + viewport.client_h

        horizontal_overlap = ex1 < vx2 and ex2 > vx1
        vertical_overlap = ey1 < vy2 and ey2 > vy1

        return horizontal_overlap and vertical_overlap

    def find_nodes_by_backend_node_ids(
        self,
        target_ids: list[int],
    ) -> list[Optional["DomNode"]]:
        node_map = {}

        stack: list["DomNode"] = [self]
        target_set = set(target_ids)

        while stack and target_set:
            node = stack.pop()
            if node.backend_node_id in target_set:
                node_map[node.backend_node_id] = node
                target_set.remove(node.backend_node_id)

            stack.extend(reversed(node.children))

        return [node_map.get(i) for i in target_ids]

    def find_ancestor_by_backend_node_id(self, target_id: int) -> Optional["DomNode"]:
        cur = self.parent
        while cur is not None:
            if cur.backend_node_id == target_id:
                return cur
            cur = cur.parent
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


async def load_dom(cdp_session: CDPSession) -> DomNode:
    dom_json = await cdp_session.send("DOM.getDocument", {"depth": -1})
    dom_snapshot = await cdp_session.send(
        "DOMSnapshot.captureSnapshot",
        {
            "computedStyles": ["display", "visibility", "opacity"],
            "includeDOMRects": True,
        },
    )

    dom_snapshot_strings = dom_snapshot["strings"]
    bounds_dict = {}
    visible_dict = {}

    def is_invisible_by_style(display: str, visibility: str, opacity: str) -> bool:
        if display.strip().lower() == "none":
            return True
        if visibility.strip().lower() == "hidden":
            return True
        if float(opacity) <= 0:
            return True

        return False

    for doc in dom_snapshot["documents"]:
        node_index_to_backend_node_ids = doc["nodes"]["backendNodeId"]
        for node_index, bounds, styles in zip(
            doc["layout"]["nodeIndex"], doc["layout"]["bounds"], doc["layout"]["styles"]
        ):
            backend_node_id = node_index_to_backend_node_ids[node_index]
            bounds_dict[backend_node_id] = bounds
            string_styles = [dom_snapshot_strings[idx] for idx in styles]

            is_visible = True
            if len(string_styles) == 3 and is_invisible_by_style(*string_styles):
                is_visible = False
            visible_dict[backend_node_id] = is_visible

    root = DomNode(dom_json["root"], None, bounds_dict, visible_dict)
    return root


def cluster_dom_rects(nodes: list[DomNode], alpha: float = 0.5, distance_threshold: float = 0.5) -> list[list[DomNode]]:
    """
    基于节点的空间距离和DOM结构距离进行层次聚类。
    参数:
    - nodes: DomNode节点列表
    - alpha: 空间距离的权重 (0~1)，其余权重(1-alpha)分配给DOM距离
    - distance_threshold: 聚类阈值（用于划分簇，取决于归一化距离）
    """

    n = len(nodes)
    if not n:
        return []
    if n == 1:
        return [nodes]
    # ============= 1. 计算节点中心点坐标 ============= #
    centers = []
    for node in nodes:
        if node.bounds is not None:
            x1, y1, x2, y2 = node.bounds
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        else:
            centers.append((0.0, 0.0))
    centers = np.array(centers)
    # ============= 2. 计算空间距离 (欧氏距离) ============= #
    spatial_dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    # 归一化
    if spatial_dist.max() > 0:
        spatial_dist /= spatial_dist.max()

    # ============= 3. 计算DOM树距离 ============= #
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
    # ============= 4. 综合距离矩阵 ============= #
    combined_dist = alpha * spatial_dist + (1 - alpha) * dom_dist
    # ============= 5. 层次聚类 ============= #
    z = linkage(squareform(combined_dist), method="average")
    labels: np.ndarray = fcluster(z, t=distance_threshold, criterion="distance")

    clusters: defaultdict[int, list[DomNode]] = defaultdict(list)
    for node, label in zip(nodes, labels):
        clusters[label].append(node)

    return list(clusters.values())


def get_cluster_covered_bounds(nodes: list[DomNode]) -> tuple[float, float, float, float]:
    max_bounds = [node.max_bounds() for node in nodes]
    valid_bounds = [b for b in max_bounds if b is not None]
    assert valid_bounds

    x1 = min(b[0] for b in valid_bounds)
    y1 = min(b[1] for b in valid_bounds)
    x2 = max(b[2] for b in valid_bounds)
    y2 = max(b[3] for b in valid_bounds)

    return x1, y1, x2, y2


def map_bounds_to_viewport(
    bounds: tuple[float, float, float, float], viewport: Viewport
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bounds
    vx1 = x1 - viewport.page_x
    vy1 = y1 - viewport.page_y
    vx2 = x2 - viewport.page_x
    vy2 = y2 - viewport.page_y

    # 完全无重叠
    if vx2 <= 0 or vy2 <= 0 or vx1 >= viewport.client_w or vy1 >= viewport.client_h:
        return None

    return max(vx1, 0), max(vy1, 0), min(vx2, int(viewport.client_w)), min(vy2, int(viewport.client_h))


def merge_overlapped_rects(
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


def image_layout_compaction(
    screenshot: Image.Image,
    cluster_rects: list[tuple[list[DomNode], tuple[int, int, int, int]]],
    viewport: Viewport,
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
            node_bounds = map_bounds_to_viewport(node_bounds, viewport)
            if node_bounds is None:
                continue
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
