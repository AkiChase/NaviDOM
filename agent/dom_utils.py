import itertools
import math
from collections import defaultdict, deque
from enum import IntEnum
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import tiktoken
from PIL import ImageFont, ImageDraw, Image

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
    page_count_y: int
    page_index_y: int
    page_count_x: int
    page_index_x: int

    def __init__(self, viewport: Dict[str, Any]):
        self.page_x = viewport["cssLayoutViewport"]["pageX"]
        self.page_y = viewport["cssLayoutViewport"]["pageY"]
        self.client_w = viewport["cssLayoutViewport"]["clientWidth"]
        self.client_h = viewport["cssLayoutViewport"]["clientHeight"]
        self.total_height = viewport["cssContentSize"]["height"]
        self.total_width = viewport["cssContentSize"]["width"]

        self.page_count_y = math.ceil(self.total_height / self.client_h)
        self.page_index_y = math.floor(self.page_y / self.client_w) + 1
        self.page_count_x = math.ceil(self.total_width / self.client_w)
        self.page_index_x = math.floor(self.page_y / self.client_h) + 1

    def get_page_info(self) -> str:
        y = f"Y Page: {self.page_index_y}/{self.page_count_y}" if self.page_count_y > 1 else ""
        return f"{y}".strip()


class DomNode:
    backend_node_id: int
    node_type: NodeType
    local_name: str
    node_value: str
    repr_value: str
    attributes: Dict[str, str]
    repr_attributes: Dict[str, str]
    children: List["DomNode"]
    parent: "DomNode | None"
    bounds: tuple[float, float, float, float] | None  # (x1, y1, x2, y2)
    style_visible: bool
    level: int  # depth

    def __init__(self, node: Dict[str, Any], parent: "DomNode | None" = None, bounds_dict=None, visible_dict=None):
        if bounds_dict is None:
            bounds_dict = {}
        if visible_dict is None:
            visible_dict = {}

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
            self.style_visible = True

        self.children: List[DomNode] = []
        self.parent = parent
        if self.parent is None:
            self.level = 0
        else:
            self.level = self.parent.level + 1

        for child in node.get("children", []):
            self.children.append(DomNode(child, self, bounds_dict))

    def __repr__(self):
        return f"<DomNode {self.local_name} backend_node_id={self.backend_node_id}>"

    def get_description(self) -> str:

        def collect_text(node: "DomNode") -> list[str]:
            out: list[str] = []
            if node.node_type == NodeType.TEXT_NODE:
                out.append(node.node_value)

            for child in node.children:
                out.extend(collect_text(child))
            return out

        text = collect_text(self)
        content = f'({" ".join(text)})' if text else ""
        return f"{self.local_name}{content}"

    def max_bounds(self) -> tuple[float, float, float, float] | None:
        bounds_list: list[tuple[float, float, float, float]] = []

        def collect(node: "DomNode"):
            if not node.style_visible or node.bounds is None:
                return

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
                node_text += f"[{self.backend_node_id}]{self.node_value}"
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
        target_ids: List[int],
    ) -> List[Optional["DomNode"]]:
        node_map = {}

        stack: List["DomNode"] = [self]
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


def parse_dom(dom_json: Dict[str, Any], dom_snapshot: Dict[str, Any]) -> DomNode:
    dom_snapshot_strings = dom_snapshot["strings"]
    bounds_dict = {}
    visible_dict = {}

    def is_invisible_by_style(display: str, visibility: str, opacity: str) -> bool:
        if display == "none":
            return True
        if visibility == "hidden":
            return True
        if float(opacity) <= 0:
            return True

        return False

    for doc in dom_snapshot["documents"]:
        node_index_to_backend_node_ids = doc["nodes"]["backendNodeId"]
        for node_index, bounds, styles in zip(
            doc["layout"]["nodeIndex"], doc["layout"]["bounds"], doc["layout"]["styles"]
        ):
            bounds_dict[node_index_to_backend_node_ids[node_index]] = bounds
            string_styles = [dom_snapshot_strings[idx] for idx in styles]
            if string_styles and is_invisible_by_style(*string_styles):
                visible_dict[node_index_to_backend_node_ids[node_index]] = False
            else:
                visible_dict[node_index_to_backend_node_ids[node_index]] = True

    root = DomNode(dom_json["root"], None, bounds_dict, visible_dict)
    return root


def cluster_dom_rects(nodes: List[DomNode], alpha: float = 0.5, distance_threshold: float = 0.5) -> List[List[DomNode]]:
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


def get_cluster_covered_bounds(nodes: List[DomNode]) -> tuple[float, float, float, float]:
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
