import math
from collections import defaultdict
from enum import IntEnum
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import tiktoken
from PIL import ImageFont
from PIL.ImageDraw import ImageDraw

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
        x = f"Landscape page: {self.page_index_x}/{self.page_count_x}" if self.page_count_x > 1 else ""
        y = f"Vertical page: {self.page_index_y}/{self.page_count_y}" if self.page_count_y > 1 else ""
        return f"{x} {y}".strip()


# Interactive elements
GROUP_ELEMENTS = {
    'a',
    'button',
    'input',
    'textarea',
    'select',
    'form',

    # 'ul',
    # 'ol',
    # 'li',
    # 'img',
    # 'svg',
    # 'label',
    # 'nav',
    # 'header',
    # 'footer',
    # 'article',
    # 'section',
}

SUPPORT_ELEMENTS = {
    # ========= 文档 / 布局语义 =========
    'html',
    'body',
    'main',
    'header',
    'footer',
    'nav',
    'section',
    'article',
    'aside',
    # ========= 通用容器 =========
    'div',
    'span',
    # ========= 标题 =========
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    # ========= 文本 / 内容 =========
    'p',
    'label',
    'strong',
    'em',
    'b',
    'i',
    'small',
    'mark',
    'code',
    'pre',
    'blockquote',
    # ========= 列表 =========
    'ul',
    'ol',
    'li',
    'dl',
    'dt',
    'dd',
    # ========= 链接与媒体 =========
    'a',
    'img',
    'svg',
    'canvas',
    'video',
    'audio',
    'iframe',
    # ========= 表单（交互关键） =========
    'form',
    'input',
    'textarea',
    'button',
    'select',
    'option',
    'fieldset',
    'legend',
    # ========= 表格（常见信息密集区） =========
    'table',
    'thead',
    'tbody',
    'tfoot',
    'tr',
    'th',
    'td',
    'caption',
    # ========= HTML5 交互 / 状态 =========
    'details',
    'summary',
    'dialog',
    'progress',
    'meter',
}


class DomNode:
    backend_node_id: int
    node_type: NodeType
    local_name: str
    node_value: str
    repr_value: str
    attributes: Dict[str, str]
    repr_attributes: Dict[str, str]
    children: List['DomNode']
    parent: 'DOMNode | None'
    bounds: tuple[float, float, float, float] | None  # (x1, y1, x2, y2)
    style_visible: bool
    level: int  # depth

    def __init__(self, node: Dict[str, Any], parent: 'DOMNode | None' = None, bounds_dict=None, visible_dict=None):
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
        return f"<DOMNode {self.local_name} backend_node_id={self.backend_node_id}>"

    def is_group(self):
        return self.local_name in GROUP_ELEMENTS

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

    def draw_bounds(self, draw: ImageDraw, viewport: Viewport, outline: str = "red", width=1, draw_id=False,
                    recursive=True, max_bounds=False):
        bounds = self.max_bounds() if max_bounds else self.bounds
        if bounds:
            rect = map_bounds_to_viewport(bounds, viewport)
            if rect:
                draw.rectangle(rect, outline=outline, width=width)
                if draw_id:
                    draw.text((rect[0] + width, rect[1] + width), str(self.backend_node_id), fill=outline,
                              font=default_font)

        if recursive:
            for child in getattr(self, "children", []):
                child.draw_bounds(draw=draw, viewport=viewport, outline=outline, width=width, draw_id=draw_id,
                                  recursive=recursive, max_bounds=max_bounds)

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

    def get_human_tree_repr(self, indent: int = 0, no_end=False, no_id=False):
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

    def find_nodes_by_backend_node_ids(self, target_ids: List[int], ) -> List[Optional['DomNode']]:
        node_map = {}

        stack = [self]
        target_set = set(target_ids)

        while stack and target_set:
            node = stack.pop()
            if node.backend_node_id in target_set:
                node_map[node.backend_node_id] = node
                target_set.remove(node.backend_node_id)

            stack.extend(reversed(node.children))

        return [node_map.get(i) for i in target_ids]

    def find_ancestor_by_backend_node_id(self, target_id: int) -> Optional['DomNode']:
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
        for node_index, bounds, styles in zip(doc["layout"]["nodeIndex"], doc["layout"]["bounds"],
                                              doc["layout"]["styles"]):
            bounds_dict[node_index_to_backend_node_ids[node_index]] = bounds
            string_styles = [dom_snapshot_strings[idx] for idx in styles]
            if string_styles and is_invisible_by_style(*string_styles):
                visible_dict[node_index_to_backend_node_ids[node_index]] = False
            else:
                visible_dict[node_index_to_backend_node_ids[node_index]] = True

    root = DomNode(dom_json["root"], None, bounds_dict, visible_dict)
    return root


def extract_interactive_nodes(root: DomNode):
    def dfs(node):
        output = []
        for child in node.children:
            output.extend(dfs(child))

        if not output:
            output.append(node)
        else:
            if node.is_group():
                output = [node]

        return output

    return dfs(root)


def cluster_dom_rects(
        nodes: List[DomNode], alpha: float = 0.5, distance_threshold: float = 0.5
) -> List[List[DomNode]]:
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
    z = linkage(squareform(combined_dist), method='average')
    labels: np.ndarray = fcluster(z, t=distance_threshold, criterion='distance')

    clusters: defaultdict[int, list[DomNode]] = defaultdict(list)
    for node, label in zip(nodes, labels):
        clusters[label].append(node)

    return list(clusters.values())


def get_covered_bounds(nodes: List[DomNode]) -> tuple[float, float, float, float]:
    max_bounds = [node.max_bounds() for node in nodes]
    valid_bounds = [b for b in max_bounds if b is not None]
    assert valid_bounds

    x1 = min(b[0] for b in valid_bounds)
    y1 = min(b[1] for b in valid_bounds)
    x2 = max(b[2] for b in valid_bounds)
    y2 = max(b[3] for b in valid_bounds)

    return x1, y1, x2, y2


def map_bounds_to_viewport(
        bounds: tuple[float, float, float, float],
        viewport: Viewport
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


def trim_dom_tree_by_visibility(root: DomNode, viewport: Viewport):
    def post_order(node: DomNode) -> bool:
        if not node.style_visible:
            return False

        if not node.children:
            return node.is_overlap_viewport(viewport)

        node.children = [child for child in node.children if post_order(child)]

        if node.is_overlap_viewport(viewport) and node.style_visible:
            return True

        # Keep the node if it still has any children
        if node.children:
            return True

        return False

    post_order(root)


def filter_dom_tree_by_node(root: DomNode):
    root.children = [child for child in root.children if _filter_dom_node(child)]

    for child in root.children:
        filter_dom_tree_by_node(child)


def clean_dom_tree_attrs(root: DomNode):
    _clean_dom_node_repr_attrs(root)

    for child in root.children:
        clean_dom_tree_attrs(child)


def promote_dom_tree_children(root: DomNode):
    if _promote_dom_node_children(root):
        def update_level(node, base_level):
            node.level = base_level
            for c in node.children:
                update_level(c, base_level + 1)

        idx = root.parent.children.index(root)
        parent = root.parent
        children = list(root.children)
        root.parent.children[idx:idx + 1] = children

        for child in children:
            child.parent = parent
            update_level(child, root.level)

        root.parent = None
        root.children = []

    for child in root.children:
        promote_dom_tree_children(child)


def merge_dom_tree_children(root: DomNode):
    if _merge_dom_node_children(root):
        new_value = []

        def recur(n: DomNode):
            if n.node_type == NodeType.TEXT_NODE:
                new_value.append(n.node_value)
            for c in n.children:
                recur(c)

        recur(root)
        root.node_value = " ".join(new_value)
        root.node_type = NodeType.TEXT_NODE
        root.children = []

    for child in root.children:
        merge_dom_tree_children(child)


def _filter_dom_node(node: DomNode) -> bool:
    """
    Decide whether this DOM node should be retained in the DOM tree
    过滤特定类型的一些节点
    """
    if node.node_type == NodeType.TEXT_NODE or node.node_type == NodeType.DOCUMENT_NODE:
        return True

    if node.local_name in SUPPORT_ELEMENTS:
        return True

    return False


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
    "type"
}


def _clean_dom_node_repr_attrs(node: DomNode):
    """
    Simplify and normalize the repr_attributes and value of a DOM node
    压缩节点属性
    """

    new_attrs = {}
    for name, value in node.repr_attributes.items():
        if name in IMPORTANT_ATTRS or name.startswith("aria-"):
            value = value.strip()
            if name == 'class':
                value = " ".join(value.strip().split()[:2])
            if len(value) > 15:
                value = value[:15] + "..."
            new_attrs[name] = value

    node.repr_attributes = new_attrs

    if node.node_type == NodeType.TEXT_NODE and len(node.repr_value) > 100:
        node.repr_value = node.repr_value[:100] + "..."


def _promote_dom_node_children(node: DomNode) -> bool:
    """
    Decide whether this node should be removed and its children promoted
    当节点的边界等于父节点的边界时
    """
    if node.parent is None:
        return False

    if node.local_name in {"html", "body"}:
        return True

    if node.bounds is not None and node.bounds == node.parent.bounds:
        return True

    return False


def _merge_dom_node_children(node: DomNode) -> bool:
    """
    Decide whether this subtree should be merged into a single DOM node
    当所有孩子都是文本节点时
    """
    if node.children and all([child.node_type == NodeType.TEXT_NODE for child in node.children]):
        return True

    return False
