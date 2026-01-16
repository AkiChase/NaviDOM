from agent.dom_utils import DomNode, NodeType, Viewport
from agent.llm import SecondaryLLM
from PIL import Image


# Interactive elements
GROUP_ELEMENTS = {
    "a",
    "button",
    "input",
    "textarea",
    "select",
    "form",
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


class Pruning:

    @staticmethod
    def extract_interactive_nodes(root: DomNode):
        def dfs(node: DomNode):
            output = []
            for child in node.children:
                output.extend(dfs(child))

            if not output:
                output.append(node)
            else:
                if node.local_name in GROUP_ELEMENTS:
                    output = [node]

            return output

        return dfs(root)

    @staticmethod
    async def determine_cluster_related(nodes: list[DomNode], image: Image.Image, task: str):
        prompt = f"""
You are an experienced UI automation testing developer.
You need to determine whether the UI elements in the current region screenshot could be related to "{task}".
You should make this judgment based on the DOM elements list provided below and the corresponding UI screenshot (the relevant elements are highlighted with red boxes).
If there is any potential relationship, output Yes; if they are completely unrelated, output No. Do not output any other text.

DOM elements:
{'\n\n'.join([node.get_human_tree_repr(no_end=True, no_id=True) for node in nodes])}
    """.strip()

        # resized_image = dom_image.copy().resize((round(dom_image.size[0] * 0.75), round(dom_image.size[1] * 0.75)))
        res = await SecondaryLLM.chat_with_image_detail(
            prompt=prompt,
            image=image,
        )

        content: str = res["content"].strip()
        assert content.lower() in {"yes", "no"}
        flag = content.lower() == "yes"
        return flag, res

    @staticmethod
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

    @staticmethod
    def filter_dom_tree_by_node(root: DomNode):
        retained_children = []
        for child in root.children:
            if Pruning.filter_dom_tree_by_node(child):  # 递归剪枝子树
                retained_children.append(child)

        root.children = retained_children
        return _filter_dom_node(root)

    @staticmethod
    def clean_dom_tree_attrs(root: DomNode):
        _clean_dom_node_repr_attrs(root)

        for child in root.children:
            Pruning.clean_dom_tree_attrs(child)

    @staticmethod
    def promote_dom_tree_children(root: DomNode):
        if _promote_dom_node_children(root):

            def update_level(node, base_level):
                node.level = base_level
                for c in node.children:
                    update_level(c, base_level + 1)

            if root.parent is None:
                return

            idx = root.parent.children.index(root)
            parent = root.parent
            children = list(root.children)
            root.parent.children[idx : idx + 1] = children

            for child in children:
                child.parent = parent
                update_level(child, root.level)

            root.parent = None
            root.children = []

        for child in root.children:
            Pruning.promote_dom_tree_children(child)

    @staticmethod
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
            Pruning.merge_dom_tree_children(child)

    @staticmethod
    def extract_dom_tree_text(root: DomNode) -> str:
        lines = _extract_text_dom(root)
        return "\n".join(lines)


def _extract_text_dom(
    node: DomNode,
    indent: int = 0,
    indent_unit: str = "  ",
) -> list[str]:
    lines: list[str] = []

    if node.node_type == NodeType.TEXT_NODE:
        text = (node.node_value or "").strip()
        if text:
            lines.append(f"{indent_unit * indent}{text}")
        return lines

    child_lines: list[str] = []
    for child in node.children:
        child_lines.extend(_extract_text_dom(child, indent + 1, indent_unit))

    if child_lines:
        lines.extend(child_lines)

    return lines


def _filter_dom_node(node: DomNode) -> bool:
    """
    Decide whether this DOM node should be retained in the DOM tree
    过滤特定类型的一些节点
    """

    # 如果节点在支持列表中，保留
    if node.local_name in SUPPORT_ELEMENTS or node.node_type == NodeType.TEXT_NODE:
        return True

    # 如果有子节点也保留（作为容器）
    if len(node.children) > 0:
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
    "type",
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
            if name == "class":
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
