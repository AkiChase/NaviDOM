import asyncio
import itertools
import json
import math
import random
import time
from collections import defaultdict, deque
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from playwright.async_api import async_playwright, BrowserContext, Page

from agent.config import Config
import agent.dom_utils as dom_utils
import agent.pruning as pruning
from agent.llm import PrimaryLLM, estimate_image_tokens
from agent.pruning import Pruning


async def apply_task_action_env(
    context: BrowserContext,
    annotation_id: str,
    action_uid: str,
    bounds: dict,
    last_action_pos: tuple[float, float] | None,
    data_dir: Path,
):
    mhtml_path = data_dir / annotation_id / f"{action_uid}_before.mhtml"
    page = await context.new_page()
    await page.goto(str(mhtml_path))
    await asyncio.sleep(1)
    if last_action_pos is not None:
        await page.mouse.move(*last_action_pos)
        await asyncio.sleep(0.5)

    elem_top = bounds["y"]
    elem_bottom = bounds["y"] + bounds["height"]
    xy = (bounds["x"], elem_top, bounds["x"] + bounds["width"], elem_bottom)
    return page, xy


def time_stamp():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


async def main():
    data_path = Path("local/mind2web")
    Config.init("env.json")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=Config.browser_executable_path,
            headless=Config.browser_headless,
        )
        context = await browser.new_context(
            viewport={"width": Config.browser_viewport_w, "height": Config.browser_viewport_h}
        )

        all_data_names = ["test_domain", "test_task", "test_website"]
        for data_name in all_data_names:
            with open(f"local/{data_name}.json", "r", encoding="utf-8") as f:
                tasks: list[dict] = json.load(f)
            for t_index, task_data in enumerate(tasks[:10]):
                with open(data_path / task_data["annotation_id"] / "screenshot.json", "r", encoding="utf-8") as f:
                    actions_data = json.load(f)

                task_progress = f"[{t_index + 1}/{len(tasks)}]"
                annotation_id = task_data["annotation_id"]

                confirmed_task = task_data["confirmed_task"]

                logger.info(f"{task_progress} Processing task: {annotation_id}")
                last_action_pos = None
                for a_index, action in enumerate(actions_data):
                    action_progress = f"[{a_index + 1}/{len(actions_data)}]"
                    action_uid = action["action_uid"]

                    logger.info(f"{task_progress}{action_progress} Processing action: {action_uid}")
                    if "bounding_box" not in action["action"] or not action["action"]["bounding_box"]:
                        logger.warning(f"{task_progress}{action_progress} No bounding box for the action, skip")
                        continue
                    bounding_box = action["action"]["bounding_box"]

                    page, target_bounds = await apply_task_action_env(
                        context, annotation_id, action_uid, bounding_box, last_action_pos, data_path
                    )

                    out_dir_path = Path(f"out/{annotation_id}/{time_stamp()}_{a_index + 1:02}_{action_uid}")
                    out_dir_path.mkdir(parents=True, exist_ok=True)
                    last_action_pos = await evaluate(
                        page, annotation_id, confirmed_task, action_uid, target_bounds, out_dir_path
                    )
                    await page.close()
                    await asyncio.sleep(3)


def find_most_match_node(tree: dom_utils.DomNode, action_target_bounds: tuple[float, float, float, float]):
    min_dist_node: dom_utils.DomNode | None = None
    min_dist: float | None = None

    def cal_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        return sum([abs(ba - bb) for ba, bb in zip(a, b)])

    def find_element_by_bounds(root: dom_utils.DomNode):
        if root.bounds is not None:
            nonlocal min_dist, min_dist_node
            cur_dist = cal_dist(root.bounds, action_target_bounds)
            if min_dist is None or cur_dist < min_dist:
                min_dist = cur_dist
                min_dist_node = root

        for child in root.children:
            find_element_by_bounds(child)

    find_element_by_bounds(tree)
    if min_dist_node is None:
        raise ValueError("[Action] Failed to find element by bounds")

    for n in min_dist_node.get_ancestor_set():
        if n.local_name == "svg":
            min_dist_node = n
            break

    return min_dist_node


async def evaluate(
    page: Page,
    annotation_id: str,
    task_description: str,
    action_uid: str,
    action_target_bounds: tuple[float, float, float, float],
    out_dir: Path,
):
    stage_time = [("start", time.time())]

    # 获取 DOM 等数据
    cdp_session = await page.context.new_cdp_session(page)
    dom = await cdp_session.send("DOM.getDocument", {"depth": -1})
    snapshot = await cdp_session.send(
        "DOMSnapshot.captureSnapshot",
        {
            "computedStyles": ["display", "visibility", "opacity"],
            "includeDOMRects": True,
        },
    )
    viewport = dom_utils.Viewport(await cdp_session.send("Page.getLayoutMetrics"))
    font_18 = ImageFont.load_default(size=18)
    tree = dom_utils.parse_dom(dom, snapshot)

    # 基于bounds寻找最匹配目标元素
    target_node = find_most_match_node(tree, action_target_bounds)
    center_x = (target_node.bounds[0] + target_node.bounds[2]) / 2
    center_y = (target_node.bounds[1] + target_node.bounds[3]) / 2

    result: dict[str, Any] = {
        "task": {
            "annotation_id": annotation_id,
            "description": task_description,
            "action_uid": action_uid,
            "pos": [center_x, center_y],
        },
    }

    # 滚动到目标元素可见
    save_padding = 16
    viewport_h = viewport.client_h
    elem_h = action_target_bounds[3] - action_target_bounds[1]
    elem_top = action_target_bounds[1]
    elem_bottom = action_target_bounds[3]
    # 情况 A：元素可完整放进 viewport
    if elem_h <= viewport_h:
        min_scroll = elem_bottom - viewport_h
        max_scroll = elem_top
        # 情况 1：无需滚动即可完全可见
        if min_scroll <= 0 <= max_scroll:
            min_scroll = max_scroll = 0
    else:
        # 情况 B：元素比 viewport 还高
        min_scroll = elem_top
        max_scroll = elem_bottom - viewport_h
    # 调整安全间距
    if min_scroll == max_scroll and max_scroll == 0:
        if elem_top > viewport_h - save_padding:
            min_scroll = max_scroll = save_padding
    else:
        min_scroll += save_padding
        max_scroll -= save_padding
    final_scroll = random.randrange(round(min_scroll), round(max_scroll) + 1)
    if final_scroll != 0:
        await page.evaluate(f"window.scrollTo(0, {final_scroll});")

    screenshot_bytes = await page.screenshot(full_page=False, type="jpeg")
    screenshot = Image.open(BytesIO(screenshot_bytes))

    target_op_screenshot = screenshot.copy()
    target_op_screenshot_draw = ImageDraw.Draw(target_op_screenshot)
    target_op_screenshot_draw.rectangle(action_target_bounds, outline="red", width=3)
    target_node.draw_bounds(
        target_op_screenshot_draw, viewport, outline="green", width=3, draw_id=True, recursive=False
    )
    target_op_screenshot.save(out_dir / "0_target_op.jpg")

    stage_time.append(("fetch", time.time()))
    stage_tree_repr = [("raw", tree.get_human_tree_repr())]

    Pruning.trim_dom_tree_by_visibility(tree, viewport)
    stage_tree_repr.append(("visibility", tree.get_human_tree_repr()))

    Pruning.filter_dom_tree_by_node(tree)
    stage_tree_repr.append(("filter", tree.get_human_tree_repr()))

    Pruning.promote_dom_tree_children(tree)
    stage_tree_repr.append(("promote", tree.get_human_tree_repr()))

    Pruning.merge_dom_tree_children(tree)
    stage_tree_repr.append(("merge", tree.get_human_tree_repr()))

    Pruning.clean_dom_tree_attrs(tree)
    stage_tree_repr.append(("clean", tree.get_human_tree_repr()))
    stage_time.append(("refine", time.time()))

    # 最细粒度元素可视化
    atomic_screenshot = screenshot.copy()
    atomic_screenshot_draw = ImageDraw.Draw(atomic_screenshot)
    tree.draw_bounds(atomic_screenshot_draw, viewport, draw_id=True)
    atomic_screenshot.save(out_dir / "1_atomic.jpg")

    interactive_nodes = dom_utils.extract_interactive_nodes(tree)
    # 交互粒度元素可视化
    interactive_screenshot = screenshot.copy()
    interactive_screenshot_draw = ImageDraw.Draw(interactive_screenshot)
    for node in interactive_nodes:
        node.draw_bounds(
            interactive_screenshot_draw,
            viewport,
            outline="blue",
            width=2,
            draw_id=True,
            recursive=False,
            max_bounds=True,
        )
    interactive_screenshot.save(out_dir / "2_interactive.jpg")
    stage_time.append(("_debug", time.time()))

    # alpha 高时以空间距离为主，方便裁剪
    # alpha 低时以DOM结构为主，更符合分类
    clusters = dom_utils.cluster_dom_rects(interactive_nodes, alpha=0.45, distance_threshold=0.45)
    clusters = [
        c
        for c in clusters
        if dom_utils.map_bounds_to_viewport(dom_utils.get_cluster_covered_bounds(c), viewport) is not None
    ]
    assert len(clusters) >= 1
    if len(clusters) == 1:
        logger.warning("[Cluster] Cluster label count too small: {}", len(clusters))
    else:
        logger.debug("[Cluster] Cluster count: {}", len(clusters))
    stage_time.append(("cluster", time.time()))

    # 聚类区域可视化
    cluster_screenshot = screenshot.copy()
    cluster_screenshot_draw = ImageDraw.Draw(cluster_screenshot)
    colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink"]
    for index, cluster in enumerate(clusters):
        color = colors[index % len(colors)]
        width = 2
        for node in cluster:
            node.draw_bounds(
                cluster_screenshot_draw,
                viewport,
                outline=color,
                width=width,
                draw_id=True,
                recursive=False,
                max_bounds=True,
            )

        bounds = dom_utils.get_cluster_covered_bounds(cluster)
        rect = dom_utils.map_bounds_to_viewport(bounds, viewport)
        assert rect is not None
        cluster_screenshot_draw.rectangle(rect, outline=color, width=width + 3)
        cluster_screenshot_draw.text(
            (rect[0] + width + 3, rect[1] + width + 3), f"{index + 1}", fill=color, font=font_18
        )
    cluster_screenshot.save(out_dir / "3_cluster_all.jpg")
    stage_time.append(("_debug", time.time()))

    # 基于LLM过滤与任务不相关的聚类区域
    related_tasks = []
    related_rects = []
    for index, cluster in enumerate(clusters):
        bounds = dom_utils.get_cluster_covered_bounds(cluster)
        rect = dom_utils.map_bounds_to_viewport(bounds, viewport)
        assert rect is not None
        related_rects.append(rect)
        cluster_region = screenshot.crop(rect)
        cluster_region_draw = ImageDraw.Draw(cluster_region)
        for node in cluster:
            node_bounds = node.max_bounds()
            if node_bounds is None:
                continue
            node_bounds = dom_utils.map_bounds_to_viewport(node_bounds, viewport)
            if node_bounds is None:
                continue
            node_bounds = (
                node_bounds[0] - rect[0],
                node_bounds[1] - rect[1],
                node_bounds[2] - rect[0],
                node_bounds[3] - rect[1],
            )
            cluster_region_draw.rectangle(node_bounds, outline="red", width=2)
        cluster_region.save(out_dir / f"4_cluster_{index + 1}.jpg")
        related_tasks.append(pruning._determine_cluster_related(cluster, cluster_region, task=task_description))
    # if len(clusters) > 1:
    related_tasks_res = await asyncio.gather(*related_tasks)
    related_cluster = [
        (cluster, rect) for (flag, _), cluster, rect in zip(related_tasks_res, clusters, related_rects) if flag
    ]
    flags = [flag for flag, _ in related_tasks_res]
    if related_cluster:
        logger.debug(
            "[Pruning] {} related, {} unrelated: {}",
            len(related_cluster),
            len(flags) - len(related_cluster),
            str(flags),
        )
        # else:
        #     logger.warning("[Pruning] All unrelated, fallback to all, {}", str(flags))
        #     related_cluster = list(zip(clusters, related_rects))
    # else:
    #     flags = None
    #     logger.warning("[Pruning] Only one cluster, skip pruning")
    #     related_cluster = [(clusters[0], related_rects[0])]

    result["clusters"] = {"all_count": len(clusters), "related_count": len(related_cluster), "result": flags}

    stage_time.append(("pruning", time.time()))

    if related_cluster:
        # 合并边界存在重叠的区域
        merged_rects = merge_overlapped_rects(related_cluster)

        # 构建压缩后的UI图像
        compacted_screenshot = image_layout_compaction(screenshot, merged_rects, viewport=viewport, default_gap=1)
        compacted_screenshot.save(out_dir / "5_compacted.jpg")
        stage_time.append(("compact", time.time()))

        final_nodes: list[dom_utils.DomNode] = list(
            itertools.chain.from_iterable(cluster for cluster, _rect in related_cluster)
        )
        stage_tree_repr.append(("pruning", "\n\n".join([node.get_human_tree_repr() for node in final_nodes])))

        # 输出 token, time 对比信息
        result["text_tokens"] = pretty_print_tokens(stage_tree_repr)
        result["image_tokens"] = pretty_print_image_token(screenshot, compacted_screenshot, PrimaryLLM.image_model)
        result["time"] = pretty_print_time(stage_time)

        # 判断操作目标元素是否被过滤
        success = False
        for node in final_nodes:
            if node.find_ancestor_by_backend_node_id(target_node.backend_node_id) is not None:
                success = True
                break

            if node.find_nodes_by_backend_node_ids([target_node.backend_node_id])[0] is not None:
                success = True
                break
        if not success:
            logger.warning("[Result] Action target node is filtered")
        else:
            logger.success("[Result] Action target node available")

            agent_determined_id, details = await determine_action_by_llm(
                task_description, stage_tree_repr[-1][1], compacted_screenshot
            )
            determined_node = tree.find_nodes_by_backend_node_ids([agent_determined_id])[0]
            if determined_node is not None:
                agent_determined_image = screenshot.copy()
                agent_determined_image_draw = ImageDraw.Draw(agent_determined_image)
                determined_node.draw_bounds(
                    agent_determined_image_draw, viewport, outline="red", width=2, draw_id=True, recursive=False
                )
                agent_determined_image.save(out_dir / "6_agent_determined.jpg")
            result["agent"] = details

            if (
                target_node.find_ancestor_by_backend_node_id(agent_determined_id) is not None
                or target_node.find_nodes_by_backend_node_ids([agent_determined_id])[0] is not None
            ):
                success = True
                logger.success("[Result] Agent determined node is correct")
            else:
                success = False
                logger.warning("[Result] Agent determined node is incorrect")
    else:
        logger.warning("[Pruning] All unrelated")
        logger.warning("[Result] Action target node is filtered")
        success = False

    result["success"] = success

    # 保存结果
    with open(out_dir / "raw_dom.txt", "w", encoding="utf-8") as f:
        f.write(stage_tree_repr[0][1])
    with open(out_dir / "dom.txt", "w", encoding="utf-8") as f:
        f.write(stage_tree_repr[-1][1])
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return center_x, center_y


def merge_overlapped_rects(
    related_cluster: list[tuple[list[dom_utils.DomNode], tuple[int, int, int, int]]],
) -> list[tuple[list[dom_utils.DomNode], tuple[int, int, int, int]]]:
    merged = []

    def overlaps(r1, r2):
        _, (x11, y11, x12, y12) = r1
        _, (x21, y21, x22, y22) = r2
        return not (x12 < x21 or x22 < x11 or y12 < y21 or y22 < y11)

    def union(r1, r2):
        n1, (x11, y11, x12, y12) = r1
        n2, (x21, y21, x22, y22) = r2
        return n1 + n2, (min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22))

    rects = related_cluster.copy()
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


def image_layout_compaction(
    screenshot: Image.Image,
    merged_rects: list[tuple[list[dom_utils.DomNode], tuple[int, int, int, int]]],
    viewport: dom_utils.Viewport,
    default_gap=5,
):
    crops = []
    for cluster, rect in merged_rects:
        crop = screenshot.crop(rect)
        draw = ImageDraw.Draw(crop)
        for node in cluster:
            node_bounds = node.max_bounds()
            if node_bounds is None:
                continue
            node_bounds = dom_utils.map_bounds_to_viewport(node_bounds, viewport)
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
    for i, (_, (x1, y1, x2, y2)) in enumerate(merged_rects):
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


def pretty_print_tokens(stage_reprs: list[tuple[str, str]]):
    if not stage_reprs:
        return

    stage_tokens = [(name, len(dom_utils.tiktoken_enc.encode(text))) for name, text in stage_reprs]

    raw_cnt = stage_tokens[0][1]
    if PRINT_FLAG:
        print("=" * 46)
        print("📊 DOM Token Compression Summary")
        print("=" * 46)
        print(f"{'Stage':<10}{'Tokens':>10}{'ΔTokens':>12}{'Compression':>12}")
        print("-" * 46)

    out = []
    prev = None
    for name, cnt in stage_tokens:
        if prev is None:
            delta = 0
            compression = 1.0
        else:
            delta = prev - cnt  # 只表示减少量
            compression = raw_cnt / cnt if cnt > 0 else float("inf")
        out.append({"name": name, "tokens": cnt, "delta": delta, "compression": round(compression, 2)})

        if PRINT_FLAG:
            print(f"{name:<10}" f"{cnt:>10}" f"{delta:>12}" f"{compression:>11.2f}×")

        prev = cnt

    return out


def pretty_print_time(stage_time: list[tuple[str, float]]):
    if len(stage_time) < 2:
        print("Not enough stage records.")
        return

    rows = []
    total_time = 0.0
    total_time_without_debug = 0.0

    for i in range(1, len(stage_time)):
        _, prev_ts = stage_time[i - 1]
        stage, ts = stage_time[i]
        cost = ts - prev_ts

        rows.append((stage, cost))
        total_time += cost

        if stage != "_debug":
            total_time_without_debug += cost

    name_width = max(len(name) for name, _ in rows)
    if PRINT_FLAG:
        print("=" * (name_width + 36))
        print("📊 Stage Time Breakdown")
        print("=" * (name_width + 36))
        print(f"{'Stage'.ljust(name_width)} | Cost (ms) |  % Total")
        print("-" * (name_width + 36))

    out = []
    for stage, cost in rows:
        percent = (cost / total_time * 100) if total_time > 0 else 0.0
        out.append({"name": stage, "time": round(cost, 2), "percent": round(percent, 2)})
        if PRINT_FLAG:
            print(f"{stage.ljust(name_width)} | " f"{cost * 1000:9.2f} | " f"{percent:7.2f}%")
    if PRINT_FLAG:
        print("-" * (name_width + 36))

    true_percent = (total_time_without_debug / total_time * 100) if total_time > 0 else 0.0

    if PRINT_FLAG:
        print(f"{'True'.ljust(name_width)} | " f"{total_time_without_debug * 1000:9.2f} | " f"{true_percent:7.2f}%")

        print(f"{'Total'.ljust(name_width)} | " f"{total_time * 1000:9.2f} | " f"{100.00:7.2f}%")

    return {
        "stages": out,
        "total_time": total_time,
        "total_time_without_debug": total_time_without_debug,
    }


def pretty_print_image_token(
    screenshot: Image.Image,
    compacted_img: Image.Image,
    model_id: str,
):
    raw_token, raw_resized_size = estimate_image_tokens(screenshot.size[0], screenshot.size[1], model_id)
    compacted_token, compacted_resized_size = estimate_image_tokens(
        compacted_img.size[0], compacted_img.size[1], model_id
    )

    token_diff = raw_token - compacted_token
    compression_ratio = raw_token / compacted_token if compacted_token != 0 else float("inf")

    def fmt_size(size: tuple[float, float]) -> str:
        return f"{size[0]:.1f} x {size[1]:.1f}"

    def fmt_token(t: int) -> str:
        return f"{t:,}"

    if PRINT_FLAG:
        print("=" * 72)
        print(f"📸 Image Token Comparison (model={model_id})")
        print("=" * 72)

        header = f"{'':<12} | {'Token':>12} | {'Raw size':>18} | {'Resized size':>18}"
        print(header)
        print("-" * len(header))

        print(
            f"{'Original':<12} | "
            f"{fmt_token(raw_token):>12} | "
            f"{screenshot.size[0]} x {screenshot.size[1]:<14} | "
            f"{fmt_size(raw_resized_size):>18}"
        )

        print(
            f"{'Compacted':<12} | "
            f"{fmt_token(compacted_token):>12} | "
            f"{compacted_img.size[0]} x {compacted_img.size[1]:<14} | "
            f"{fmt_size(compacted_resized_size):>18}"
        )

        print("-" * len(header))

        sign = "+" if token_diff > 0 else ""

        print(f"{'Δ Token':<12} | " f"{sign}{token_diff:,} | " f"Compression: {compression_ratio:.2f}x")

    return {
        "original": {
            "token": raw_token,
            "size": screenshot.size,
        },
        "compacted": {
            "token": compacted_token,
            "size": compacted_img.size,
        },
        "token_reduction": token_diff,
        "compression_ratio": compression_ratio,
    }


if __name__ == "__main__":
    PRINT_FLAG = False
    asyncio.run(main())
