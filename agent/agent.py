import asyncio
from datetime import datetime
import itertools
from pathlib import Path
import time
from typing import Any
from json_repair import repair_json
from loguru import logger
from playwright.async_api import BrowserContext, Page, CDPSession
from PIL import ImageFont, ImageDraw

from agent import dom_utils
from agent.action import Action, ActionDetails
from agent.config import Config
from agent.llm import PrimaryLLM
from agent.pruning import Pruning
from agent.record import ActRecord, Record, TimeLine
from agent.utils import draw_text_label, gen_uid, load_default_font, bg_colors, page_screenshot, time_stamp


class Agent:
    context: BrowserContext
    font_18: ImageFont.FreeTypeFont | ImageFont.ImageFont
    records: list[Record] = []

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, context: BrowserContext, user_request: str):
        self.font_18 = load_default_font(18)
        # 简单起见假设只有一个页面
        page = await context.new_page()
        cdp_session = await page.context.new_cdp_session(page)

        # TODO 测试
        await page.goto("https://www.google.com/")
        await asyncio.sleep(1)
        await self.act(page, cdp_session, "已进入google页面", "在搜索框输入'doubao'并搜索")

    # async def planning

    async def act(
        self,
        page: Page,
        cdp_session: CDPSession,
        current_state: str,
        next_objective: str,
    ):
        time_line = TimeLine()
        record_index = len(self.records) + 1
        record_prefix = f"{record_index:02}_act"
        logger.info(f"[N{record_index:02}][Act] Next objective: {next_objective}")

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

        screenshot = await page_screenshot(page)
        time_line.add("fetch")

        stage_tree_repr = []
        Pruning.trim_dom_tree_by_visibility(tree, viewport)
        stage_tree_repr.append(("visibility", tree.get_human_tree_repr()))
        Pruning.filter_dom_tree_by_node(tree)
        # stage_tree_repr.append(("filter", tree.get_human_tree_repr()))
        Pruning.promote_dom_tree_children(tree)
        # stage_tree_repr.append(("promote", tree.get_human_tree_repr()))
        Pruning.merge_dom_tree_children(tree)
        # stage_tree_repr.append(("merge", tree.get_human_tree_repr()))
        Pruning.clean_dom_tree_attrs(tree)
        stage_tree_repr.append(("rule", tree.get_human_tree_repr()))

        time_line.add("refine")

        if Config.debug:
            # 最细粒度元素可视化
            atomic_screenshot = screenshot.copy()
            atomic_screenshot_draw = ImageDraw.Draw(atomic_screenshot)
            tree.draw_bounds(atomic_screenshot_draw, viewport, draw_id=True)
            atomic_screenshot.save(self.out_dir / f"{record_prefix}_debug_1_atomic.jpg")

        interactive_nodes = Pruning.extract_interactive_nodes(tree)

        if Config.debug:
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
            interactive_screenshot.save(self.out_dir / f"{record_prefix}_debug_2_interactive.jpg")

            time_line.add("_debug")

        # 基于语义对元素进行剪枝
        # 空间聚类， alpha 高时以像素坐标的距离为主；alpha 低时以DOM结构的距离为主
        clusters = dom_utils.cluster_dom_rects(interactive_nodes, alpha=0.45, distance_threshold=0.45)
        clusters = [
            c
            for c in clusters
            if dom_utils.map_bounds_to_viewport(dom_utils.get_cluster_covered_bounds(c), viewport) is not None
        ]
        assert len(clusters) >= 1
        if len(clusters) == 1:
            logger.warning("[Cluster] Only one cluster, fallback to no pruning")
        else:
            logger.debug("[Cluster] Cluster count: {}", len(clusters))
        time_line.add("cluster")

        if Config.debug:
            # 聚类区域可视化
            cluster_screenshot = screenshot.copy()
            cluster_screenshot_draw = ImageDraw.Draw(cluster_screenshot)
            for index, cluster in enumerate(clusters):
                color = bg_colors[index % len(bg_colors)]
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
                cluster_screenshot_draw.rectangle(rect, outline=color, width=width + 2)
                draw_text_label(
                    draw=cluster_screenshot_draw,
                    position=(rect[0] + width + 5, rect[1] + width + 5),
                    text=f"{index + 1}",
                    font=self.font_18,
                    text_color="white",
                    bg_color=color,
                )
            cluster_screenshot.save(self.out_dir / f"{record_prefix}_debug_3_cluster_all.jpg")
            time_line.add("_debug")

        fallback = False
        final_nodes = interactive_nodes

        if len(clusters) == 1:
            fallback = True
        else:
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
                if Config.debug:
                    cluster_region.save(self.out_dir / f"{record_prefix}_debug_4_cluster_{index + 1}.jpg")
                related_tasks.append(Pruning.determine_cluster_related(cluster, cluster_region, task=next_objective))
            related_tasks_res = await asyncio.gather(*related_tasks)
            related_cluster = [
                (cluster, rect) for (flag, _), cluster, rect in zip(related_tasks_res, clusters, related_rects) if flag
            ]
            flags = [flag for flag, _ in related_tasks_res]
            logger.debug(
                f"[Pruning] Related {len(related_cluster)}, unrelated {len(flags) - len(related_cluster)}: {str(flags)}"
            )

            if not related_cluster:
                logger.warning("[Pruning] All unrelated, fallback to no pruning")
                fallback = True
            else:
                # 合并边界存在重叠的区域，用于构建紧凑的UI图
                merged_rects = dom_utils.merge_overlapped_rects(related_cluster)
                # 构建压缩后的UI图像
                final_screenshot = dom_utils.image_layout_compaction(
                    screenshot, merged_rects, viewport=viewport, default_gap=1
                )
                final_nodes: list[dom_utils.DomNode] = list(
                    itertools.chain.from_iterable(cluster for cluster, _ in related_cluster)
                )

        if fallback:
            # 回退到无语义剪枝
            final_screenshot = screenshot.copy()
            final_nodes = interactive_nodes
            final_screenshot_draw = ImageDraw.Draw(final_screenshot)
            for node in final_nodes:
                node.draw_bounds(
                    final_screenshot_draw,
                    viewport,
                    outline="red",
                    width=1,
                    draw_id=False,
                    recursive=False,
                    max_bounds=True,
                )
        if Config.debug:
            final_screenshot.save(self.out_dir / f"{record_prefix}_debug_5_final.jpg")

        stage_tree_repr.append(("pruning", "\n\n".join([node.get_human_tree_repr() for node in final_nodes])))
        time_line.add("pruning")
        pruned_dom_repr = stage_tree_repr[-1][1]

        prompt = f"""
You are an AI agent designed to operate in an iterative loop to automate browser tasks.
Based on the current task state, construct appropriate Actions to move the task toward the nearest next objective.

## Task Description
Current State: {current_state}
Nearest Next Objective: {next_objective}

## Page State
{viewport.get_page_info()}
Interactive Elements. Each element is prefixed with a unique [backend_node_id]:
{pruned_dom_repr}

## Available Actions
{Action.get_format_prompt()}

## Requirements
You may construct one or more Actions. If multiple Actions are provided, they will be executed sequentially.
All Actions MUST strictly follow the formats listed in Available Actions. Each Action must be output on its own line.
Do NOT include explanations or any additional text.
""".strip()

        llm_action_detail = await PrimaryLLM.chat_with_image_detail(prompt, final_screenshot)
        time_line.add("llm")
        llm_action_res = llm_action_detail["content"]

        action_details_list = []
        raw_action_list = llm_action_res.strip().split("\n")
        raw_action_types = [raw_action.split(",", maxsplit=1)[0].strip() for raw_action in raw_action_list]
        logger.debug(f"[Action] Construct {len(raw_action_list)} actions: [{', '.join(raw_action_types)}]")
        for action_index, raw_action in enumerate(raw_action_list, 1):
            action_uid = gen_uid()
            action_prefix = f"{record_prefix}_{action_index:02}_{action_uid}"
            action = Action.from_raw_action(uid=action_uid, csv_line=raw_action, dom_nodes=final_nodes)

            action_screenshot = await page_screenshot(page)
            action_screenshot_draw = ImageDraw.Draw(action_screenshot)
            draw_text_label(action_screenshot_draw, text=action.type.name, position=(10, 10), font=self.font_18)
            if action.target is not None:
                action.target.draw_bounds(
                    action_screenshot_draw,
                    viewport,
                    outline="red",
                    width=3,
                    draw_id=True,
                    recursive=False,
                    max_bounds=True,
                )
            action_screenshot_path = self.out_dir / f"{action_prefix}_action.jpg"
            action_screenshot.save(action_screenshot_path)
            execute_time = datetime.now()
            await action.execute(page, cdp_session)
            await asyncio.sleep(1)

            result_screenshot = await page_screenshot(page)
            result_screenshot_path = self.out_dir / f"{action_prefix}_result.jpg"
            result_screenshot.save(result_screenshot_path)

            action_details_list.append(
                ActionDetails(
                    raw_action=raw_action,
                    execute_time=execute_time,
                    action=action,
                    action_screenshot_path=action_screenshot_path,
                    result_screenshot_path=result_screenshot_path,
                )
            )
            time_line.add(f"execute_{action_index:02}")

        record = ActRecord(
            index=record_index,
            action_details_list=action_details_list,
            time_line=time_line,
            pruned_dom_repr=pruned_dom_repr,
            llm_details=llm_action_detail,
        )
        if Config.debug:
            record.save(self.out_dir)
        self.records.append(record)
