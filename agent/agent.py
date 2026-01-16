import asyncio
from datetime import datetime
import itertools
from pathlib import Path
import re
from loguru import logger
from playwright.async_api import BrowserContext, Page, CDPSession
from PIL import ImageFont, ImageDraw, Image

from agent import dom_utils
from agent.action import (
    Action,
    ActionDetails,
    ActionExecuteException,
    ActionExecuteResult,
    ActionParseException,
    ActionType,
)
from agent.config import Config
from agent.llm import PrimaryLLM, SecondaryLLM
from agent.pruning import Pruning
from agent.record import ActRecord, ObservationRecord, PlanningRecord, Record, TimeLine
from agent.utils import (
    draw_text_label,
    format_time_delta,
    gen_uid,
    load_default_font,
    bg_colors,
    page_screenshot,
)


class Agent:
    context: BrowserContext
    out_dir: Path
    user_request: str
    font_18: ImageFont.FreeTypeFont | ImageFont.ImageFont
    records: list[Record]
    last_planning_record: PlanningRecord | None
    last_act_record: ActRecord | None = None
    last_observation_record: ObservationRecord | None
    memory: list[tuple[str, str]]

    def __init__(self, out_dir: Path, context: BrowserContext, user_request: str):
        self.context = context
        self.out_dir = out_dir
        self.user_request = user_request
        self.font_18 = load_default_font(18)
        self.records = []
        self.last_planning_record = None
        self.last_act_record = None
        self.last_observation_record = None
        self.memory = []

        out_dir.mkdir(parents=True, exist_ok=True)

    def get_formatted_memory(self) -> str:
        if not self.memory:
            return "No entries."
        return "\n".join([f"- {label}: {value}" for label, value in self.memory])

    async def run(self):
        start_time = datetime.now()
        # 简单起见假设只有一个页面
        page = await self.context.new_page()
        await page.add_init_script("Object.defineProperties(navigator, {webdriver:{get:()=>undefined}});")

        cdp_session = await page.context.new_cdp_session(page)

        # 初次导航
        await self.act_initial(page, cdp_session)
        # 初次规划
        await self.planning_initial(page)

        iteration_times = 0
        while True:
            iteration_times += 1

            before_act_screenshot = await page_screenshot(page)
            last_error = None
            for act_time in range(1, Config.max_act_retry_times + 1):
                # Act
                await self.act(page, cdp_session, last_error)
                assert self.last_act_record is not None
                if all(a.execute_result["success"] == False for a in self.last_act_record.action_details_list):
                    if act_time >= Config.max_act_retry_times:
                        logger.error(f"[Act] Last act failed, stop: {act_time}/{Config.max_act_retry_times}")
                        return
                    else:
                        last_error = "\n".join(
                            [
                                f'{index}. {a.execute_result["additional"]}'
                                for index, a in enumerate(self.last_act_record.action_details_list)
                                if a.execute_result["success"] == False
                            ]
                        )
                        logger.warning(f"[Act] Last act failed, try again: {act_time}/{Config.max_act_retry_times}")
                else:
                    break
            # Observation
            await self.observation(page, before_act_screenshot)
            # Planning
            await self.planning()

            assert self.last_planning_record is not None
            if self.last_planning_record.task_completed:
                logger.success(f"[Planning] Task completed: {self.last_planning_record.current_state}")
                break

            if iteration_times >= Config.max_iteration_times:
                logger.warning(
                    f"[Planning] Max iteration times reached, stop: {iteration_times}/{Config.max_iteration_times}"
                )
                break

        logger.info(f"[Time] Total time cost: {format_time_delta(start_time, datetime.now())}")

    async def observation(self, page: Page, before_act_screenshot: Image.Image):
        time_line = TimeLine()
        record_index = len(self.records) + 1

        assert self.last_act_record is not None
        success_action_details_list = [
            d for d in self.last_act_record.action_details_list if d.execute_result["success"]
        ]
        assert len(success_action_details_list) > 0
        success_ui_change_action_list = [
            d for d in success_action_details_list if d.action.type not in {ActionType.Extract, ActionType.Memory}
        ]

        if success_ui_change_action_list:
            logger.info(f"[{record_index:03}] Observation")
            after_act_screenshot = await page_screenshot(page)
            if Config.debug:
                record_prefix = f"{record_index:03}_observation"
                before_act_screenshot.save(self.out_dir / f"{record_prefix}_debug_before.jpg")
                after_act_screenshot.save(self.out_dir / f"{record_prefix}_debug_after.jpg")

            actions_info = "\n".join(
                [f"{index}. {d.action.get_description()}" for index, d in enumerate(success_ui_change_action_list)]
            )

            time_line.add("fetch")
            prompt = f"""
You are an assistant with strong visual analysis skills.
You are given:
- Screenshot 1: previous page BEFORE the actions
- Screenshot 2: current page AFTER the actions
- A brief description of the actions that were performed:
{actions_info}

Based only on what you see, write one concise paragraph that:
- Describes the main visual differences between the two screenshots (what appeared, disappeared, moved, or changed state, e.g., dropdowns, tabs, modals, buttons).
- Explains, at a high level, what result is reflected on the page and what the user can do on the second page in its current state.

Important constraints:
- Do NOT infer user intent or speculate on why the actions were taken.
- Do NOT restate or interpret the actions themselves.
- Use the action description only as background context to better understand the visual transition, not as evidence.
- Focus strictly on observable UI changes and the resulting page state.
            """.strip()
            llm_details = await SecondaryLLM.chat_with_image_list_detail(
                prompt, [before_act_screenshot, after_act_screenshot]
            )
            logger.debug(f"[Result] {llm_details['content']}")
            observation = f"### Last Actions Result\n{llm_details['content']}"
            time_line.add("llm")
        else:
            observation = f"UI page not changed"
            llm_details = None
            time_line.add("fetch")

        record = ObservationRecord(
            index=record_index,
            llm_details=llm_details,
            observation=observation,
            time_line=time_line,
        )
        if Config.debug:
            record.save(self.out_dir)
        self.records.append(record)
        self.last_observation_record = record

    async def planning_initial(self, page: Page):
        time_line = TimeLine()
        record_index = len(self.records) + 1
        logger.info(f"[{record_index:03}] Planning initial")

        screenshot = await page_screenshot(page)
        time_line.add("fetch")

        prompt = f"""
You are an AI agent designed to make a planning to accomplish the given user request.
Your job is to analyze the current situation and produce a concise, structured plan.

## User Request
{self.user_request}

## Requirements
- Do not assume information that is not visible in the UI screenshot.
- Do not describe specific actions.
- Do not include additional explanations.
- Focus on intent, state, and goals rather than implementation details.
- Respond in the following structure and format:
[Current State]
Briefly describe what the page currently represents and what progress (if any) has been made toward the task objective.
[Nearest Next Objective]
State the single most immediate sub-goal that should be achieved next in order to move closer to the task objective. This should be concrete and actionable (e.g., “open login form”, “submit search query”, “navigate to checkout page”).
[Future Plan]
Outline the expected high-level steps that will follow AFTER the nearest next objective is completed. Keep this concise and forward-looking; do not describe low-level actions or UI mechanics.
        """.strip()

        llm_details = await PrimaryLLM.chat_with_image_detail(prompt, screenshot)
        time_line.add("llm")

        content = llm_details["content"]
        pattern = re.compile(r"\[Current State\]\s*(.*?)\s*(?=\[Nearest Next Objective\]|\Z)", re.DOTALL)
        match = pattern.search(content)
        assert match is not None
        current_state: str = match.group(1).strip()

        pattern = re.compile(r"\[Nearest Next Objective\]\s*(.*?)\s*(?=\[Future Plan\]|\Z)", re.DOTALL)
        match = pattern.search(content)
        assert match is not None
        nearest_next_objective: str = match.group(1).strip()

        pattern = re.compile(r"\[Future Plan\]\s*(.*)", re.DOTALL)
        match = pattern.search(content)
        assert match is not None
        future_plan: str = match.group(1).strip()

        logger.debug(f"[Current State] {current_state}")
        logger.debug(f"[Nearest Next Objective] {nearest_next_objective}")
        logger.debug(f"[Future Plan] {future_plan}")

        record = PlanningRecord(
            index=record_index,
            llm_details=llm_details,
            current_state=current_state,
            nearest_next_objective=nearest_next_objective,
            future_plan=future_plan,
            task_completed=False,
        )
        if Config.debug:
            record.save(self.out_dir)
        self.records.append(record)
        self.last_planning_record = record

    async def planning(self):
        time_line = TimeLine()
        record_index = len(self.records) + 1
        logger.info(f"[{record_index:03}] Planning")

        assert self.last_planning_record is not None
        assert self.last_observation_record is not None

        time_line.add("fetch")

        prompt = f"""
You are an AI agent designed to make a planning to accomplish the given user request.
Your job is to analyze the current situation and produce a concise, structured plan,
or determine that the task has already been completed.

## User Request
{self.user_request}

## Last Planning Details
Last Task State: {self.last_planning_record.current_state}
Last Nearest Next Objective: {self.last_planning_record.nearest_next_objective}
Last Future Plan: {self.last_planning_record.future_plan}

## Observation Details
{self.last_observation_record.observation}

## Memory Entries
The following items were stored related to the user request in previous iterations.
They represent intermediate results that will be refined and returned to the user as part of the final task result.
{self.get_formatted_memory()}

## Requirements
1. First, determine whether the user request has already been fully completed based on the current above information.
2. If the task is completed, based strictly on the User Request, output the final task result. Do not add any extra content. Output exactly: TASK_COMPLETED, <result>
3. If the task is not completed, reply according to the following rules:
- Do not describe specific user actions or UI-level operations.
- Focus on intent, current state, and goals rather than implementation details.
- Base the plan strictly on: the provided User Request, Last Planning Details, Observation Details. Do not invent or assume information that is not explicitly provided.
- Reflect task progress accurately; do not restate or repeat already completed objectives.
- If the task is NOT completed, Output should follow this exact structure and headings:
[Current State]
Briefly describe what the page currently represents and what progress (if any) has been made toward the task objective.
[Nearest Next Objective]
State the single most immediate sub-goal that should be achieved next in order to move closer to the task objective. This should be concrete and actionable (e.g., “open login form”, “submit search query”, “navigate to checkout page”).
[Future Plan]
Outline the expected high-level steps that will follow AFTER the nearest next objective is completed. Keep this concise and forward-looking; do not describe low-level actions or UI mechanics.
        """.strip()

        llm_details = await PrimaryLLM.chat_with_text_detail(prompt)
        time_line.add("llm")

        content = llm_details["content"].strip()
        if content.startswith("TASK_COMPLETED"):
            current_state = content.split(",", maxsplit=1)[1].strip()
            nearest_next_objective = ""
            future_plan = ""
            task_completed = True
        else:
            pattern = re.compile(r"\[Current State\]\s*(.*?)\s*(?=\[Nearest Next Objective\]|\Z)", re.DOTALL)
            match = pattern.search(content)
            assert match is not None
            current_state: str = match.group(1).strip()

            pattern = re.compile(r"\[Nearest Next Objective\]\s*(.*?)\s*(?=\[Future Plan\]|\Z)", re.DOTALL)
            match = pattern.search(content)
            assert match is not None
            nearest_next_objective: str = match.group(1).strip()

            pattern = re.compile(r"\[Future Plan\]\s*(.*)", re.DOTALL)
            match = pattern.search(content)
            assert match is not None
            future_plan: str = match.group(1).strip()
            task_completed = False

            logger.debug(f"[Current State] {current_state}")
            logger.debug(f"[Nearest Next Objective] {nearest_next_objective}")
            logger.debug(f"[Future Plan] {future_plan}")

        record = PlanningRecord(
            index=record_index,
            llm_details=llm_details,
            current_state=current_state,
            nearest_next_objective=nearest_next_objective,
            future_plan=future_plan,
            task_completed=task_completed,
        )
        if Config.debug:
            record.save(self.out_dir)
        self.records.append(record)
        self.last_planning_record = record

    async def act_initial(self, page: Page, cdp_session: CDPSession):
        prompt = f"""
You are an AI agent designed to automate browser tasks.
Your task is to analyze the User Request and determine the initial browser page to navigate to.

Available Actions:
{Action.get_format_prompt([ActionType.Navigate, ActionType.Search])}

Requirements:
- If the user request explicitly specifies a URL to start from, output a NAVIGATE action with that URL.
- If the user request asks to start from a clearly identifiable, well-known website, output a NAVIGATE action with that website's URL.
- Otherwise, infer suitable search keywords from the User Request and output a SEARCH action with those keywords.
- Output only one action. Do NOT include any explanations.

User Request:
{self.user_request}
        """.strip()

        time_line = TimeLine()
        record_index = len(self.records) + 1
        action_index = 1
        action_uid = gen_uid()
        record_prefix = f"{record_index:03}_act"
        action_prefix = f"{record_prefix}_{action_index:03}_{action_uid}"
        logger.info(f"[{record_index:03}] Act")

        llm_details = await PrimaryLLM.chat_with_text_detail(prompt)
        time_line.add("llm")
        raw_action = llm_details["content"]
        action = Action.from_raw_action(uid=action_uid, raw_action=raw_action, dom_nodes=[])
        logger.debug(f"[Action] Construct 1 action: [{action.type.name}]")

        action_screenshot = await page_screenshot(page)
        action_screenshot_draw = ImageDraw.Draw(action_screenshot)
        draw_text_label(action_screenshot_draw, text=action.type.name, position=(10, 10), font=self.font_18)
        action_screenshot_path = self.out_dir / f"{action_prefix}_action.jpg"
        action_screenshot.save(action_screenshot_path)
        execute_time = datetime.now()
        execute_result: ActionExecuteResult = {
            "success": True,
            "additional": await action.execute(page, cdp_session, self.memory),
        }

        result_screenshot = await page_screenshot(page)
        result_screenshot_path = self.out_dir / f"{action_prefix}_result.jpg"
        result_screenshot.save(result_screenshot_path)
        action_details = ActionDetails(
            raw_action=raw_action,
            execute_time=execute_time,
            execute_result=execute_result,
            action=action,
            action_screenshot_path=action_screenshot_path,
            result_screenshot_path=result_screenshot_path,
        )
        time_line.add(f"execute_{action_index:03}")

        record = ActRecord(
            index=record_index,
            action_details_list=[action_details],
            time_line=time_line,
            llm_details=llm_details,
            pruned_dom_repr="",
        )
        if Config.debug:
            record.save(self.out_dir)
        self.records.append(record)
        self.last_act_record = record

    async def act(
        self,
        page: Page,
        cdp_session: CDPSession,
        last_error: str | None = None,
    ):
        time_line = TimeLine()
        record_index = len(self.records) + 1
        record_prefix = f"{record_index:03}_act"

        assert self.last_planning_record is not None
        next_objective = self.last_planning_record.nearest_next_objective
        current_state = self.last_planning_record.current_state
        logger.info(f"[{record_index:03}] Act")

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

        if last_error:
            error_prompt = "## Error Message\nYour last action failed with the following error message:\n{last_error}\n"
        else:
            error_prompt = ""

        prompt = f"""
You are an AI agent designed to operate in an iterative loop to automate browser tasks.
The User Request represents the overall task objective and must always guide your decisions.
The Current State and Nearest Next Objective are the planning results of the last iteration.
Based on the Task Description, construct appropriate Actions to move the task toward the nearest next objective.

## Task Description
User Request: {self.user_request}
Current State: {current_state}
Nearest Next Objective: {next_objective}

## Page State
{viewport.get_page_info()}
Interactive Elements. Each element is prefixed with a unique [backend_node_id]:
{pruned_dom_repr}

## Available Actions
{Action.get_format_prompt()}

{error_prompt}
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
            action_prefix = f"{record_prefix}_{action_index:03}_{action_uid}"
            action_screenshot = await page_screenshot(page)
            action_screenshot_draw = ImageDraw.Draw(action_screenshot)

            try:
                # parse action
                action = Action.from_raw_action(uid=action_uid, raw_action=raw_action, dom_nodes=final_nodes)
                # record action
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
                execute_result: ActionExecuteResult = {
                    "success": True,
                    "additional": await action.execute(page, cdp_session, self.memory),
                }
            except ActionParseException as e:
                logger.warning(f"[Act] Parse {action.type.value} action failed: {e}")
                draw_text_label(action_screenshot_draw, text=action.type.name, position=(10, 10), font=self.font_18)
                draw_text_label(action_screenshot_draw, text=str(e), position=(10, 40), font=self.font_18)
                action_screenshot_path = self.out_dir / f"{action_prefix}_error.jpg"
                action_screenshot.save(action_screenshot_path)
                execute_result: ActionExecuteResult = {
                    "success": False,
                    "additional": str(e),
                }
            except ActionExecuteException as e:
                logger.warning(f"[Act] Execute {action.type.value} action failed: {e}")
                execute_result: ActionExecuteResult = {
                    "success": False,
                    "additional": str(e),
                }
                result_screenshot = await page_screenshot(page)
                result_screenshot_path = self.out_dir / f"{action_prefix}_error.jpg"
                result_screenshot.save(result_screenshot_path)

            if execute_result["success"]:
                result_screenshot = await page_screenshot(page)
                result_screenshot_path = self.out_dir / f"{action_prefix}_result.jpg"
                result_screenshot.save(result_screenshot_path)

            action_details_list.append(
                ActionDetails(
                    raw_action=raw_action,
                    execute_time=execute_time,
                    execute_result=execute_result,
                    action=action,
                    action_screenshot_path=action_screenshot_path,
                    result_screenshot_path=result_screenshot_path,
                )
            )
            time_line.add(f"execute_{action_index:03}")

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
        self.last_act_record = record
