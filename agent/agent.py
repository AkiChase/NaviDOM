import asyncio
from datetime import datetime
import itertools
import json
from pathlib import Path
import re
from typing import Awaitable, cast
from loguru import logger
from playwright.async_api import BrowserContext
from PIL import ImageFont, ImageDraw, Image, ImageChops

from agent.action import (
    Action,
    ActionDetails,
    ActionExecuteException,
    ActionExecuteResult,
    ActionParseException,
    ActionType,
)
from agent.config import Config
from agent.dom import DomCluster, DomNode, DomState
from agent.llm import PrimaryLLM, SecondaryLLM
from agent.record import ActRecord, ObservationRecord, PlanningRecord, Record, TimeLine
from agent.tab import TabManager
from agent.utils import (
    draw_text_label,
    format_seconds,
    format_time_delta,
    gen_uid,
    load_default_font,
    bg_colors,
    load_prompts,
    tab_screenshot,
)


class Agent:
    prompts_dict: dict[str, str]
    context: BrowserContext
    tab_manager: TabManager
    out_dir: Path
    user_request: str
    font_18: ImageFont.FreeTypeFont | ImageFont.ImageFont
    records: list[Record]
    last_planning_record: PlanningRecord | None
    last_act_record: ActRecord | None = None
    last_observation_record: ObservationRecord | None
    progress: list[str]

    def __init__(self, out_dir: Path, context: BrowserContext, user_request: str):
        self.prompts_dict = load_prompts()
        self.context = context
        self.tab_manager = TabManager(context)
        self.out_dir = out_dir
        self.user_request = user_request
        self.font_18 = load_default_font(18)
        self.records = []
        self.last_planning_record = None
        self.last_act_record = None
        self.last_observation_record = None
        self.progress = []

        out_dir.mkdir(parents=True, exist_ok=True)

    def get_formated_progress_history(self) -> str:
        if not self.progress:
            return "No entries."
        return "\n".join(self.progress)

    async def run(self, start_url: str | None = None):
        start_time = datetime.now()
        # close all tabs
        await self.tab_manager.reset_context_tabs()
        if start_url:
            logger.info(f"[Run] Start with URL: {start_url}")
            await self.tab_manager.front_tab.goto(start_url)
        else:
            # 初次导航
            before_act_screenshot = await tab_screenshot(self.tab_manager.front_tab)
            before_act_tabs_info = await self.tab_manager.get_tabs_info()
            await self.act_initial()
            await self.observation(before_act_screenshot, before_act_tabs_info)

        # 初次规划
        extract_data_task = await self.planning()
        # TODO 下一轮planning要等待extract_data_task完成

        iteration_times = 0
        while True:
            iteration_times += 1

            before_act_screenshot = await tab_screenshot(self.tab_manager.front_tab)
            before_act_tabs_info = await self.tab_manager.get_tabs_info()
            # Act
            await self.act()
            # Observation
            await self.observation(before_act_screenshot, before_act_tabs_info)
            # Planning
            await self.planning()

            assert self.last_planning_record is not None
            if self.last_planning_record.task_completed:
                # TODO 这里等待Extract（甚至其返回的Feedback）否则主动执行一次Feedback
                # logger.success(f"[Planning] Task completed: {self.last_planning_record.current_state}")
                break

            if iteration_times >= Config.max_iteration_times:
                logger.warning(
                    f"[Planning] Max iteration times reached, stop: {iteration_times}/{Config.max_iteration_times}"
                )
                break

        end_time = datetime.now()
        logger.info(f"[Time] Total time cost: {format_time_delta(start_time, end_time)}")
        logger.info(
            f"[Time] Per act time cost: {(end_time - start_time).total_seconds() / len([r for r in self.records if isinstance(r, ActRecord)]):.2f}s"
        )
        out = self.save_result((end_time - start_time).total_seconds())
        self.save_report(**out, out_dir=self.out_dir)

    def parse_response(
        self,
        content: str,
        pattern_list: list[re.Pattern | None],
    ):
        fields: list[str | None] = [None] * len(pattern_list)
        for i, pattern in enumerate(pattern_list):
            if pattern is None:
                continue
            match = pattern.search(content)
            if match is not None:
                fields[i] = match.group(1).strip()
                assert isinstance(fields[i], str)
            else:
                break

        return fields

    async def feedback(self):
        pass

    async def extract(self, req_data_found: str) -> asyncio.Task:
        # TODO 解析并视情况执行
        # TODO 如果执行了记得再执行一个Feedback（无所谓结束时机）应该返回这个feedback相关的Task，任务结束可能需要等待
        feed_back_task = asyncio.create_task(self.feedback())
        return feed_back_task

    async def planning(self):
        time_line = TimeLine()
        record_index = len(self.records) + 1
        logger.info(f"[{record_index:03}] Planning")

        if self.last_planning_record is None:
            last_task_state = "The first Planning. No Last Task State."
            last_act_goal = "The first Planning. No Last Act Goal."
        else:
            last_task_state = self.last_planning_record.task_state
            last_act_goal = self.last_planning_record.act_goal

        if self.last_act_record is None:
            last_act = "Navigate to the initial page (See Current Tabs for details)."
        else:
            last_act = []
            for index, d in enumerate(self.last_act_record.action_details_list, 1):
                if d.execute_result["success"]:
                    assert d.action is not None
                    last_act.append(f"{index}. {d.action.get_description()}")
                else:
                    last_act.append(
                        f"{index}. Failed to execute {d.raw_action}, reason: {d.execute_result['additional']}"
                    )
            last_act = "\n".join(last_act)

        if self.last_observation_record is None:
            last_act_observation = "The first Planning. No Last Act Observation."
        else:
            last_act_observation = self.last_observation_record.observation

        cur_tab = self.tab_manager.front_tab
        screenshot = await tab_screenshot(cur_tab)

        screenshot = await tab_screenshot(cur_tab)
        time_line.add("fetch")

        prompt = self.prompts_dict["planning"].format(
            user_request=self.user_request,
            progress_history="No entries.",
            last_task_state=last_task_state,
            last_act_goal=last_act_goal,
            last_act=last_act,
            last_act_observation=last_act_observation,
            current_tabs=self.tab_manager.get_tabs_info(),
        )

        new_progress_pattern = re.compile(r"<New Progress>(.*?)</New Progress>", re.DOTALL)
        requested_data_found_pattern = re.compile(r"<Requested Data Found>(.*?)</Requested Data Found>", re.DOTALL)
        task_state_pattern = re.compile(r"<Task State>(.*?)</Task State>", re.DOTALL)
        act_goal_pattern = re.compile(r"<Act Goal>(.*?)</Act Goal>", re.DOTALL)

        extract_data_task = None

        async def hook_extract_and_feedback(content: str):
            nonlocal extract_data_task
            if extract_data_task is not None:
                return
            requested_data_found = self.parse_response(content, [requested_data_found_pattern])[0]
            if requested_data_found is None:
                return
            extract_data_task = asyncio.create_task(self.extract(requested_data_found))

        llm_details = await PrimaryLLM.chat_with_image_detail(prompt, screenshot, hook=hook_extract_and_feedback)
        time_line.add("llm")

        content = llm_details["content"]
        fields = self.parse_response(
            content, [new_progress_pattern, requested_data_found_pattern, task_state_pattern, act_goal_pattern]
        )
        new_progress, requested_data_found, task_state, act_goal = fields
        assert new_progress is not None, "Failed to parse expected New Progress in LLM response"
        assert requested_data_found is not None, "Failed to parse expected Requested Data Found in LLM response"
        assert task_state is not None, "Failed to parse expected Task State in LLM response"
        assert act_goal is not None, "Failed to parse expected Act Goal in LLM response"
        task_completed = "TASK_FULLY_FINISHED" in task_state and "TASK_FULLY_FINISHED" in act_goal

        logger.debug(f"[New Progress] {new_progress}")
        logger.debug(f"[Requested Data Found] {requested_data_found}")
        logger.debug(f"[Task State] {task_state}")
        logger.debug(f"[Act Goal] {act_goal}")

        time_line.add("end")
        record = PlanningRecord(
            index=record_index,
            llm_details=llm_details,
            new_progress=new_progress,
            requested_data_found=requested_data_found,
            task_state=task_state,
            act_goal=act_goal,
            task_completed=task_completed,
            time_line=time_line,
        )
        record.save(self.out_dir)
        self.records.append(record)
        self.last_planning_record = record

        assert isinstance(extract_data_task, asyncio.Task), "Failed to create extract data task"
        return extract_data_task

    async def observation(self, before_act_screenshot: Image.Image, before_act_tabs_info: str):
        cur_tab = self.tab_manager.front_tab

        time_line = TimeLine()
        record_index = len(self.records) + 1

        assert self.last_act_record is not None

        after_act_screenshot = await tab_screenshot(cur_tab)
        after_act_tabs_info = await self.tab_manager.get_tabs_info()
        tab_changed = before_act_tabs_info != after_act_tabs_info
        ui_changed = ImageChops.difference(before_act_screenshot, after_act_screenshot).getbbox() is not None
        if tab_changed or ui_changed:
            logger.info(f"[{record_index:03}] Observation")
            success_action_descriptions = [
                d.action.get_description()
                for d in self.last_act_record.action_details_list
                if d.execute_result["success"] and d.action is not None
            ]
            if success_action_descriptions:
                actions_info = "\n".join([f"{index}. {d}" for index, d in enumerate(success_action_descriptions, 1)])
            else:
                actions_info = "No actions."

            if Config.debug:
                record_prefix = f"{record_index:03}_observation"
                before_act_screenshot.save(self.out_dir / f"{record_prefix}_debug_before.jpg")
                after_act_screenshot.save(self.out_dir / f"{record_prefix}_debug_after.jpg")

            time_line.add("fetch")
            prompt = f"""
You are an assistant with strong visual analysis skills.
You are given:
- Screenshot 1: previous active tab page BEFORE the actions
- Screenshot 2: current active tab page AFTER the actions
- Browser tabs information before the actions:
{before_act_tabs_info}
- Browser tabs information after the actions:
{after_act_tabs_info}
- A brief description of the actions that were performed:
{actions_info}

Based only on what you see, write one concise paragraph that:
- Describes the main visual differences between the two screenshots (what appeared, disappeared, moved, or changed state, e.g., dropdowns, modals, buttons).
- If there are noticeable changes in the browser tab information between the two states, include them; otherwise, do not mention them.
- Explains, at a high level, what result is reflected on the tab page and what the user can do on current active tab page.

Important constraints:
- Do NOT infer user intent or speculate on why the actions were taken.
- Do NOT restate or interpret the actions themselves.
- Use the action description only as background context to better understand the visual transition, not as evidence.
- Focus strictly on observable UI changes and the resulting tab page state.
- Keep the response compact and tightly written in a single paragraph
            """.strip()
            llm_details = await SecondaryLLM.chat_with_image_list_detail(
                prompt, [before_act_screenshot, after_act_screenshot]
            )
            observation = llm_details["content"]
            logger.debug(f"[Result] {llm_details['content']}")
            time_line.add("llm")
        else:
            observation = f"Browser tabs and UI page not changed"
            logger.debug(f"[Result] {observation}")
            llm_details = None
            time_line.add("fetch")

        time_line.add("end")
        record = ObservationRecord(
            index=record_index,
            llm_details=llm_details,
            observation=observation,
            time_line=time_line,
        )
        record.save(self.out_dir)
        self.records.append(record)
        self.last_observation_record = record

    async def act_initial(self):
        cur_tab = self.tab_manager.front_tab

        prompt = f"""
You are a web AI agent designed to automate browser tasks.
Your task is to analyze the User Request and determine the initial browser tab page to navigate to.

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
        action = Action.from_raw_action(
            uid=action_uid, raw_action=raw_action, dom_nodes=[], tab_manager=self.tab_manager
        )
        logger.debug(f"[Action] Construct 1 action: [{action.type.name}]")

        action_screenshot = await tab_screenshot(cur_tab)
        action_screenshot_draw = ImageDraw.Draw(action_screenshot)
        draw_text_label(action_screenshot_draw, text=action.type.name, position=(10, 10), font=self.font_18)
        action_screenshot_path = self.out_dir / f"{action_prefix}_action.jpg"
        action_screenshot.save(action_screenshot_path)
        execute_result: ActionExecuteResult = {
            "success": True,
            "additional": await action.execute(cur_tab, self.tab_manager),
        }

        result_screenshot = await tab_screenshot(cur_tab)
        result_screenshot_path = self.out_dir / f"{action_prefix}_result.jpg"
        result_screenshot.save(result_screenshot_path)
        action_details = ActionDetails(
            raw_action=raw_action,
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
        record.save(self.out_dir)
        self.records.append(record)
        self.last_act_record = record

    async def act(self):
        record_index = len(self.records) + 1
        record_prefix = f"{record_index:03}_act"
        cur_tab = self.tab_manager.front_tab
        time_line = TimeLine()
        logger.info(f"[{record_index:03}] Act")

        assert self.last_planning_record is not None
        act_goal = self.last_planning_record.act_goal
        task_state = self.last_planning_record.task_state

        dom_state = await DomState.load_dom_state(cur_tab)
        screenshot = await tab_screenshot(cur_tab)
        dom = dom_state.dom
        viewport = dom_state.viewport
        time_line.add("fetch")

        if Config.debug:
            # 最细粒度元素可视化
            atomic_screenshot = screenshot.copy()
            atomic_screenshot_draw = ImageDraw.Draw(atomic_screenshot)
            dom.draw_bounds(atomic_screenshot_draw, draw_id=True)
            atomic_screenshot.save(self.out_dir / f"{record_prefix}_debug_1_atomic.jpg")

        interactive_nodes = dom.extract_interactive_nodes(dom)

        if Config.debug:
            # 交互粒度元素可视化
            interactive_screenshot = screenshot.copy()
            interactive_screenshot_draw = ImageDraw.Draw(interactive_screenshot)
            for node in interactive_nodes:
                node.draw_bounds(
                    interactive_screenshot_draw,
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
        clusters = DomCluster.cluster_construct(interactive_nodes, alpha=0.45, distance_threshold=0.4)
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
                        outline=color,
                        width=width,
                        draw_id=True,
                        recursive=False,
                        max_bounds=True,
                    )

                rect = DomCluster.cluster_covered_xyxy(cluster)
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
                rect = DomCluster.cluster_covered_xyxy(cluster)
                related_rects.append(rect)
                cluster_region = screenshot.crop(rect)
                cluster_region_draw = ImageDraw.Draw(cluster_region)
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
                    cluster_region_draw.rectangle(node_bounds, outline="red", width=2)
                if Config.debug:
                    cluster_region.save(self.out_dir / f"{record_prefix}_debug_4_cluster_{index + 1}.jpg")
                related_tasks.append(DomCluster.determine_cluster_related(cluster, cluster_region, task=act_goal))
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
                # TODO 这里需要特殊处理，不要再fallback了，应该是在prompt中提示无相关Node然后available_actions中不包含Click之类Action
            else:
                # 合并边界存在重叠的区域，用于构建紧凑的UI图
                merged_rects = DomCluster.cluster_merge_overlapped(related_cluster)
                # 构建压缩后的UI图像
                final_screenshot = DomCluster.cluster_image_layout_compaction(screenshot, merged_rects, default_gap=1)
                final_nodes: list[DomNode] = list(
                    itertools.chain.from_iterable(cluster for cluster, _ in related_cluster)
                )

        # 去除fallback
        if fallback:
            # 回退到无语义剪枝
            final_screenshot = screenshot.copy()
            final_nodes = interactive_nodes
            final_screenshot_draw = ImageDraw.Draw(final_screenshot)
            for node in final_nodes:
                node.draw_bounds(
                    final_screenshot_draw,
                    outline="red",
                    width=1,
                    draw_id=False,
                    recursive=False,
                    max_bounds=True,
                )
        if Config.debug:
            final_screenshot.save(self.out_dir / f"{record_prefix}_debug_5_final.jpg")

        pruned_dom_repr = "\n\n".join(
            [node.convert_to_repr_node().get_human_tree_repr(no_end=True) for node in final_nodes]
        )
        time_line.add("pruning")

        prompt = f"""
You are a web AI agent designed to operate in an iterative loop to automate browser tasks.
The User Request represents the overall task objective and must always guide your decisions.
The Current State and Nearest Next Objective are the planning results of the last iteration.
Based on the Task Description, construct appropriate Actions to move the task toward the nearest next objective.

## Available Actions
{Action.get_format_prompt(include_types=None, exclude_types=[ActionType.Search])}

## Requirements
- You may construct one or more Actions. If multiple Actions are provided, they will be executed sequentially.
- All Actions MUST strictly follow the formats listed in Available Actions. Each Action must be output on its own line.
- Do NOT include explanations or any additional text.

## Task Description
User Request: {self.user_request}
Current State: {task_state}
{action_reflection}
Nearest Next Objective: {act_goal}

## Browser Tabs
{await self.tab_manager.get_tabs_info()}

## Current Tab
- {self.tab_manager.get_tab_id_info()}
- {viewport.get_viewport_scroll_info()}
- Interactive Nodes. Each element is prefixed with a unique [node_id]. If action contains target node, it must be one of the following:
{pruned_dom_repr}
""".strip()

        llm_action_detail = await PrimaryLLM.chat_with_image_detail(prompt, final_screenshot)
        time_line.add("llm")
        llm_action_res = llm_action_detail["content"]

        action_details_list = []
        raw_action_list = [line for line in llm_action_res.strip().split("\n") if line.strip()]
        raw_action_types = [raw_action.split(",", maxsplit=1)[0].strip() for raw_action in raw_action_list]
        logger.debug(f"[Action] Generate {len(raw_action_list)} actions: [{', '.join(raw_action_types)}]")

        old_tab_id = self.tab_manager.cur_tab_id
        for action_index, raw_action in enumerate(raw_action_list, 1):
            # tab changed, skip rest actions
            tab_changed = self.tab_manager.cur_tab_id != old_tab_id
            last_action_failed = (
                False if not action_details_list else action_details_list[-1].execute_result["success"] == False
            )

            if tab_changed or last_action_failed:
                rest_raw_action_list = raw_action_list[action_index - 1 :]
                error_reason = "Tab changed" if tab_changed else "Last action failed"
                error_msg = f"{error_reason}, stop executing the remaining actions"
                logger.debug(
                    f"[Action] {error_reason}, stop executing the remaining {len(rest_raw_action_list)} actions"
                )
                action_screenshot = await tab_screenshot(cur_tab)
                draw_text_label(action_screenshot_draw, text=error_msg, position=(10, 40), font=self.font_18)
                result_screenshot_path = action_screenshot_path = (
                    self.out_dir
                    / f"{record_prefix}_{action_index:03}-{action_index+len(rest_raw_action_list)-1:03}_error.jpg"
                )
                action_screenshot.save(action_screenshot_path)
                for rest_raw_action in rest_raw_action_list:
                    action_details_list.append(
                        ActionDetails(
                            raw_action=rest_raw_action,
                            execute_result={
                                "success": False,
                                "additional": error_msg,
                            },
                            action=None,
                            action_screenshot_path=action_screenshot_path,
                            result_screenshot_path=result_screenshot_path,
                        )
                    )
                break

            action_uid = gen_uid()
            action_prefix = f"{record_prefix}_{action_index:03}_{action_uid}"
            logger.debug(f"[Action] {action_index}. {raw_action}")
            action_screenshot = await tab_screenshot(cur_tab)
            action_screenshot_draw = ImageDraw.Draw(action_screenshot)
            draw_text_label(
                action_screenshot_draw,
                text=raw_action if len(raw_action) < 50 else f"{raw_action[:50]}...",
                position=(10, 10),
                font=self.font_18,
            )
            try:
                # parse action
                action = Action.from_raw_action(
                    uid=action_uid, raw_action=raw_action, dom_nodes=final_nodes, tab_manager=self.tab_manager
                )
                # draw action target
                if action.target is not None:
                    action.target.draw_bounds(
                        action_screenshot_draw,
                        outline="red",
                        width=3,
                        draw_id=True,
                        recursive=False,
                        max_bounds=True,
                    )
                action_screenshot_path = self.out_dir / f"{action_prefix}_action.jpg"
                action_screenshot.save(action_screenshot_path)

                # execute action
                execute_result: ActionExecuteResult = {
                    "success": True,
                    "additional": await action.execute(cur_tab, self.tab_manager),
                }
                # update latest tab
                cur_tab = self.tab_manager.front_tab
                result_screenshot = await tab_screenshot(cur_tab)
                result_screenshot_path = self.out_dir / f"{action_prefix}_result.jpg"
                result_screenshot.save(result_screenshot_path)
            except ActionParseException as e:
                logger.warning(f"[Act] Parse action failed, {e}. Action: {raw_action}")
                draw_text_label(action_screenshot_draw, text=str(e), position=(10, 40), font=self.font_18)
                action_screenshot_path = self.out_dir / f"{action_prefix}_error.jpg"
                action_screenshot.save(action_screenshot_path)
                result_screenshot_path = action_screenshot_path  # the same
                execute_result: ActionExecuteResult = {
                    "success": False,
                    "additional": str(e),
                }
                action = None
            except ActionExecuteException as e:
                assert action is not None
                logger.warning(f"[Act] Execute {action.type.value} action failed: {e}")
                execute_result: ActionExecuteResult = {
                    "success": False,
                    "additional": str(e),
                }
                # update latest tab
                cur_tab = self.tab_manager.front_tab

                result_screenshot = await tab_screenshot(cur_tab)
                result_screenshot_path = self.out_dir / f"{action_prefix}_error.jpg"
                result_screenshot.save(result_screenshot_path)

            action_details_list.append(
                ActionDetails(
                    raw_action=raw_action,
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
        record.save(self.out_dir)
        self.records.append(record)
        self.last_act_record = record

    def save_result(self, total_time_cost: float):
        user_request = self.user_request
        token = PrimaryLLM.token_dict
        for k, v in SecondaryLLM.token_dict.items():
            if k not in token:
                token[k] = v
            else:
                token[k]["completion_tokens"] += v["completion_tokens"]
                token[k]["prompt_tokens"] += v["prompt_tokens"]
                token[k]["total_tokens"] += v["total_tokens"]

        success = self.last_planning_record is not None and self.last_planning_record.task_completed
        if success:
            assert self.last_planning_record is not None
            result = self.last_planning_record.current_state
        else:
            result = "Task failed."

        records = []
        for record in self.records:
            if isinstance(record, ActRecord):
                prompt_tokens = record.llm_details["prompt_tokens"]
                completion_tokens = record.llm_details["completion_tokens"]
                for d in record.action_details_list:
                    additional = d.execute_result["additional"]
                    if (
                        isinstance(additional, dict)
                        and "prompt_tokens" in additional
                        and "completion_tokens" in additional
                    ):
                        prompt_tokens += additional["prompt_tokens"]
                        completion_tokens += additional["completion_tokens"]

                records.append(
                    {
                        "type": "act",
                        "actions": [
                            {
                                "raw": d.raw_action,
                                "description": d.action.get_description() if d.action is not None else "Parse Failed",
                                "success": d.execute_result["success"],
                                "action_screenshot_name": d.action_screenshot_path.name,
                                "result_screenshot_name": d.result_screenshot_path.name,
                                "additional": (
                                    d.execute_result["additional"]["content"]
                                    if isinstance(d.execute_result["additional"], dict)
                                    else d.execute_result["additional"]
                                ),
                            }
                            for d in record.action_details_list
                        ],
                        "token_usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                        "time_cost": round(record.time_line.total_time(), 4),
                    }
                )
            elif isinstance(record, ObservationRecord):
                token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
                if record.llm_details is not None:
                    token_usage["prompt_tokens"] = record.llm_details["prompt_tokens"]
                    token_usage["completion_tokens"] = record.llm_details["completion_tokens"]

                records.append(
                    {
                        "type": "observation",
                        "observation": record.observation,
                        "time_cost": round(record.time_line.total_time(), 4),
                        "token_usage": token_usage,
                    }
                )
            elif isinstance(record, PlanningRecord):
                if record.task_completed:
                    records.append(
                        {
                            "type": "planning",
                            "task_completed": record.task_completed,
                            "task_result": record.current_state,
                            "time_cost": round(record.time_line.total_time(), 4),
                            "token_usage": {
                                "prompt_tokens": record.llm_details["prompt_tokens"],
                                "completion_tokens": record.llm_details["completion_tokens"],
                            },
                        }
                    )
                else:
                    records.append(
                        {
                            "type": "planning",
                            "current_state": record.current_state,
                            "nearest_next_objective": record.nearest_next_objective,
                            "unfinished_content": record.unfinished_content,
                            "task_completed": record.task_completed,
                            "time_cost": round(record.time_line.total_time(), 4),
                            "token_usage": {
                                "prompt_tokens": record.llm_details["prompt_tokens"],
                                "completion_tokens": record.llm_details["completion_tokens"],
                            },
                        }
                    )

        time_cost = {
            "total_time": round(total_time_cost, 4),
            "act_time": 0,
            "observation_time": 0,
            "planning_time": 0,
        }
        for record in self.records:
            if isinstance(record, ActRecord):
                time_cost["act_time"] += record.time_line.total_time()
            elif isinstance(record, ObservationRecord):
                time_cost["observation_time"] += record.time_line.total_time()
            elif isinstance(record, PlanningRecord):
                time_cost["planning_time"] += record.time_line.total_time()
        time_cost["act_time"] = round(time_cost["act_time"], 4)
        time_cost["observation_time"] = round(time_cost["observation_time"], 4)
        time_cost["planning_time"] = round(time_cost["planning_time"], 4)

        out = {
            "user_request": user_request,
            "token": token,
            "records": records,
            "time_cost": time_cost,
            "success": success,
            "result": result,
        }

        with open(self.out_dir / "result.json", "w") as f:
            json.dump(
                out,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return out

    @staticmethod
    def save_report(
        user_request: str, success: bool, result: str, token: dict, time_cost: dict, records: list, out_dir: Path
    ):
        report_lines: list[str] = []

        # Header
        report_lines.append("# Task Execution Report")
        report_lines.append("")

        # Part 1: Overall Summary
        report_lines.append("## 1. Overall Summary")
        report_lines.append("")
        # User Request
        report_lines.append("### User Request")
        report_lines.append("")
        report_lines.append(f"> {user_request.strip()}")
        report_lines.append("")
        # Success
        report_lines.append("### Task Result")
        report_lines.append(f"- **Success**: {'✅ Yes' if success else '❌ No'}")
        report_lines.append(f"- **Result**: {result}")
        report_lines.append("")
        # Token usage
        report_lines.append("### Token Usage")
        report_lines.append("")
        report_lines.append("| Model | Prompt Tokens | Completion Tokens | Total Tokens |")
        report_lines.append("|------|---------------|-------------------|--------------|")
        for model_name, t in token.items():
            report_lines.append(
                f"| {model_name} | {t.get('prompt_tokens', 0)} | "
                f"{t.get('completion_tokens', 0)} | {t.get('total_tokens', 0)} |"
            )
        report_lines.append("")
        # Time cost
        report_lines.append("### Time Cost (seconds)")
        report_lines.append("")
        report_lines.append("| Type | Time |")
        report_lines.append("|------|------|")
        report_lines.append(f"| Total | {time_cost['total_time']} |")
        report_lines.append(f"| Act | {time_cost['act_time']} |")
        report_lines.append(f"| Observation | {time_cost['observation_time']} |")
        report_lines.append(f"| Planning | {time_cost['planning_time']} |")
        report_lines.append("")

        # Part 2: Execution Records
        report_lines.append("## 2. Execution Records")
        report_lines.append("")

        for idx, record in enumerate(records, start=1):
            record_type = record["type"]
            token_usage = record["token_usage"]
            # Act Record
            if record_type == "act":
                report_lines.append(f"### {idx}. Act")
                report_lines.append(f"- **Time Cost**: {record['time_cost']}s")
                report_lines.append(
                    f"- **Token Usage**: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}"
                )
                report_lines.append("")
                report_lines.append("**Actions:**")
                for action_idx, action in enumerate(record["actions"], start=1):
                    success_flag = "✅" if action["success"] else "❌"
                    report_lines.append(f"{action_idx}. **Action {action_idx}**")
                    report_lines.append(f"    - **Raw**: {action['raw']}")
                    report_lines.append(f"    - **Description**: {action['description']}")
                    report_lines.append(f"    - **Success**: {success_flag}")
                    if action.get("additional") is not None:
                        report_lines.append(f"    - **Additional**: {action['additional']}")

                    # 并排显示截图
                    action_img = f"![Action]({action['action_screenshot_name']})"
                    result_img = f"![Result]({action['result_screenshot_name']})" if action["success"] else ""
                    if action["success"]:
                        # 使用 HTML 表格并排
                        img_html = f"<table><tr><td>{action_img}</td><td>{result_img}</td></tr></table>"
                        report_lines.append(f"    - **Screenshots**: {img_html}")
                    else:
                        report_lines.append(f"    - **Action Screenshot**: {action_img}")

                report_lines.append("")
            # Observation Record
            elif record_type == "observation":
                report_lines.append(f"### {idx}. Observation")
                report_lines.append(f"- **Time Cost**: {record['time_cost']}s")
                report_lines.append(
                    f"- **Token Usage**: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}"
                )
                report_lines.append("")
                report_lines.append("**Observation**:")
                report_lines.append(record["observation"])
                report_lines.append("")
            # Planning Record
            elif record_type == "planning":
                report_lines.append(f"### {idx}. Planning")
                report_lines.append(f"- **Time Cost**: {record['time_cost']}s")
                report_lines.append(
                    f"- **Token Usage**: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}"
                )
                report_lines.append(f"- **Task Completed**: {record['task_completed']}")
                report_lines.append("")

                if record["task_completed"]:
                    report_lines.append("**Task Result:**")
                    report_lines.append(record.get("task_result", ""))
                else:
                    report_lines.append("**Current State:**")
                    report_lines.append(record.get("current_state", ""))
                report_lines.append("")

                if not record["task_completed"]:
                    report_lines.append("**New Progress:**")
                    report_lines.append(record.get("new_progress", ""))
                    report_lines.append("")

                    report_lines.append("**Nearest Next Objective:**")
                    report_lines.append(record.get("nearest_next_objective", ""))
                    report_lines.append("")

                    report_lines.append("")
                    report_lines.append("**Unfinished Content:**")
                    report_lines.append(record.get("unfinished_content", ""))
                    report_lines.append("")
                report_lines.append("")

        # Write md file
        report_path = out_dir / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        logger.info("Report saved to {}", report_path)
