import asyncio
from collections import defaultdict
from datetime import datetime
import itertools
import json
from pathlib import Path
import re
from loguru import logger
from matplotlib import pyplot as plt
from playwright.async_api import BrowserContext
from PIL import ImageFont, ImageDraw, Image

from agent.action import (
    Action,
    ActionDetails,
    ActionExecuteException,
    ActionExecuteResult,
    ActionParseException,
    ActionType,
)
from agent.config import Config
from agent.dom import DomCluster, DomNode, DomState, Viewport
from agent.llm import ChatImageDetails, PrimaryLLM, SecondaryLLM
from agent.record import (
    ActRecord,
    ExtractionRecord,
    FeedbackRecord,
    ObservationRecord,
    PlanningRecord,
    PruningDetails,
    Record,
    TimeLine,
)
from agent.tab import TabManager
from agent.utils import (
    draw_text_label,
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
    last_extraction_record: ExtractionRecord | None
    last_feedback_record: FeedbackRecord | None
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
        self.last_extraction_record = None
        self.last_feedback_record = None
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
            await self.act_initial()
            await self.observation(before_act_screenshot)

        # 初次规划
        extraction_task = await self.planning()

        iteration_times = 0
        while True:
            iteration_times += 1
            assert self.last_planning_record is not None
            if self.last_planning_record.task_completed:
                # ensure extraction and feedback task is done
                feedback_task = await extraction_task
                await feedback_task

                assert self.last_feedback_record is not None
                if self.last_feedback_record.all_done:
                    logger.success(f"[Planning] Task completed")
                    break
                else:
                    logger.warning(f"[Feedback] Task not completed")
                    extraction_task = await self.planning()
                    if self.last_planning_record.task_completed:
                        feedback_task = await extraction_task
                        await feedback_task
                        logger.success(f"[Planning] Task completed")
                        break

            if iteration_times > Config.max_iteration_times:
                # ensure extraction and feedback task is done
                feedback_task = await extraction_task
                await feedback_task

                logger.warning(
                    f"[Planning] Max iteration times reached, stop: {iteration_times-1}/{Config.max_iteration_times}"
                )
                break

            before_act_screenshot = await tab_screenshot(self.tab_manager.front_tab)
            # Act
            await self.act()
            # Observation
            await self.observation(before_act_screenshot)
            # Planning
            await extraction_task  # ensure extraction task (of last planning) is done before next planning
            # TODO 如果progress过长（token达到某个阈值？）后可以在背景任务中提取有效、去除重复（相对user request)的progress（记录旧的长度，提取成功后，将旧的那几条替换成新的）
            extraction_task = await self.planning()

        end_time = datetime.now()
        logger.info(f"[Time] Total time cost: {format_time_delta(start_time, end_time)}")
        logger.info(
            f"[Time] Per act time cost: {(end_time - start_time).total_seconds() / len([r for r in self.records if isinstance(r, ActRecord)]):.2f}s"
        )
        if self.last_feedback_record is not None:
            logger.info(f"[Feedback]:\n{self.last_feedback_record.repr}")
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
        time_line = TimeLine()
        record_index = len(self.records)
        self.records.append(Record(record_index, time_line))
        logger.info(f"[{record_index:03}] Feedback")

        if self.last_feedback_record is None:
            last_todo_list = "First time feedback, no previous TODO list."
        else:
            last_todo_list = self.last_feedback_record.feedback

        prompt = self.prompts_dict["feedback"].format(
            user_request=self.user_request,
            last_todo_list=last_todo_list,
            agent_history=self.get_formated_progress_history(),
        )
        llm_detail = await SecondaryLLM.chat_with_text_detail(prompt)
        content = "\n".join(line for line in llm_detail["content"].splitlines() if line.strip())
        all_done = content.find("IN PROGRESS") == -1 and content.find("NOT STARTED") == -1
        repr = content
        status_emoji = {
            "DONE": "✅",
            "IN PROGRESS": "⏳",
            "NOT STARTED": "🔴",
        }
        for k, v in status_emoji.items():
            repr = repr.replace(k, f"{v} {k}")

        logger.info(f"[Feedback] Detail:\n{repr}")
        time_line.add("llm")

        record = FeedbackRecord(
            index=record_index,
            llm_details=llm_detail,
            feedback=content,
            repr=repr,
            all_done=all_done,
            time_line=time_line,
        )
        record.save(self.out_dir)
        self.records[record_index] = record
        self.last_feedback_record = record

    async def extraction(self, new_progress: str, req_data_found: str) -> asyncio.Task[None] | None:
        if req_data_found.find("[FOUND]") != -1:
            time_line = TimeLine()
            record_index = len(self.records)
            self.records.append(Record(record_index, time_line))
            logger.info(f"[{record_index:03}] Extraction")

            cur_tab = self.tab_manager.front_tab

            dom_state = await DomState.load_dom_state(cur_tab)
            dom_repr = dom_state.dom.convert_to_repr_node(skip_omit=True)
            screenshot = await tab_screenshot(cur_tab)
            time_line.add("fetch")

            prompt = self.prompts_dict["extraction"].format(
                last_progress=new_progress,
                last_requested_data_found=req_data_found,
                html=dom_repr,
            )
            llm_detail = await SecondaryLLM.chat_with_image_detail(prompt, screenshot)
            time_line.add("llm")

            content = llm_detail["content"]
            logger.debug(f"[Extraction] {content}")
            self.progress.append(content)
            record = ExtractionRecord(
                index=record_index,
                llm_details=llm_detail,
                data=content,
                time_line=time_line,
            )
            record.save(self.out_dir)
            self.records[record_index] = record
        elif req_data_found.find("[NOT_FOUND]") != -1:
            logger.debug(f"[Extraction] No new data")
        else:
            logger.error(f"[Extraction] Invalid Requested Data Found format: {req_data_found}")

        feed_back_task = asyncio.create_task(self.feedback())
        return feed_back_task

    async def planning(self) -> asyncio.Task[asyncio.Task[None]]:
        time_line = TimeLine()
        record_index = len(self.records)
        self.records.append(Record(record_index, time_line))
        record_prefix = f"{record_index:03}_planning"
        logger.info(f"[{record_index:03}] Planning")

        if self.last_planning_record is None:
            last_task_state = "The first Planning. No Last Task State."
            last_act_goal = "The first Planning. No Last Act Goal."
            last_planning = f"- Task State: {last_task_state}\n- Act Goal: {last_act_goal}"
        else:
            if self.last_planning_record.task_completed:
                assert self.last_feedback_record is not None
                last_planning = (
                    "Previous Planning step concluded that the task (User Request) was fully finished, "
                    "but the Feedback Agent noted that some parts of the User Request are still incomplete. "
                    "Please consider the Feedback Agent's comments:\n"
                    f"{self.last_feedback_record.feedback}"
                )
            else:
                last_task_state = self.last_planning_record.task_state
                last_act_goal = self.last_planning_record.act_goal
                last_planning = f"- Task State: {last_task_state}\n- Act Goal: {last_act_goal}"

        if self.last_act_record is None:
            last_act = "Navigate to the initial page (See Current Tabs for details)."
        else:
            last_act = self.last_act_record.get_actions_descriptions()

        if self.last_observation_record is None:
            last_act_observation = "The first Planning. No Last Act Observation."
        else:
            last_act_observation = self.last_observation_record.observation

        cur_tab = self.tab_manager.front_tab

        screenshot = await tab_screenshot(cur_tab)
        tabs_info = await self.tab_manager.get_tabs_info()
        viewport = await Viewport.from_tab(cur_tab)
        current_tabs = f"{tabs_info}\n{viewport.get_viewport_scroll_info()}"
        time_line.add("fetch")

        prompt = self.prompts_dict["planning"].format(
            user_request=self.user_request,
            progress_history="No entries.",
            last_planning=last_planning,
            last_act=last_act,
            last_act_observation=last_act_observation,
            current_tabs=current_tabs,
        )

        new_progress_pattern = re.compile(r"New Progress:\s*(.*?)\s*Requested Data Found:", re.DOTALL)
        requested_data_found_pattern = re.compile(r"Requested Data Found:\s*(.*?)\s*Task State:", re.DOTALL)
        task_state_pattern = re.compile(r"Task State:\s*(.*?)\s*Act Goal:", re.DOTALL)
        act_goal_pattern = re.compile(r"Act Goal:\s*(.*)$", re.DOTALL)  # 到文本末尾

        extraction_task = None

        async def hook_extraction_and_feedback(content: str):
            nonlocal extraction_task
            if extraction_task is not None:
                return
            new_progress, requested_data_found = self.parse_response(
                content, [new_progress_pattern, requested_data_found_pattern]
            )
            if new_progress is None or requested_data_found is None:
                return
            logger.debug(f"[Planning][New Progress] {new_progress}")
            self.progress.append(content)
            logger.debug(f"[Planning][Requested Data Found] {requested_data_found}")
            extraction_task = asyncio.create_task(self.extraction(new_progress, requested_data_found))

        final_screenshot = screenshot.resize(
            (int(screenshot.width * 0.75), int(screenshot.height * 0.75)), resample=Image.Resampling.LANCZOS
        )
        final_screenshot_path = self.out_dir / f"{record_prefix}.jpg"
        final_screenshot.save(final_screenshot_path)
        llm_details = await PrimaryLLM.chat_with_image_detail(
            prompt, final_screenshot, hook=hook_extraction_and_feedback
        )
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
        task_completed = "TASK_FULLY_FINISHED" in task_state or "TASK_FULLY_FINISHED" in act_goal

        logger.debug(f"[Planning][Task State] {task_state}")
        logger.debug(f"[Planning][Act Goal] {act_goal}")

        time_line.add("end")
        record = PlanningRecord(
            index=record_index,
            llm_details=llm_details,
            new_progress=new_progress,
            requested_data_found=requested_data_found,
            task_state=task_state,
            act_goal=act_goal,
            task_completed=task_completed,
            screenshot_path=final_screenshot_path,
            time_line=time_line,
        )
        record.save(self.out_dir)
        self.records[record_index] = record
        self.last_planning_record = record

        assert isinstance(extraction_task, asyncio.Task), "Failed to create extraction data task"
        return extraction_task

    async def observation(self, before_act_screenshot: Image.Image):
        time_line = TimeLine()
        record_index = len(self.records)
        self.records.append(Record(record_index, time_line))
        assert self.last_act_record is not None
        logger.info(f"[{record_index:03}] Observation")

        after_act_screenshot = await tab_screenshot(self.tab_manager.front_tab)
        actions_info = self.last_act_record.get_actions_descriptions()

        if Config.debug:
            record_prefix = f"{record_index:03}_observation"
            before_act_screenshot.save(self.out_dir / f"{record_prefix}_debug_before.jpg")
            after_act_screenshot.save(self.out_dir / f"{record_prefix}_debug_after.jpg")

        time_line.add("fetch")
        prompt = self.prompts_dict["observation"].format(
            act_goal=self.last_act_record.act_goal,
            action_execution_description=actions_info,
        )
        llm_details = await SecondaryLLM.chat_with_image_list_detail(
            prompt, [before_act_screenshot, after_act_screenshot]
        )
        observation = llm_details["content"]
        logger.debug(f"[Observation] {llm_details['content']}")
        time_line.add("llm")

        time_line.add("end")
        record = ObservationRecord(
            index=record_index,
            llm_details=llm_details,
            observation=observation,
            time_line=time_line,
        )
        record.save(self.out_dir)
        self.records[record_index] = record
        self.last_observation_record = record

    async def act_initial(self):
        cur_tab = self.tab_manager.front_tab

        prompt = f"""
You are a web AI agent designed to automate browser tasks.
Your task is to analyze the User Request and determine the initial browser tab page to navigate to.

Available Actions:
{Action.get_available_actions_prompt([ActionType.Navigate, ActionType.Search])}

Requirements:
- If the user request explicitly specifies a URL to start from, output a NAVIGATE action with that URL.
- If the user request asks to start from a clearly identifiable, well-known website, output a NAVIGATE action with that website's URL.
- Otherwise, infer suitable search keywords from the User Request and output a SEARCH action with those keywords.
- Output only one action. Do NOT include any explanations.

User Request:
{self.user_request}
        """.strip()

        time_line = TimeLine()
        record_index = len(self.records)
        self.records.append(Record(record_index, time_line))
        action_index = 1
        action_uid = gen_uid()
        record_prefix = f"{record_index:03}_act"
        action_prefix = f"{record_prefix}_{action_index:03}_{action_uid}"
        logger.info(f"[{record_index:03}] Act")

        llm_details = await SecondaryLLM.chat_with_text_detail(prompt)
        time_line.add("llm")
        raw_action = llm_details["content"]
        action = Action.from_raw_action(
            uid=action_uid, raw_action=raw_action, dom_nodes=[], tab_manager=self.tab_manager
        )
        logger.debug(f"[Act] Construct 1 action: [{action.type.name}]")

        action_screenshot = await tab_screenshot(cur_tab)
        action_screenshot_draw = ImageDraw.Draw(action_screenshot)
        draw_text_label(action_screenshot_draw, text=action.type.name, position=(10, 10), font=self.font_18)
        action_screenshot_path = self.out_dir / f"{action_prefix}_action.jpg"
        action_screenshot.save(action_screenshot_path)
        old_tab_info = await self.tab_manager.get_cur_tab_info()
        action_result = await action.execute(cur_tab, self.tab_manager)
        new_tab_info = await self.tab_manager.get_cur_tab_info()
        execute_result: ActionExecuteResult = {
            "success": True,
            "result": action_result,
            "tab_changed_info": self.tab_manager.compare_tab_info(new_tab_info, old_tab_info),
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
        pruning_details: PruningDetails = {
            "time": 0,
            "model": SecondaryLLM.image_model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "result": [],
            "token_reduction": 0,
        }

        record = ActRecord(
            index=record_index,
            action_details_list=[action_details],
            pruning_details=pruning_details,
            time_line=time_line,
            llm_details=llm_details,
            interactive_nodes_repr="",
            act_goal="Navigate to the initial tab page.",
        )
        record.save(self.out_dir)
        self.records[record_index] = record
        self.last_act_record = record

    async def act(self):
        record_index = len(self.records)
        time_line = TimeLine()
        self.records.append(Record(record_index, time_line))
        record_prefix = f"{record_index:03}_act"
        cur_tab = self.tab_manager.front_tab
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
        logger.debug("[Act] Interactive node count: {}", len(interactive_nodes))

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
        # 筛除尺寸过小的无效聚类
        clusters: list[list[DomNode]] = []
        for c in DomCluster.cluster_construct(interactive_nodes, alpha=0.45, distance_threshold=0.45):
            rect = DomCluster.cluster_covered_xyxy(c, viewport)
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            min_side = min(w, h)
            max_side = max(w, h)
            if min_side < 10 and max_side / max(min_side, 1e-6) > 20:
                continue
            clusters.append(c)

        assert len(clusters) >= 1, "No valid cluster found"
        logger.debug("[Act][Cluster] Cluster count: {}", len(clusters))
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

                rect = DomCluster.cluster_covered_xyxy(cluster, viewport)
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

        # 基于LLM过滤与任务不相关的聚类区域
        related_tasks = []
        related_rects = []
        for index, cluster in enumerate(clusters):
            rect = DomCluster.cluster_covered_xyxy(cluster, viewport)
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
            cluster_region.save(self.out_dir / f"{record_prefix}_cluster_{index + 1}.jpg")
            related_tasks.append(DomCluster.determine_cluster_related(cluster, cluster_region, task=act_goal))
        related_tasks_res: list[tuple[bool, ChatImageDetails]] = await asyncio.gather(*related_tasks)
        related_cluster = [
            (cluster, rect) for (flag, _), cluster, rect in zip(related_tasks_res, clusters, related_rects) if flag
        ]
        flags = [flag for flag, _ in related_tasks_res]
        logger.debug(
            f"[Act][Pruning] Related {len(related_cluster)}, unrelated {len(flags) - len(related_cluster)}: {str(flags)}"
        )

        if not related_cluster:
            final_screenshot = screenshot.resize(
                (int(screenshot.width * 0.75), int(screenshot.height * 0.75)), resample=Image.Resampling.LANCZOS
            )
            final_nodes = []
            interactive_nodes_repr = "No act goal related interactive nodes."
            available_actions = Action.get_available_actions_prompt(
                exclude_types=[ActionType.Click, ActionType.Input, ActionType.Search]
            )
        else:
            # 合并边界存在重叠的区域，用于构建紧凑的UI图
            merged_rects = DomCluster.cluster_merge_overlapped(related_cluster)
            # 构建压缩后的UI图像
            final_screenshot = DomCluster.cluster_image_layout_compaction(screenshot, merged_rects, default_gap=1)
            final_nodes: list[DomNode] = list(itertools.chain.from_iterable(cluster for cluster, _ in related_cluster))
            final_screenshot.save(self.out_dir / f"{record_prefix}_final.jpg")
            interactive_nodes_repr = "\n".join(
                [node.convert_to_repr_node().get_human_tree_repr(no_end=True) for node in final_nodes]
            )
            available_actions = Action.get_available_actions_prompt(exclude_types=[ActionType.Search])
        time_line.add("pruning")

        llm_detail_list = [detail for _, detail in related_tasks_res]
        pruning_prompt_tokens = sum(detail["prompt_tokens"] for detail in llm_detail_list)
        pruning_completion_tokens = sum(detail["completion_tokens"] for detail in llm_detail_list)
        pruning_details: PruningDetails = {
            "time": time_line.content[-1][1] - time_line.content[-2][1],
            "model": SecondaryLLM.image_model,
            "prompt_tokens": pruning_prompt_tokens,
            "completion_tokens": pruning_completion_tokens,
            "result": flags,
            "token_reduction": sum(
                detail["prompt_tokens"] for detail, flag in zip(llm_detail_list, flags) if flag == False
            ),
        }

        tabs_info = await self.tab_manager.get_tabs_info()
        current_tabs = f"{tabs_info}\n{viewport.get_viewport_scroll_info()}"

        prompt = self.prompts_dict["act"].format(
            available_actions=available_actions,
            user_request=self.user_request,
            task_state=task_state,
            act_goal=act_goal,
            current_tabs=current_tabs,
            interactive_nodes=interactive_nodes_repr,
        )

        llm_action_detail = await PrimaryLLM.chat_with_image_detail(prompt, final_screenshot)
        time_line.add("llm")
        llm_action_res = llm_action_detail["content"]

        action_details_list = []
        raw_action_list = [line for line in llm_action_res.strip().split("\n") if line.strip()]
        raw_action_types = [raw_action.split(",", maxsplit=1)[0].strip() for raw_action in raw_action_list]
        logger.debug(f"[Act] Generate {len(raw_action_list)} actions: [{', '.join(raw_action_types)}]")

        old_tab_info = await self.tab_manager.get_cur_tab_info()
        for action_index, raw_action in enumerate(raw_action_list, 1):
            action_uid = gen_uid()
            action_prefix = f"{record_prefix}_{action_index:03}_{action_uid}"
            logger.debug(f"[Act] {action_index}. {raw_action}")
            action_screenshot = await tab_screenshot(cur_tab)
            action_screenshot_draw = ImageDraw.Draw(action_screenshot)
            draw_text_label(
                action_screenshot_draw,
                text=raw_action if len(raw_action) < 50 else f"{raw_action[:50]}...",
                position=(10, 10),
                font=self.font_18,
            )
            action_screenshot_path = self.out_dir / f"{action_prefix}_action.jpg"
            action_screenshot.save(action_screenshot_path)
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
                action_screenshot.save(action_screenshot_path)  # overwrite
                # execute action
                action_result = await action.execute(cur_tab, self.tab_manager)
                action_error = None
            except ActionParseException as e:
                logger.warning(f"[Act] Parse action failed, {e}. Action: {raw_action}")
                action_result = str(e)
                action_error = e
                action = None
            except ActionExecuteException as e:
                assert action is not None
                logger.warning(f"[Act] Execute {action.type.value} action failed: {e}")
                action_result = str(e)
                action_error = e

            # update latest tab
            cur_tab = self.tab_manager.front_tab
            new_tab_info = await self.tab_manager.get_cur_tab_info()
            tab_id_changed = new_tab_info["tab_id"] != old_tab_info["tab_id"]
            tab_changed_info = self.tab_manager.compare_tab_info(new_tab_info, old_tab_info)
            old_tab_info = new_tab_info

            # save result info
            action_success = action_error is None
            result_screenshot = await tab_screenshot(cur_tab)
            if action_success:
                result_screenshot_path = self.out_dir / f"{action_prefix}_result.jpg"
            else:
                result_screenshot_path = self.out_dir / f"{action_prefix}_error.jpg"
                result_screenshot_draw = ImageDraw.Draw(result_screenshot)
                draw_text_label(result_screenshot_draw, text=str(action_error), position=(10, 10), font=self.font_18)
            result_screenshot.save(result_screenshot_path)
            execute_result: ActionExecuteResult = {
                "success": action_success,
                "result": action_result,
                "tab_changed_info": tab_changed_info,
            }
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

            if tab_id_changed or not action_success:
                rest_raw_action_list = raw_action_list[action_index:]
                error_reason = "Tab id changed" if tab_id_changed else "Last action failed"
                error_msg = f"{error_reason}, stop executing the remaining actions"
                logger.debug(f"[Act] {error_reason}, stop executing the remaining {len(rest_raw_action_list)} actions")
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
                            execute_result={"success": False, "result": error_msg, "tab_changed_info": None},
                            action=None,
                            action_screenshot_path=action_screenshot_path,
                            result_screenshot_path=result_screenshot_path,
                        )
                    )
                break

        record = ActRecord(
            index=record_index,
            action_details_list=action_details_list,
            time_line=time_line,
            pruning_details=pruning_details,
            interactive_nodes_repr=interactive_nodes_repr,
            llm_details=llm_action_detail,
            act_goal=act_goal,
        )
        record.save(self.out_dir)
        self.records[record_index] = record
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
            task_result = self.last_planning_record.task_state
        else:
            task_result = "Task failed."
            if self.last_planning_record is not None:
                task_result += f" {self.last_planning_record.task_state}"

        records = []
        for record in self.records:
            if isinstance(record, ActRecord):
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
                                "result": d.execute_result["result"],
                            }
                            for d in record.action_details_list
                        ],
                        "pruning_time": round(record.pruning_details["time"], 4),
                        "pruning_tokens": {
                            "prompt_tokens": record.pruning_details["prompt_tokens"],
                            "completion_tokens": record.pruning_details["completion_tokens"],
                            "model": record.pruning_details["model"],
                        },
                        "pruning_token_reduction": record.pruning_details["token_reduction"],
                        "token_usage": {
                            "prompt_tokens": record.llm_details["prompt_tokens"],
                            "completion_tokens": record.llm_details["completion_tokens"],
                            "model": record.llm_details["model"],
                        },
                        "time_cost": round(record.time_line.total_time(), 4),
                        "time_point": record.time_line.endpoint(),
                    }
                )
            elif isinstance(record, ObservationRecord):
                records.append(
                    {
                        "type": "observation",
                        "observation": record.observation,
                        "time_cost": round(record.time_line.total_time(), 4),
                        "time_point": record.time_line.endpoint(),
                        "token_usage": {
                            "prompt_tokens": record.llm_details["prompt_tokens"],
                            "completion_tokens": record.llm_details["completion_tokens"],
                            "model": record.llm_details["model"],
                        },
                    }
                )
            elif isinstance(record, PlanningRecord):
                records.append(
                    {
                        "type": "planning",
                        "new_progress": record.new_progress,
                        "requested_data_found": record.requested_data_found,
                        "task_state": record.task_state,
                        "act_goal": record.act_goal,
                        "task_completed": record.task_completed,
                        "screenshot_name": record.screenshot_path.name,
                        "time_cost": round(record.time_line.total_time(), 4),
                        "time_point": record.time_line.endpoint(),
                        "token_usage": {
                            "prompt_tokens": record.llm_details["prompt_tokens"],
                            "completion_tokens": record.llm_details["completion_tokens"],
                            "model": record.llm_details["model"],
                        },
                    }
                )
            elif isinstance(record, ExtractionRecord):
                records.append(
                    {
                        "type": "extraction",
                        "data": record.data,
                        "time_cost": round(record.time_line.total_time(), 4),
                        "time_point": record.time_line.endpoint(),
                        "token_usage": {
                            "prompt_tokens": record.llm_details["prompt_tokens"],
                            "completion_tokens": record.llm_details["completion_tokens"],
                            "model": record.llm_details["model"],
                        },
                    }
                )
            elif isinstance(record, FeedbackRecord):
                records.append(
                    {
                        "type": "feedback",
                        "feedback": record.repr,
                        "time_cost": round(record.time_line.total_time(), 4),
                        "time_point": record.time_line.endpoint(),
                        "token_usage": {
                            "prompt_tokens": record.llm_details["prompt_tokens"],
                            "completion_tokens": record.llm_details["completion_tokens"],
                            "model": record.llm_details["model"],
                        },
                    }
                )

        time_cost = {
            "total_time": round(total_time_cost, 4),
            "act_time": 0,
            "observation_time": 0,
            "planning_time": 0,
            "extraction_time": 0,
            "feedback_time": 0,
        }
        for record in self.records:
            if isinstance(record, ActRecord):
                time_cost["act_time"] += record.time_line.total_time()
            elif isinstance(record, ObservationRecord):
                time_cost["observation_time"] += record.time_line.total_time()
            elif isinstance(record, PlanningRecord):
                time_cost["planning_time"] += record.time_line.total_time()
            elif isinstance(record, ExtractionRecord):
                time_cost["extraction_time"] += record.time_line.total_time()
            elif isinstance(record, FeedbackRecord):
                time_cost["feedback_time"] += record.time_line.total_time()

        for key in ("act_time", "observation_time", "planning_time", "extraction_time", "feedback_time"):
            time_cost[key] = round(time_cost[key], 4)

        out = {
            "user_request": user_request,
            "token": token,
            "records": records,
            "time_cost": time_cost,
            "success": success,
            "result": task_result,
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
        report_lines.append("## Overall Summary")
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
        # Feedback
        for record in reversed(records):
            if record["type"] == "feedback":
                report_lines.append("### Feedback")
                report_lines.append("")
                report_lines.append(record["feedback"])
                break

        report_lines.append("")
        # Time cost
        counter = defaultdict(int)
        for record in records:
            counter[record["type"]] += 1
        report_lines.append("### Time Cost (seconds)")
        report_lines.append("")
        report_lines.append("| Type | Count | Time | Avg Time |")
        report_lines.append("|------|-------|------|----------|")
        report_lines.append(
            f"| Total | {counter["planning"]} | {time_cost['total_time']} | {time_cost['total_time'] / counter["planning"]:.4f} |"
        )
        type_names = ("planning", "extraction", "feedback", "act", "observation")
        for type_name in type_names:
            if counter[type_name] == 0:
                continue
            time_key = type_name + "_time"
            report_lines.append(
                f"| {type_name.title()} | {counter[type_name]} | {time_cost[time_key]} | {time_cost[time_key] / counter[type_name]:.4f} |"
            )
        report_lines.append("")
        # Ganntt Chart
        # 甘特图
        global_start = min(r["time_point"][0] for r in records)
        global_end = max(r["time_point"][1] for r in records)
        ganntt_spans = []
        for index, r in enumerate(records):
            start = r["time_point"][0] - global_start
            end = r["time_point"][1] - global_start
            ganntt_spans.append(
                {
                    "index": index,
                    "label": r["type"],
                    "start": start,
                    "duration": end - start,
                }
            )
        # 1. 按 label 分组（每种 label 一行）
        label_to_spans = defaultdict(list)
        for span in ganntt_spans:
            label_to_spans[span["label"]].append(span)
        labels = tuple(reversed(type_names))
        # 2. 为每种 label 分配一行 y 轴位置
        label_to_y = {label: i for i, label in enumerate(labels)}
        # 3. 为每种 label 分配一种颜色
        cmap = plt.get_cmap("tab10")  # 颜色足够区分
        label_to_color = {label: cmap(i % cmap.N) for i, label in enumerate(labels)}
        label_height = 0.6
        _, ax = plt.subplots(figsize=(max(2, (global_end - global_start) // 20) * 5, len(labels) * label_height * 1))
        for label, spans in label_to_spans.items():
            y = label_to_y[label]
            color = label_to_color[label]
            for span in spans:
                start = span["start"]
                duration = span["duration"]
                end = start + duration
                index = span["index"]
                # 4. 绘制甘特条
                ax.barh(y=y, width=duration, left=start, height=label_height, color=color, edgecolor="black", alpha=0.8)
                # 5. 标记 start、duration、end
                ax.text(
                    start + duration + 0.1,
                    y,
                    f"{start:.2f}\n{duration:.2f}s\n{end:.2f}",
                    ha="left",
                    va="center",
                    fontsize=8,
                )
                # 6. 标记 index（放在条中间）
                ax.text(
                    start + duration / 2,
                    y,
                    f"#{index}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    weight="bold",
                )
        ax.set_yticks([label_to_y[label] for label in labels])
        ax.set_yticklabels(labels)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Label")
        ax.set_title("Gantt Chart")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(out_dir / "gantt.png")
        report_lines.append("![Gantt Chart](gantt.png)")
        report_lines.append("")

        # Part 2: Progress History
        report_lines.append("## Progress History")
        report_lines.append("")
        for record in records:
            if record["type"] == "planning":
                report_lines.append(f"- {record.get('new_progress', '')}")
            if record["type"] == "extraction":
                report_lines.append(f"- {record.get('data', '')}")
        report_lines.append("")

        # Part 3: Execution Records
        report_lines.append("## Execution Records")
        report_lines.append("")

        for idx, record in enumerate(records, start=1):
            record_type = record["type"]
            token_usage = record["token_usage"]
            # Act Record
            if record_type == "act":
                report_lines.append(f"### {idx}. Act")
                report_lines.append(f"- **Time Cost**: {record['time_cost']}s")
                report_lines.append(
                    f"- **Token Usage**: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, model={token_usage['model']}"
                )
                report_lines.append(f"- **Pruning Time**: {record['pruning_time']}s")
                pruning_tokens = record["pruning_tokens"]
                report_lines.append(
                    f"- **Pruning Token Usage**: prompt={pruning_tokens['prompt_tokens']}, completion={pruning_tokens['completion_tokens']}, model={pruning_tokens['model']}"
                )
                report_lines.append("")
                report_lines.append("**Actions:**")
                for action_idx, action in enumerate(record["actions"], start=1):
                    success_flag = "✅" if action["success"] else "❌"
                    report_lines.append(f"{action_idx}. **Action {action_idx}**")
                    report_lines.append(f"    - **Raw**: {action['raw']}")
                    report_lines.append(f"    - **Description**: {action['description']}")
                    report_lines.append(f"    - **Success**: {success_flag}")
                    if action.get("result") is not None:
                        report_lines.append(f"    - **Result**: {action['result']}")

                    # 并排显示截图
                    action_img = f"![Action]({action['action_screenshot_name']})"
                    result_img = f"![Result]({action['result_screenshot_name']})" if action["success"] else ""
                    if action_img and result_img:
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
                    f"- **Token Usage**: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, model={token_usage['model']}"
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
                    f"- **Token Usage**: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, model={token_usage['model']}"
                )
                report_lines.append(f"- **Task Completed**: {record['task_completed']}")

                report_lines.append(f"- **New Progress:** {record.get('new_progress', '')}")
                report_lines.append(f"- **Requested Data Found:** {record.get('requested_data_found', '')}")
                report_lines.append(f"- **Task State:** {record.get('task_state', '')}")
                report_lines.append(f"- **Act Goal:** {record.get('act_goal', '')}")
                report_lines.append(f"- **Planning Screenshot**: ![Planning]({record['screenshot_name']})")
                report_lines.append("")
            # Feedback Record
            elif record_type == "feedback":
                report_lines.append(f"### {idx}. Feedback")
                report_lines.append(f"- **Time Cost**: {record['time_cost']}s")
                report_lines.append(
                    f"- **Token Usage**: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, model={token_usage['model']}"
                )
                report_lines.append("")
                report_lines.append("**Feedback**:")
                report_lines.append(record["feedback"])
                report_lines.append("")
            # Extraction Record
            elif record_type == "extraction":
                report_lines.append(f"### {idx}. Extraction")
                report_lines.append(f"- **Time Cost**: {record['time_cost']}s")
                report_lines.append(
                    f"- **Token Usage**: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, model={token_usage['model']}"
                )
                report_lines.append("")
                report_lines.append("**Extraction**:")
                report_lines.append(record["data"])
                report_lines.append("")

        # Write md file
        report_path = out_dir / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        logger.info("Report saved to {}", report_path)
