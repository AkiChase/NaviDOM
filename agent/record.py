from dataclasses import dataclass
import json
from pathlib import Path
import time

from agent.action import ActionDetails
from agent.llm import ChatImageDetails, ChatImageListDetails, ChatTextDetails


@dataclass
class Record:
    index: int

    def save(self, out_dir: Path, name: str, data: dict):
        with open(out_dir / f"{self.index:03}_{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


class TimeLine:
    content: list[tuple[str, float]]

    def __init__(self, label: str = "start", time_point: float | None = None):
        if time_point is None:
            time_point = time.time()
        self.content = [(label, time_point)]

    def add(self, label: str, time_point: float | None = None):
        if time_point is None:
            time_point = time.time()
        self.content.append((label, time_point))

    def total_time(self) -> float:
        return self.content[-1][1] - self.content[0][1]

    def to_dict(self) -> dict:
        return {
            "total_time": self.total_time(),
            "time_points": self.content.copy(),
        }


@dataclass
class PlanningRecord(Record):
    llm_details: ChatImageDetails
    new_progress: str
    requested_data_found: str | None
    task_state: str
    act_goal: str
    task_completed: bool
    time_line: TimeLine

    def save(self, out_dir: Path):
        super().save(
            out_dir,
            "planning",
            {
                "llm_details": self.llm_details,
                "new_progress": self.new_progress,
                "requested_data_found": self.requested_data_found,
                "task_state": self.task_state,
                "act_goal": self.act_goal,
                "task_completed": self.task_completed,
                "time_line": self.time_line.to_dict(),
            },
        )


@dataclass
class ExtractionRecord(Record):
    llm_details: ChatImageDetails
    data: str
    time_line: TimeLine

    def save(self, out_dir: Path):
        super().save(
            out_dir,
            "extraction",
            {
                "llm_details": self.llm_details,
                "data": self.data,
                "time_line": self.time_line.to_dict(),
            },
        )


@dataclass
class FeedbackRecord(Record):
    llm_details: ChatTextDetails
    feedback: str
    time_line: TimeLine

    def save(self, out_dir: Path):
        super().save(
            out_dir,
            "feedback",
            {
                "llm_details": self.llm_details,
                "feedback": self.feedback,
                "time_line": self.time_line.to_dict(),
            },
        )


@dataclass
class ActRecord(Record):
    action_details_list: list[ActionDetails]
    time_line: TimeLine
    llm_details: ChatImageDetails | ChatTextDetails
    interactive_nodes_repr: str
    act_goal: str

    def get_actions_descriptions(self) -> str:
        actions_info = []
        for index, action_details in enumerate(self.action_details_list, 1):
            if action_details.action is not None:
                actions_info.append(f"{index}.")
                actions_info.append(f"- Describe: {action_details.action.get_description()}")
                success = action_details.execute_result["success"]
                result = action_details.execute_result["result"]
                tab_changed_info = action_details.execute_result["tab_changed_info"]
                if not success and result is not None:
                    actions_info.append(f"- Error: {result}")
                if tab_changed_info is not None:
                    actions_info.append(f"- Tab Changed: {tab_changed_info}")
        return "\n".join(actions_info)

    def save(self, out_dir: Path):
        super().save(
            out_dir,
            "act",
            {
                "action_details_list": [a.to_dict() for a in self.action_details_list],
                "time_line": self.time_line.to_dict(),
                "llm_details": self.llm_details,
                "interactive_nodes_repr": self.interactive_nodes_repr,
                "act_goal": self.act_goal,
            },
        )


@dataclass
class ObservationRecord(Record):
    llm_details: ChatImageListDetails | None
    observation: str
    time_line: TimeLine

    def save(self, out_dir: Path):
        super().save(
            out_dir,
            "observation",
            {
                "llm_details": self.llm_details,
                "observation": self.observation,
                "time_line": self.time_line.to_dict(),
            },
        )
