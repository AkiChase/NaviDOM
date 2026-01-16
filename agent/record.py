from dataclasses import dataclass
import json
from pathlib import Path
import time

from agent.action import ActionDetails
from agent.llm import ChatImageDetails, ChatImageListDetails, ChatTextDetails


@dataclass
class Record:
    index: int


class TimeLine:
    content: list[tuple[str, float]]

    def __init__(self, label: str = "start", time_point: float = time.time()):
        self.content = [(label, time_point)]

    def add(self, label: str, time_point: float = time.time()):
        self.content.append((label, time_point))


@dataclass
class PlanningRecord(Record):
    llm_details: ChatTextDetails | ChatImageDetails
    current_state: str
    nearest_next_objective: str
    future_plan: str
    task_completed: bool

    def save(self, out_dir: Path):
        with open(out_dir / f"{self.index:03}_planning.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "llm_details": self.llm_details,
                    "current_state": self.current_state,
                    "nearest_next_objective": self.nearest_next_objective,
                    "future_plan": self.future_plan,
                    "task_completed": self.task_completed,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )


@dataclass
class ActRecord(Record):
    action_details_list: list[ActionDetails]
    time_line: TimeLine
    llm_details: ChatImageDetails | ChatTextDetails
    pruned_dom_repr: str

    def save(self, out_dir: Path):
        with open(out_dir / f"{self.index:03}_act.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "action_details_list": [a.to_dict() for a in self.action_details_list],
                    "time_line": self.time_line.content,
                    "llm_details": self.llm_details,
                    "pruned_dom_repr": self.pruned_dom_repr,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )


@dataclass
class ObservationRecord(Record):
    llm_details: ChatImageListDetails | None
    observation: str
    time_line: TimeLine

    def save(self, out_dir: Path):
        with open(out_dir / f"{self.index:03}_observation.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "llm_details": self.llm_details,
                    "observation": self.observation,
                    "time_line": self.time_line.content,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
