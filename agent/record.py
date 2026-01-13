from dataclasses import dataclass
import json
from pathlib import Path
import time

from agent.action import ActionDetails
from agent.llm import ChatImageDetails, ChatTextDetails


# TODO 记录每个阶段的细节


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
    pass


@dataclass
class ActRecord(Record):
    action_details_list: list[ActionDetails]
    time_line: TimeLine
    llm_details: ChatImageDetails
    pruned_dom_repr: str

    def save(self, out_dir: Path):
        with open(out_dir / f"{self.index:02}_act.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "action_details_list": [a.to_dict() for a in self.action_details_list],
                    "time_line": self.time_line.content,
                    "llm_details": self.llm_details,
                    "pruned_dom_repr": self.pruned_dom_repr,
                },
                f,
                indent=4,
            )


@dataclass
class ObservationRecord(Record):
    pass
