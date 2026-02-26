import json
import shutil
from pathlib import Path
from typing import TypedDict
import csv


class Mind2WebTask(TypedDict):
    task_id: str
    confirmed_task: str
    website: str
    reference_length: int
    level: str


def load_tasks(task_path: Path) -> list[Mind2WebTask]:
    with open(task_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    return tasks


def save_task_as_csv(tasks: list[Mind2WebTask], csv_path: Path):
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "confirmed_task", "website", "reference_length", "level"])
        for task in tasks:
            writer.writerow(
                [
                    task["task_id"],
                    task["confirmed_task"],
                    task["website"],
                    task["reference_length"],
                    task["level"],
                ]
            )


def result_convert(task: Mind2WebTask, result_dir: Path, base_output_dir: Path):
    output_dir = base_output_dir / task["task_id"]

    screenshot_dir = output_dir / "trajectory"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "result.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    act_records = [r for r in data["records"] if r["type"] == "act"]
    action_history = []
    last_action_result_name = None
    for r in act_records:
        for a in r["actions"]:
            if a["success"]:
                raw_action: str = a["raw"].lower()
                # ActionType.Extract, ActionType.Memory
                if raw_action.startswith("extract") or raw_action.startswith("memory"):
                    continue
                action_history.append(a["description"])
                screenshot_path = result_dir / f"{a['action_screenshot_name']}"
                shutil.copy(screenshot_path, screenshot_dir / f"{len(action_history)-1}_screenshot.jpg")
                last_action_result_name = a["result_screenshot_name"]
    assert last_action_result_name is not None
    shutil.copy(result_dir / last_action_result_name, screenshot_dir / f"{len(action_history)-1}_screenshot.jpg")

    result = {"task_id": task["task_id"], "action_history": action_history, "task": task["confirmed_task"]}
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Output to {output_dir}")


if __name__ == "__main__":

    def main():
        task_path = Path("local/Online_Mind2Web.json")
        tasks = load_tasks(task_path)
        # save_task_as_csv(tasks, task_path.with_suffix(".csv"))
        task_dict = {task["task_id"]: task for task in tasks}

        result_dir = Path("output/test")
        task = task_dict["b7258ee05d75e6c50673a59914db412e_110325"]
        output_dir = Path("output/mind2web2-leaderboard/")
        result_convert(task, result_dir, output_dir)

    main()
