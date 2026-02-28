import asyncio
import base64
import io
import json
import os
import re
import shutil
from pathlib import Path
from typing import TypedDict
import csv
from PIL import Image
from openai import OpenAI


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


def result_convert(task: Mind2WebTask, result_dir: Path, mind2web_base_output_dir: Path):
    output_dir = mind2web_base_output_dir / task["task_id"]
    if output_dir.exists():
        shutil.rmtree(output_dir)

    screenshot_dir = output_dir / "trajectory"
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

    return output_dir


class OpenaiEngine:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature=0,
    ) -> None:

        self.temperature = temperature
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def generate(self, messages, max_new_tokens=512):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
        )
        return [choice.message.content for choice in response.choices]


def encode_image(image):
    """Convert a PIL image to base64 string."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def identify_key_points(task, model):
    system_msg = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""
    prompt = """Task: {task}"""
    text = prompt.format(task=task)
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    ]
    responses = await asyncio.to_thread(model.generate, messages)
    return responses[0]


async def judge_image(task, image_path, key_points, model):
    system_msg = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze the provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:  
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.  
- Does the image show actions, progress indicators, or critical information directly related to completing the task?  
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Provide your response in the following format:  
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the image that indicate necessary steps, evidence, or lack thereof.  
- **Score**: Assign a score based on the reasoning, using the following scale:  
    - **1**: The image does not contain any necessary steps or relevant information.  
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.  
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.  
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.  
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

Respond with:  
1. **Reasoning**: [Your explanation]  
2. **Score**: [1-5]"""

    jpg_base64_str = encode_image(Image.open(image_path))

    prompt = """**Task**: {task}

**Key Points for Task Completion**: {key_points}

The snapshot of the web page is shown in the image."""
    text = prompt.format(task=task, key_points=key_points)

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}", "detail": "high"},
                },
            ],
        },
    ]

    responses = await asyncio.to_thread(model.generate, messages)
    return responses[0]


async def WebJudge_Online_Mind2Web_eval(task, last_actions, images_path, model, score_threshold):
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
Examples of Failure Cases:
- If the requirement is less than $50, but the applied filter is less than $25, it is a failure.
- If the requirement is $1500-$2500, but the applied filter is $2000-$2500, it is a failure.
- If the requirement is $25-$200, but the applied filter is $0-$200, it is a failure.
- If the required years are 2004-2012, but the filter applied is 2001-2012, it is a failure.
- If the required years are before 2015, but the applied filter is 2000-2014, it is a failure.
- If the task requires exactly 2 beds, but the filter applied is 2+ beds, it is a failure.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: If the current page already displays all available items, then applying a filter is not necessary. As long as the agent selects items that meet the requirements (e.g., the cheapest or lowest price), the task is still considered successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""
    prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{thoughts}"""

    key_points = await identify_key_points(task, model)
    key_points = key_points.replace("\n\n", "\n")

    try:
        key_points = key_points.split("**Key Points**:")[1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    except:
        key_points = key_points.split("Key Points:")[-1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())

    # tasks = [judge_image(task, image_path, key_points, model) for image_path in images_path]
    # image_responses = await asyncio.gather(*tasks)
    # 串行避免服务端崩溃
    image_responses = [await judge_image(task, image_path, key_points, model) for image_path in images_path]

    whole_content_img = []
    whole_thoughts = []
    record = []
    pattern = r"[1-5]"
    for response, image_path in zip(image_responses, images_path):
        try:
            score_text = response.split("Score")[1]
            thought = response.split("**Reasoning**:")[-1].strip().lstrip("\n").split("\n\n")[0].replace("\n", " ")
            score = re.findall(pattern, score_text)[0]
            record.append({"Response": response, "Score": int(score)})
        except Exception as e:
            print(f"Error processing response: {e}")
            score = 0
            record.append({"Response": response, "Score": 0})

        if int(score) >= score_threshold:
            jpg_base64_str = encode_image(Image.open(image_path))
            whole_content_img.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{jpg_base64_str}", "detail": "high"}}
            )
            if thought != "":
                whole_thoughts.append(thought)

    whole_content_img = whole_content_img[:MAX_IMAGE]
    whole_thoughts = whole_thoughts[:MAX_IMAGE]
    if len(whole_content_img) == 0:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}"""
    text = prompt.format(
        task=task,
        last_actions="\n".join(f"{i+1}. {action}" for i, action in enumerate(last_actions)),
        key_points=key_points,
        thoughts="\n".join(f"{i+1}. {thought}" for i, thought in enumerate(whole_thoughts)),
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": [{"type": "text", "text": text}] + whole_content_img},
    ]
    return messages, text, system_msg, record, key_points


MAX_IMAGE = 15


async def judge_task(task_dir: Path, llm: OpenaiEngine, score_threshold: int):
    trajectory_images_path = task_dir / "trajectory"
    screenshot_paths = []
    action_history = None
    task_description = None
    # Load results
    with open(task_dir / "result.json") as f:
        output_results = json.load(f)

    task_id = output_results["task_id"]
    task_description = output_results["task"]
    action_history = output_results["action_history"]

    for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r"\d+", x)[0])):
        screenshot_paths.append(os.path.join(trajectory_images_path, image))
    messages, text, system_msg, record, key_points = await WebJudge_Online_Mind2Web_eval(
        task_description, action_history, screenshot_paths, llm, score_threshold
    )
    output_results["image_judge_record"] = record
    output_results["key_points"] = key_points

    response = llm.generate(messages)[0]
    predicted_label = -1
    try:
        assert response is not None, "LLM response is None"
        if "success" in response.lower().split("status:")[1]:
            predicted_label = 1
        else:
            predicted_label = 0
    except:
        predicted_label = -1


    # Store evaluation details
    evaluation_results = {"response": response, "predicted_label": predicted_label}
    output_results["task_id"] = task_id
    output_results["input_text"] = text
    output_results["system_msg"] = system_msg
    output_results["evaluation_details"] = evaluation_results
    output_results["predicted_label"] = predicted_label

    return output_results


def record_judge_result(result_path: Path, judge_result: dict):
    os.makedirs(result_path.parent, exist_ok=True)
    with open(result_path, "a+") as f_out:
        f_out.write(json.dumps(judge_result) + "\n")


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
