import json
import random
from pathlib import Path

import pandas as pd
from loguru import logger

import globus_util

# globus 相关
source_collection_id = "32e6b738-a0b0-47f8-b475-26bf1c5ebf19"  # mind2web
dest_collection_id = "20743934-e559-11f0-86b8-0eb0b913a0ab"  # 本地端点
source_base_path = "/data/raw_dump/task"
dest_base_path = "/~/D/globus/mind2web"


def read_mind2web(dataset_path: Path, data_name: str, file_index: int = 0):
    files = sorted(dataset_path.glob(f"{data_name}-*.parquet"))
    if file_index >= len(files):
        return None
    df = pd.read_parquet(files[file_index])
    return df


def save_test_data():
    dataset_path = Path("D:/mind2web/data")

    for data_name in ["test_domain", "test_task", "test_website", "train"]:
        logger.info("开始处理数据: {}", data_name)
        tasks = []
        df_task_file_index = 0
        df_task = read_mind2web(dataset_path, data_name, df_task_file_index)
        if df_task is None:
            return

        row_index = 0
        while True:
            if row_index >= len(df_task):
                df_task_file_index += 1
                logger.info("读取下一个数据文件: {}", df_task_file_index)
                df_task = read_mind2web(dataset_path, data_name, df_task_file_index)
                row_index = 0
                if df_task is None:
                    logger.info("所有数据已读取完毕")
                    break

            row = df_task.iloc[row_index]
            row_index += 1
            cur_action = {
                "action_uid": row["action_uid"],
                "target_action_index": row["target_action_index"],
                "target_action_reprs": row["target_action_reprs"],
                "operation": row["operation"],
            }

            if tasks and tasks[-1]["annotation_id"] == row["annotation_id"]:
                tasks[-1]["actions"].append(cur_action)
            else:
                tasks.append({
                    "annotation_id": row["annotation_id"],
                    "confirmed_task": row["confirmed_task"],
                    "actions": [cur_action],
                })

        logger.info("已添加任务: {}", len(tasks))

        with open(f"local/{data_name}.json", "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=4)


def split_task_meta():
    test_domain_tasks = []
    test_task_tasks = []
    test_website_tasks = []
    train_tasks = []

    with open("local/task_meta.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)
    for task in tasks:
        if task["split"] == "test_domain":
            test_domain_tasks.append(task)
        elif task["split"] == "test_task":
            test_task_tasks.append(task)
        elif task["split"] == "test_website":
            test_website_tasks.append(task)
        elif task["split"] == "train":
            train_tasks.append(task)
        else:
            logger.error("未知的split: {}", task["split"])

    with open("local/test_domain.json", "w", encoding="utf-8") as f:
        json.dump(test_domain_tasks, f, ensure_ascii=False, indent=4)
    with open("local/test_task.json", "w", encoding="utf-8") as f:
        json.dump(test_task_tasks, f, ensure_ascii=False, indent=4)
    with open("local/test_website.json", "w", encoding="utf-8") as f:
        json.dump(test_website_tasks, f, ensure_ascii=False, indent=4)
    with open("local/train.json", "w", encoding="utf-8") as f:
        json.dump(train_tasks, f, ensure_ascii=False, indent=4)


def choose_random_tasks(count: int = 10):
    out = []
    for data_name in ["test_domain", "test_task", "test_website"]:
        with open(f"local/{data_name}.json", "r", encoding="utf-8") as f:
            tasks = json.load(f)

        # 随机选择10个
        out_tasks = random.choices(tasks, k=count)
        for t in out_tasks:
            t["type"] = data_name
        out.extend(out_tasks)

    with open("local/random.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)


def download_all_data():
    all_tasks = []
    all_data_names = ["test_domain", "test_task", "test_website"]
    for name in all_data_names:
        json_file_name = f"{name}.json"
        with open(f"local/{json_file_name}", "r", encoding="utf-8") as f:
            tasks = json.load(f)
            all_tasks.extend(tasks)

    logger.info("准备下载数据: {}, 共{}个任务", all_data_names, len(all_tasks))
    data_items = []
    for task in all_tasks:
        data_items.append(
            (
                f"{source_base_path}/{task['annotation_id']}/processed/snapshots/",
                f"{dest_base_path}/{task['annotation_id']}/"
            )
        )
        data_items.append(
            (
                f"{source_base_path}/{task['annotation_id']}/processed/screenshot.json",
                f"{dest_base_path}/{task['annotation_id']}/screenshot.json"
            )
        )

    transfer_client = globus_util.auth_load()
    globus_util.globus_download(transfer_client, source_collection_id, dest_collection_id, data_items)


def test_data():
    root = Path("D:/globus/mind2web/")
    for index, record in enumerate(root.glob("**/screenshot.json")):
        print(index)
        with open(record, "r", encoding="utf-8") as f:
            actions = json.load(f)
            for action in actions:
                action_detail = action["action"]
                if len(action_detail.keys()) != 1:
                    print(action_detail)
                if "bounding_box" not in action_detail:
                    print(action_detail)


if __name__ == "__main__":
    download_all_data()
