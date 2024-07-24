import numpy as np
import os
import re
from openai import OpenAI
from config.core import prompts_path, gpt_api_key, gpt_base_url


def main(response):
    parts = response.split("Subtasks:")
    reasoning = parts[0].strip()

    subtasks_part, actions_part = parts[1].split("Corresponding primitive actions:")

    # 解析Subtasks
    subtask_pattern = r"\d+\.\s+(.+)"
    subtasks = re.findall(subtask_pattern, subtasks_part)

    # 解析Corresponding primitive actions
    action_pattern = r"\d+\.\s+(.+)"
    actions = re.findall(action_pattern, actions_part)

    # 创建字典，将subtasks和actions映射起来
    task_mapping = dict(zip(subtasks, actions))

    return reasoning, task_mapping



if __name__ == "__main__":
    response1="""The task is to put the red block in the green bowl. Since both the red block and the green bowl are on the table, we can use the Pick and PlaceOn primitive actions to accomplish this task.

    Subtasks:
    1. Pick up the red block
    2. Place the red block in the green bowl

    Corresponding primitive actions:
    1. Pick('red block')
    2. PlaceOn('green bowl')"""

    reasoning, task_mapping = main(response1)