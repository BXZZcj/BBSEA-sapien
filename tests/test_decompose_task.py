import numpy as np
import os
import re
from openai import OpenAI
from config.core import prompts_path, gpt_api_key, gpt_base_url


def main(response):
    subtask_list = []
    primitive_action_list = []
    primitive_action_str_list = []
    reasoning = response.split("\nanswer:\n")[0]

    if ' no.' in response:
        return subtask_list, primitive_action_str_list, primitive_action_list, reasoning
    steps = response.split("\nanswer:\n")[-1].split("\n")  # str list
    for step in steps:
        primitive_actions = []
        subtask = ''
        try:
            subtask = step.split('|')[0].split('.')[1]
            action = step.split('|')[1]
        
            match = re.search(r'\[(.*)\]', action)
            if match:
                primitive_action_str = match.group(1)
                primitive_action_str_list.append(primitive_action_str)
                
                for item in primitive_action_str.split(';'):
                    obj = eval(item) 
                    primitive_actions.append(obj)
            else:
                # import pdb;pdb.set_trace()
                return subtask_list, primitive_action_str_list, primitive_action_list, reasoning
        except:
            try:
                subtask = step.split(':')[0].split('.')[1]
                action = step.split(':')[1]
                # match = re.search(r'\[(.*)\]', step)

                match = re.search(r'\[(.*)\]', action)
                if match:
                    primitive_action_str = match.group(1)
                    primitive_action_str_list.append(primitive_action_str)

                    for item in primitive_action_str.split(';'):
                        obj = eval(item) 
                        primitive_actions.append(obj)
                else:
                    # import pdb;pdb.set_trace()
                    return subtask_list, primitive_action_str_list, primitive_action_list, reasoning
            except:
                print("step", step)
            # import pdb;pdb.set_trace()
        subtask_list.append(subtask)
        primitive_action_list.append(primitive_actions)
    # is_first_time = False
    return subtask_list, primitive_action_str_list, primitive_action_list, reasoning



if __name__ == "__main__":
    response1="""==========================
    - Stack the red block on top of the green bowl
    - Align the green bowl with the edge of the table
    - Gather the red block and the green bowl to one side of the table
    - Sort the objects on the table by color
    - Move the red block to the center of the table
    ==========================
    The task is to put the red block in the green bowl. Since both the red block and the green bowl are on the table, we can use the Pick and PlaceOn primitive actions to accomplish this task.

    Subtasks:
    1. Pick up the red block
    2. Place the red block in the green bowl

    Corresponding primitive actions:
    1. Pick('red block')
    2. PlaceOn('green bowl')
    ==========================
    The red block is in the green bowl.
    reasoning: 
    answer:"""

    subtask_list, primitive_action_str_list, primitive_action_list, reasoning=main(response1)

    print(primitive_action_str_list)