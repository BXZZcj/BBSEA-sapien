import os
import re
from openai import OpenAI
from config.core import prompts_path, gpt_api_key, gpt_base_url


client = OpenAI(
    api_key=gpt_api_key,
    base_url=gpt_base_url
)


def gpt_35_api(messages: list, temperature):
    completion = client.chat.completions.create(model="gpt-4", messages=messages, temperature=temperature)
    return completion.choices[0].message.content


def propose_task(scene_graph, temperature=0.2):
    with open(prompts_path["task_propose"], 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    prompt = base_prompt.format(scene_graph)
    messages = [{'role': 'user','content': prompt},]
    response = gpt_35_api(messages, temperature)

    tasks = response.split("\ntasks:\n")[-1].split("\n")  # str list
    task_desc_list = []
    for task in tasks:
        try:
            if task.strip()[0]=='-':
                task_desc_list.append(task.split('- ')[1].strip('.'))
            else:
                break
        except:
            try:
                task_desc_list.append(task.split('. ')[1].strip('.'))
            except:
                task_desc_list.append(task.split(': ')[1].strip('.'))
                print(tasks)
                print(task)

    return task_desc_list


def decompose_task(task_desc, scene_graph, temperature=0.2):
    with open(prompts_path["task_decompose"], 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    prompt = base_prompt.format(task_desc, scene_graph)

    messages = [{'role': 'user','content': prompt},]
    response = gpt_35_api(messages, temperature)

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


def infer_if_success(task_desc, scene_graph_list, temperature=0.2):
    scene_graph_list_str = ''
    for scene_graph in scene_graph_list:
        scene_graph_list_str += f'  ----------\n{scene_graph}'

    with open(prompts_path["success_infer"], 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    prompt = base_prompt.format(task_desc, scene_graph_list_str)

    messages = [{'role': 'user','content': prompt},]
    response = gpt_35_api(messages, temperature)

    answer = response.split('answers:\n')[-1]

    if 'yes' in answer:
        return True, response
    elif 'no' in answer:
        return False, response
    else:
        return 'not sure', response


if __name__ == '__main__':
    # Task Proposal
    scene_graph = '''  [Nodes]:
    - red block
    - green bowl
    - table
  [Edges]:
    - red block -> on top of -> table
    - green bowl -> on top of -> table'''
    task_list = propose_task(scene_graph)
    print("Task Proposal\n  task_list:\n\n", task_list)

    # Task Decompose
    task_desc = 'put the red block in the green bowl'
    scene_graph = '''  [Nodes]:
    - red block
    - green bowl
    - table
  [Edges]:
    - red block -> on top of -> table
    - green bowl -> on top of -> table'''
    subtask_list, primitive_action_str_list, primitive_action_list, reasoning = decompose_task(task_desc, scene_graph)
    print("Task Decompose")
    # print("  reasoning:\n\n", reasoning)
    # print("\n  subtask_list:\n\n", subtask_list)
    print("\n  primitive_action_str_list: \n\n", primitive_action_str_list)

    # Success Infer
    task_desc = 'put the red block in the green bowl'
    scene_graph = '''  [Nodes]:
    - red block
    - green bowl
    - table
  [Edges]:'''
    is_successful = infer_if_success(task_desc, scene_graph)
    print("Success Infer\n  is_successful:\n\n", is_successful)