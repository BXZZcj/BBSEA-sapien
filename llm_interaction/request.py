import os
import re
from openai import OpenAI, AzureOpenAI
from typing import Tuple, List, Union
from PIL import Image
import base64
from io import BytesIO


from config.core import prompts_path,\
    chatanywhere_api_key,\
    chatanywhere_base_url,\
    azure_api_key,\
    azure_endpoint,\
    azure_api_version


def gpt_api(
        messages: list, 
        temperature: float, 
        model:str="vision", 
        max_tokens:int=100000
)->str:
    if model == "vision" or "4o":
        client = AzureOpenAI(
            api_key=azure_api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
        )
    else:
        client = OpenAI(
            api_key=chatanywhere_api_key,
            base_url=chatanywhere_base_url,
        )

    completion = client.chat.completions.create(
        model=model, 
        messages=messages, 
        temperature=temperature, 
        max_tokens=max_tokens,
        )
    
    return completion.choices[0].message.content



def encode_image(
        img:Image.Image, 
        max_image:int=512
)->str:
    width, height = img.size
    max_dim = max(width, height)
    if max_dim > max_image:
        scale_factor = max_image / max_dim
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height))

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def propose_task_w_img(
        perception_info:List[Image.Image], 
        model:str, 
        temperature:float=0.2
)->list:
    """
    perception_info: Pictures token
    """
    with open(prompts_path["task_propose_w_img"], 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    imgs_encoded = []
    for img in perception_info:
        imgs_encoded.append(encode_image(img, max(img.size)))
        
    prompt = base_prompt
    messages = [
        {
            'role': 'user',
            'content': [
                {"type":"text", "text": prompt},
            ]+[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_encoded}"},
                } for img_encoded in imgs_encoded
            ],
        },
    ]
    response = gpt_api(messages, temperature, model=model)

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

    return task_desc_list


def propose_task_w_SG(
        perception_info:str, 
        model:str, 
        temperature:float=0.2
)->list:
    """
    perception_info: Scene Graph
    """
    with open(prompts_path["task_propose_w_SG"], 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    prompt = base_prompt.format(perception_info)
    messages = [
        {
            'role': 'user',
            'content': [
                {"type":"text", "text": prompt},
            ]
        },
    ]
    response = gpt_api(messages, temperature, model=model)

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

    return task_desc_list


def decompose_task_w_img(
        task_desc:str, 
        perception_info:List[Image.Image], 
        model:str, 
        temperature:float=0.2
)->Tuple[List[str], List[str], str]:
    """
    perception_info: Pictures token
    """
    with open(prompts_path["task_decompose_w_img"], 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    prompt = base_prompt.format(task_desc)

    imgs_encoded = []
    for img in perception_info:
        imgs_encoded.append(encode_image(img, max(img.size)))

    messages = [
        {
            'role': 'user',
            'content': [
                {"type":"text", "text": prompt},
            ]+[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_encoded}"},
                } for img_encoded in imgs_encoded
            ],
        },
    ]
    response = gpt_api(messages, temperature, model=model)

    subtask_list = []
    primitive_action_str_list = []
    reasoning = response.split("\nanswer:\n")[0]
    
    if ' no.' in response:
        return subtask_list, primitive_action_str_list, reasoning
    subtasks = response.split("\nanswer:\n")[-1].split("\n")  # str list
    for step in subtasks:
        subtask=""
        try:
            subtask = step.split('|')[0].split('.')[1]
            action = step.split('|')[1]
        
            match = re.search(r'\[(.*)\]', action)
            if match:
                primitive_action_str = match.group(1)
                primitive_action_str_list.append(primitive_action_str)
            else:
                # import pdb;pdb.set_trace()
                return subtask_list, primitive_action_str_list, reasoning
        except:
            try:
                subtask = step.split(':')[0].split('.')[1]
                action = step.split(':')[1]

                match = re.search(r'\[(.*)\]', action)
                if match:
                    primitive_action_str = match.group(1)
                    primitive_action_str_list.append(primitive_action_str)
                else:
                    # import pdb;pdb.set_trace()
                    return subtask_list, primitive_action_str_list, reasoning
            except:
                raise 
            # import pdb;pdb.set_trace()
        subtask_list.append(subtask)
    # is_first_time = False
    return subtask_list, primitive_action_str_list, reasoning


def decompose_task_w_SG(
        task_desc:str, 
        perception_info:str, 
        model:str, 
        temperature:float=0.2
)->Tuple[List[str], List[str], str]:
    with open(prompts_path["task_decompose_w_SG"], 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    prompt = base_prompt.format(task_desc, perception_info)

    messages = [{'role': 'user','content': prompt},]
    response = gpt_api(messages, temperature, model=model)

    subtask_list = []
    primitive_action_str_list = []
    reasoning = response.split("\nanswer:\n")[0]
    
    if ' no.' in response:
        return subtask_list, primitive_action_str_list, reasoning
    subtasks = response.split("\nanswer:\n")[-1].split("\n")  # str list
    for step in subtasks:
        subtask=""
        try:
            subtask = step.split('|')[0].split('.')[1]
            action = step.split('|')[1]
        
            match = re.search(r'\[(.*)\]', action)
            if match:
                primitive_action_str = match.group(1)
                primitive_action_str_list.append(primitive_action_str)
            else:
                # import pdb;pdb.set_trace()
                return subtask_list, primitive_action_str_list, reasoning
        except:
            try:
                subtask = step.split(':')[0].split('.')[1]
                action = step.split(':')[1]

                match = re.search(r'\[(.*)\]', action)
                if match:
                    primitive_action_str = match.group(1)
                    primitive_action_str_list.append(primitive_action_str)
                else:
                    # import pdb;pdb.set_trace()
                    return subtask_list, primitive_action_str_list, reasoning
            except:
                raise 
            # import pdb;pdb.set_trace()
        subtask_list.append(subtask)
    # is_first_time = False
    return subtask_list, primitive_action_str_list, reasoning


def infer_if_success_w_img(
        task_desc:str, 
        perception_info:List[Image.Image], 
        model:str, 
        temperature:float=0.2
)->Tuple[Union[bool, str], str]:
    """
    perception_info: Pictures token
    """
    with open(prompts_path["success_infer_w_img"], 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    prompt = base_prompt.format(task_desc)

    imgs_encoded = []
    for img in perception_info:
        imgs_encoded.append(encode_image(img, max(img.size)))

    messages = [
        {
            'role': 'user',
            'content': [
                {"type":"text", "text": prompt},
            ]+[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_encoded}"},
                } for img_encoded in imgs_encoded
            ],
        },
    ]
    response = gpt_api(messages, temperature, model=model)

    is_suc = response.split('answer:\n')[-1]
    suc_info = response.split('answer:\n')[0]

    if 'yes' in is_suc:
        return True, suc_info
    elif 'no' in is_suc:
        return False, suc_info
    else:
        return 'not sure', suc_info
    

def infer_if_success_w_SG(
        task_desc:str, 
        perception_info:List[str], 
        model:str, 
        temperature:float=0.2
)->Tuple[Union[bool, str], str]:
    scene_graph_list_str = ''
    for scene_graph in perception_info:
        scene_graph_list_str += f'  ----------\n{scene_graph}'

    with open(prompts_path["success_infer_w_SG"], 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    prompt = base_prompt.format(task_desc, scene_graph_list_str)

    messages = [{'role': 'user','content': prompt},]
    response = gpt_api(messages, temperature, model=model)

    is_suc = response.split('answer:\n')[-1]
    suc_info = response.split('answer:\n')[0]

    if 'yes' in is_suc:
        return True, suc_info
    elif 'no' in is_suc:
        return False, suc_info
    else:
        return 'not sure', suc_info


if __name__ == '__main__':
    # Task Proposal
    scene_graph = '''  [Nodes]:
    - red block
    - green bowl
    - table
  [Edges]:
    - red block -> on top of -> table
    - green bowl -> on top of -> table'''
    task_list = propose_task(scene_graph=scene_graph, model="vision")
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
    subtask_list, primitive_action_str_list, reasoning = decompose_task(task_desc=task_desc, scene_graph=scene_graph, model="vision")
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
    is_successful = infer_if_success(task_desc=task_desc, scene_graph_list=[scene_graph],model="vision")
    print("Success Infer\n  is_successful:\n\n", is_successful)