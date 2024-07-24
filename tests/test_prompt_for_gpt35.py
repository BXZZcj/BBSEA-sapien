import os
import re
from openai import OpenAI
from config.core import prompts_path, gpt_api_key, gpt_base_url


client = OpenAI(
    api_key=gpt_api_key,
    base_url=gpt_base_url
)


scene_graph = '''  [Nodes]:
    - red block
    - green bowl
    - table
  [Edges]:'''

task_desc = 'put the red block in the green bowl'


base_prompt = '''You are a robot with a single arm in a tabletop robot manipulation environment. 
Given a task description and a list of scene graphs, the goal is to infer if the task has been completed successfully. The list of scene graphs are arranged in chronological order (the first is the initial scene graph, and the last is the scene graph after the policy is executed). The nodes in the scene graph indicate the name, the state, the position and the bounding box (in the unit of meter) of an object. The positive direction of the x-axis represents the front, and the negative direction represents the rear. The positive direction of the y-axis represents the right side, and the negative direction represents the left side. The positive direction of the z-axis represents upward, and the negative direction represents downward. One type of object may have multiple instances, nodes use numbers to distinguish them. The edges indicate the spatial relationships between the objects, and no edges between two objects (nodes) means the two objects are far apart. The position of the robot is (0.0, 0.0, 0.0). 
Note that you should firstly reason whether the task is completed based on the task description and scene graphs, then output the answer. The answer should be "yes" or "no" or "not sure". Below are some examples:
```
task description: move the blue block on the plate
scene graph list:
  ----------
  [Nodes]:
    - red block -- position: [0.40, -0.20, 0.08], x_range: [0.37, 0.43], y_range: [-0.23, -0.17], z_range: [0.05, 0.11]
    - blue block -- position: [0.30, -0.20, 0.08], x_range: [0.27, 0.33], y_range: [-0.23, -0.17], z_range: [0.05, 0.12]
    - plate -- position: [0.52, -0.21, 0.09], x_range: [0.44, 0.60], y_range: [-0.30, -0.13], z_range: [0.07, 0.11]
  [Edges]:
  ----------
  [Nodes]:
    - red block -- position: [0.48, -0.20, 0.15], x_range: [0.45, 0.51], y_range: [-0.23, -0.17], z_range: [0.13, 0.18]
    - blue block -- position: [0.30, -0.20, 0.08], x_range: [0.27, 0.33], y_range: [-0.23, -0.17], z_range: [0.05, 0.12]
    - plate -- position: [0.52, -0.21, 0.09], x_range: [0.44, 0.60], y_range: [-0.30, -0.13], z_range: [0.07, 0.11]
  [Edges]:
    - red block -> on top of -> blue block
success metric: The blue block is on the plate
reasoning: The scene graph indicates the blue block is not on the plate. So the task was not accomplished.
answer:
  no
```
```
task description: stack the red block and the blue block on the plate
scene graph list:
  ----------
  [Nodes]:
    - red block -- position: [0.40, -0.29, 0.08], x_range: [0.37, 0.43], y_range: [-0.32, -0.26], z_range: [0.05, 0.11]
    - blue block -- position: [0.32, -0.20, 0.08], x_range: [0.29, 0.35], y_range: [-0.23, -0.17], z_range: [0.05, 0.11]
    - plate -- position: [0.52, -0.21, 0.09], x_range: [0.44, 0.60], y_range: [-0.30, -0.13], z_range: [0.07, 0.11]
  [Edges]:
  ----------
  [Nodes]:
    - red block -- position: [0.40, -0.29, 0.08], x_range: [0.37, 0.43], y_range: [-0.32, -0.26], z_range: [0.05, 0.11]
    - blue block -- position: [0.48, -0.20, 0.15], x_range: [0.45, 0.51], y_range: [-0.23, -0.17], z_range: [0.12, 0.18]
    - plate -- position: [0.52, -0.21, 0.09], x_range: [0.44, 0.60], y_range: [-0.30, -0.13], z_range: [0.07, 0.11]
  [Edges]:
    - blue block -> on top of -> plate
  ----------
  [Nodes]:
    - red block -- position: [0.48, -0.20, 0.21], x_range: [0.45, 0.51], y_range: [-0.23, -0.17], z_range: [0.18, 0.24]
    - blue block -- position: [0.48, -0.20, 0.15], x_range: [0.45, 0.51], y_range: [-0.23, -0.17], z_range: [0.12, 0.18]
    - plate -- position: [0.52, -0.21, 0.09], x_range: [0.44, 0.60], y_range: [-0.30, -0.13], z_range: [0.07, 0.11]
  [Edges]:
    - blue block -> on top of -> plate
    - red block -> on top of -> blue block
success metric: The blue block is on top of the plate and the red block is on top of the blue block.
reasoning: The scene graph indicates the blue block is on top of the plate and the red block is on top of the blue block. So "stack the red block and the blue block on the plate" has been accomplished.
answer:
  yes
```
```
task description: move the red block to the left of the plate
scene graph list:
  ----------
  [Nodes]:
    - red block -- position: [0.40, -0.20, 0.08], x_range: [0.37, 0.43], y_range: [-0.23, -0.17], z_range: [0.05, 0.11]
    - green block -- position: [0.05, -0.26, 0.08], x_range: [0.02, 0.08], y_range: [-0.29, -0.23], z_range: [0.05, 0.11]
    - plate -- position: [0.52, -0.21, 0.09], x_range: [0.44, 0.60], y_range: [-0.30, -0.13], z_range: [0.07, 0.11]
  [Edges]:
  ----------
  [Nodes]:
    - red block -- position: [0.40, -0.05, 0.08], x_range: [0.37, 0.43], y_range: [-0.08, -0.02], z_range: [0.05, 0.11]
    - green block -- position: [0.05, -0.26, 0.08], x_range: [0.02, 0.08], y_range: [-0.29, -0.23], z_range: [0.05, 0.11]
    - plate -- position: [0.52, -0.21, 0.09], x_range: [0.44, 0.60], y_range: [-0.30, -0.13], z_range: [0.07, 0.11]
  [Edges]:
success metric: The red block is on the left of the plate. In other words, the y of the red block is less than the minimum value of the plate's y_range.
reasoning: The y of the red block is -0.05, which is larger than the minimum value of the plate's y_range (-0.30). So "move the red block to the left of the plate" has been accomplished.
answer:
  no
```
Now I'd like you to help me infer whether the proposed task is completed. You should read the task description and the scene graph list I provide with you, and then complete the "success metric", "reasoning" and "answer".
```
task description: 
{}
scene graph list:
{}
success metric: '''


def gpt_35_api(messages: list, temperature):
    completion = client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=messages, temperature=temperature)
    return completion.choices[0].message.content


if __name__=="__main__":
    prompt = base_prompt.format(task_desc, scene_graph)
    messages = [{'role': 'user','content': prompt},]
    response = gpt_35_api(messages, 0.9)
    print(response)