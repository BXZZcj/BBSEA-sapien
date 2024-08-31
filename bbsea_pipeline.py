import os
import json
from typing import Callable

from scene.SimplePickPlace_v4 import SimplePickPlaceScene
from scene.core import TaskScene
from llm_interaction import *
from config.core import dataset_path


def main(
        scene: TaskScene,
        get_perception_info: Callable,
):
    #########################################################
    # Initialization the task scene
    #########################################################
    Pick = TableTop.primitives.Pick
    Push = TableTop.primitives.Push
    PlaceOn = TableTop.primitives.PlaceOn
    PlaceAt = TableTop.primitives.PlaceAt
    DrawerOpen = TableTop.primitives.DrawerOpen
    DrawerClose = TableTop.primitives.DrawerClose
    Press = TableTop.primitives.Press


    perception_info = get_perception_info()
    task_index=1
    is_task_subtask_suc_list=[]
    for i in range(1):        
        #########################################################
        # Task Propose
        #########################################################
        task_list = propose_task_w_img(scene_graph=task_SG, model="vision")

        #########################################################
        # Task Decompose
        #########################################################
        for task_desc in task_list[:3]:    
            subtask_list, primitive_action_str_list, reasoning = decompose_task(task_desc=task_desc, scene_graph=task_SG, model="vision") 
            
            TableTop.set_task_index(task_index=task_index)
            with open(os.path.join(TableTop.get_task_dir(), "description.txt"), "w") as f:
                f.write(f"task description:\n{task_desc}\n\nreasoning:\n{reasoning}\n\npre scenegraph:\n{task_SG}\n\n")

            #########################################################
            # Task Execution
            #########################################################
            task_SG_list=[task_SG]
            subtask_index = 1
            subtask_SG = task_SG
            is_subtask_suc_tmp_list = []
            for subtask, primitive_action_str in zip(subtask_list, primitive_action_str_list):
                subtask_SG_list=[subtask_SG]

                TableTop.set_task_index(task_index=task_index, subtask_index=subtask_index)
                with open(os.path.join(TableTop.get_subtask_dir(), "description.txt"), "w") as f:
                    f.write(f"subtask:\n{subtask}\n\npre scenegraph:\n{subtask_SG}\n\nprimitive action:\n{primitive_action_str}\n\n")
                
                TableTop.set_step_index(0)
                print(primitive_action_str)
                for primitive in primitive_action_str.strip().split(';'):
                    try:
                        exec(primitive.strip())
                    except Exception as e:
                        print(f"A error occured when execute {primitive}:\n{e}")
                TableTop.set_step_index(0)

                subtask_SG = TableTop.get_scene_graph()
                subtask_SG_list.append(subtask_SG)

                is_subtask_suc_info = infer_if_success(task_desc=subtask, scene_graph_list=subtask_SG_list, model="vision")
                is_subtask_suc_tmp_list.append(is_subtask_suc_info[0])
                with open(os.path.join(TableTop.get_subtask_dir(), "description.txt"), "a") as f:
                    f.write(f"post scenegraph:\n{subtask_SG}\n\nis successful:\n{is_subtask_suc_info[0]}\n{is_subtask_suc_info[1]}")

                task_SG_list += subtask_SG_list

                subtask_index += 1

            #########################################################
            # Success Judgement
            #########################################################
            is_task_suc_info = infer_if_success(task_desc=task_desc, scene_graph_list=task_SG_list, model="vision")
            is_task_subtask_suc_list.append({f"task":is_task_suc_info[0],"subtask":is_subtask_suc_tmp_list})
            
            task_SG = TableTop.get_scene_graph()

            TableTop.set_task_index(task_index=task_index)
            with open(os.path.join(TableTop.get_task_dir(), "description.txt"), "a") as f:
                f.write(f"post scenegraph:\n{task_SG}\n\nis successful:\n{is_task_suc_info[0]}\n{is_task_suc_info[1]}")

            task_index+=1
        
    with open(os.path.join(dataset_path, "result.json"), "w") as f:
        json.dump(is_task_subtask_suc_list, f, indent=4)
    # is_task_suc_list = [i['task'] for i in is_task_subtask_suc_list]
    # is_subtask_suc_list = [j for i in is_task_subtask_suc_list for j in i['subtask']]


if __name__=="__main__":
    TableTop = SimplePickPlaceScene()
    
    main(scene=TableTop, get_perception_info=TableTop.get_rgb_picture)