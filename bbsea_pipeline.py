from scene import SimplePickPlaceScene
from llm_interaction import propose_task, decompose_task, infer_if_success




def main():
    TableTop = SimplePickPlaceScene()
    TableTop.scene.step() 
    TableTop.scene.update_render()

    Pick = TableTop.primitives.Pick
    Push = TableTop.primitives.Push
    PlaceOn = TableTop.primitives.PlaceOn
    PlaceAt = TableTop.primitives.PlaceAt

    init_scene_graph = TableTop.get_scene_graph()

    task_list = propose_task(init_scene_graph)
    print("Task Proposal\n  task_list:\n\n", task_list)

    # Task Decompose
    scene_graph_list = []
    is_successful_list = []
    for task_desc in task_list:
        subtask_list, primitive_action_str_list, primitive_action_list, reasoning = decompose_task(task_desc, init_scene_graph)
        print("Task Decompose")
        print("  reasoning:\n\n", reasoning)
        print("\n  subtask_list:\n\n", subtask_list)
        print("\n  primitive_action_str_list: \n\n", primitive_action_str_list)

        for primitive_action_str in primitive_action_str_list:
            exec(primitive_action_str)
            scene_graph_list.append(TableTop.get_scene_graph())

        # Success Infer  
        is_successful = infer_if_success(task_desc, scene_graph_list)
        print("Success Infer\n  is_successful:\n\n", is_successful)
        is_successful_list.append(is_successful)

    # exec("Pick('box'); PlaceOn('green_pad')")



if __name__=="__main__":
    main()