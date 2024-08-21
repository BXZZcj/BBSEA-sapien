import json
import os

from config.core import dataset_path


def main():
    with open(os.path.join(dataset_path, 'result.json'), 'r') as f:
        result = json.load(f)
    
    if len(result)==0:
        print("Nothing in the result.json")
        return None

    n_task_suc = 0
    n_task = len(result)
    n_subtask_suc = 0
    n_subtask = 0

    for item in result:
        if item['task']:
            n_task_suc += 1
        
        n_subtask_suc += sum(item['subtask'])
        n_subtask += len(item['subtask'])

    task_suc_rate = n_task_suc / n_task
    subtask_suc_rate = n_subtask_suc / n_subtask if n_subtask else 0

    print(f"Task True Rate: {task_suc_rate * 100:.2f}%")
    print(f"Subtask True Rate: {subtask_suc_rate * 100:.2f}%")



if __name__=="__main__":
    main()