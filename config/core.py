import torch


manipulate_root_path="/home/jiechu/Data/TinyRobotBench/manipulate/"
TINYROBOTBENCH_root_path="/home/jiechu/Data/TinyRobotBench/"
dataset_path="/home/jiechu/Data/TinyRobotBench/dataset_test"


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SAM_model_path = {
    "default": "path/to/vit_h",
    "vit_h": "/root/autodl-tmp/checkpoints/SAM/sam_vit_h_4b8939.pth",
    "vit_l": "path/to/vit_l",
    "vit_b": "path/to/vit_b",
}

YOLOv8_model_path = {
    "yolov8n":"/root/autodl-tmp/checkpoints/yolov8/yolov8n.pt",
    "yolov8x":"/root/autodl-tmp/checkpoints/yolov8/yolov8x.pt",
}

prompts_path = {
    "task_propose":"/home/jiechu/Data/TinyRobotBench/manipulate/config/prompts/task_propose.txt",
    "task_decompose":"/home/jiechu/Data/TinyRobotBench/manipulate/config/prompts/task_decompose.txt",
    "success_infer":"/home/jiechu/Data/TinyRobotBench/manipulate/config/prompts/success_infer.txt",
}

gpt_api_key = "sk-es3y9DmO56UVUNFHPN7W2ksixPkR6W8QXKGs0h3FQoggujJQ"
gpt_base_url = "https://api.chatanywhere.tech"