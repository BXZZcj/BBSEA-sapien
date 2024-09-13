import torch


manipulate_root_path="/home/admin01/Data/BBSEA-sapien/"
# TINYROBOTBENCH_root_path="/home/jiechu/Data/TinyRobotBench/"
dataset_path="/home/admin01/Data/generated_dataset"


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

prompts_dir = "/home/admin01/Data/BBSEA-sapien/config/prompts/"

prompts_path = {
    "task_propose_w_img":"/home/admin01/Data/BBSEA-sapien/config/prompts/task_propose_w_img.txt",
    "task_propose_w_SG":"/home/admin01/Data/BBSEA-sapien/config/prompts/task_propose_w_SG.txt",
    "task_decompose_w_img":"/home/admin01/Data/BBSEA-sapien/config/prompts/task_decompose_w_img.txt",
    "task_decompose_w_SG":"/home/admin01/Data/BBSEA-sapien/config/prompts/task_decompose_w_SG.txt",
    "success_infer_w_img":"/home/admin01/Data/BBSEA-sapien/config/prompts/success_infer_w_img.txt",
    "success_infer_w_SG":"/home/admin01/Data/BBSEA-sapien/config/prompts/success_infer_w_SG.txt",
}

chatanywhere_api_key = "sk-es3y9DmO56UVUNFHPN7W2ksixPkR6W8QXKGs0h3FQoggujJQ"
chatanywhere_base_url = "https://api.chatanywhere.tech"

azure_api_key="c5ae1c3ed4e74e209fbb45cfc8cb3b2f"
azure_endpoint="https://gpt4v-0.openai.azure.com/"
azure_api_version="2023-12-01-preview"
