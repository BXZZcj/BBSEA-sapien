import cv2
import os
import numpy as np

from config import dataset_path

def create_video_from_RGB(
        RGB_folder: str, 
        output_video_file: str, 
        fps=30, 
        sort_numerically=True
    ):
    RGB_folder=os.path.join(RGB_folder, "RGB")
    files = [os.path.join(RGB_folder, f) for f in os.listdir(RGB_folder) if f.endswith('.png')]
    
    if sort_numerically:
        files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    
    frame = cv2.imread(files[0])
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Read each file and write it to the output video
    for file in files:
        frame = cv2.imread(file)
        out.write(frame)  # Write frame to video

    # Release everything when job is finished
    out.release()
    print(f"Video saved as {output_video_file}")

# Example usage
image_folder = os.path.join(dataset_path, "task_0001/subtask_001/Backward")
output_video_file = 'output_video.avi'
create_video_from_RGB(image_folder, output_video_file, fps=100)