import argparse
import os
import glob
import subprocess
import random
import json
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_config", type=str,
                    help="'ideal', 'bright', 'dark'")

data_root="/root/autodl-tmp/dex-ycb-20210415/"
output_root=data_root

# set configs
dataset_config=args.dataset_config
if not os.path.exists(output_root):
    os.makedirs(output_root)
if dataset_config == "ideal":
    thre_low, thre_high = 0.05, 0.5
    sigma_low, sigma_high = 0, 0
    cutoff_hz_low, cutoff_hz_high = 0, 0
    leak_rate_hz_low, leak_rate_hz_high = 0, 0
    shot_noise_rate_hz_low, shot_noise_rate_hz_high = 0, 0
elif dataset_config == "bright":
    thre_low, thre_high = 0.05, 0.5
    sigma_low, sigma_high = 0.03, 0.05
    cutoff_hz_low, cutoff_hz_high = 200, 200
    leak_rate_hz_low, leak_rate_hz_high = 0.1, 0.5
    shot_noise_rate_hz_low, shot_noise_rate_hz_high = 0, 0
elif dataset_config == "dark":
    thre_low, thre_high = 0.05, 0.5
    sigma_low, sigma_high = 0.03, 0.05
    cutoff_hz_low, cutoff_hz_high = 10, 100
    leak_rate_hz_low, leak_rate_hz_high = 0, 0
    shot_noise_rate_hz_low, shot_noise_rate_hz_high = 1, 10
    
    
valid_folders=glob.glob(os.path.join(data_root,"*-subject-*/*/*/"))

params_collector = {}
stats_record=open(os.path.join(output_root,"stats_record_"+dataset_config+".txt"), "w")

for folder in valid_folders:
    out_filename = folder.split('/')[-2]+".h5"
    out_folder = os.path.join(folder,"event_"+dataset_config)

    #看看该样本是否此前已经生成了事件流
    if not os.path.exists(os.path.join(out_folder,out_filename)):
        stats_record.seek(0)
        stats_record.write(out_folder+out_filename+"creating...                    /n")
    else:
        continue

    # configure paramters
    thres = random.uniform(thre_low, thre_high)
    # sigma should be about 15%~25% range as low and high
    # threshold higher than 0.2: 0.03-0.05
    # threshold lower than 0.2: 15%~25%
    #  sigma = random.uniform(sigma_low, sigma_high)
    sigma = random.uniform(
        min(thres*0.15, sigma_low), min(thres*0.25, sigma_high)) \
        if dataset_config != "ideal" else 0

    leak_rate_hz = random.uniform(leak_rate_hz_low, leak_rate_hz_high)
    shot_noise_rate_hz = random.uniform(
        shot_noise_rate_hz_low, shot_noise_rate_hz_high)

    if dataset_config == "dark":
        # cutoff hz follows shot noise config
        cutoff_hz = shot_noise_rate_hz*10
    else:
        cutoff_hz = random.uniform(cutoff_hz_low, cutoff_hz_high)

    params_collector[os.path.join(out_folder, out_filename)] = {
        "thres": thres,
        "sigma": sigma,
        "cutoff_hz": cutoff_hz,
        "leak_rate_hz": leak_rate_hz,
        "shot_noise_rate_hz": shot_noise_rate_hz}

    # dump bias configs all the time
    with open(os.path.join(output_root,"dvs_params_settings_"+dataset_config+".json"), "w") as f:
        json.dump(params_collector, f, indent=4)
    # # 这是当翻录终端要重新翻录时使用的代码：
    # # 首先，打开并读取文件
    # with open(os.path.join(output_root, "dvs_params_settings.json"), "r") as f:
    #     data = json.load(f)
    # # 然后，更新数据
    # data.update(params_collector)
    # # 最后，写入新的数据
    # with open(os.path.join(output_root, "dvs_params_settings.json"), "w") as f:
    #     json.dump(data, f, indent=4)

    #让输出的frame是源文件夹中的.jpg文件，取出.npz标注文件和.png深度图像文件的影响
    tmp_folder=os.path.join("/root/autodl-tmp/tmp_folder_"+dataset_config,folder.split("dex-ycb-20210415/")[-1])
    shutil.copytree(folder, tmp_folder)
    for file in os.listdir(tmp_folder):
        if ".jpg" not in file:
            rm_path=os.path.join(tmp_folder,file)
            if os.path.isfile(rm_path):
                os.remove(rm_path)
            else:
                shutil.rmtree(rm_path)
    v2e_command = [
        "python",
        "v2e.py",
        "-i", tmp_folder,
        "-o", out_folder,
        "--overwrite",
        "--unique_output_folder", "false",
        "--davis_output",
        "--dvs_h5",out_filename,
        "--dvs_aedat2","None",
        "--dvs_text","None",
        "--no_preview",
        "--dvs_exposure", "duration", "0.033",
        "--skip_video_output",
        "--input_frame_rate", "30",
        "--input_slowmotion_factor", "1",
        "--slomo_model","/root/autodl-tmp/v2e/input/SuperSloMo39.ckpt",
        "--auto_timestamp_resolution", "true",
        "--pos_thres", "{}".format(thres),
        "--neg_thres", "{}".format(thres),
        "--sigma_thres", "{}".format(sigma),
        "--cutoff_hz", "{}".format(cutoff_hz),
        "--leak_rate_hz", "{}".format(leak_rate_hz),
        "--shot_noise_rate_hz", "{}".format(shot_noise_rate_hz),
        "--dvs640"
        ]
        
    print("\n\n\n********************************************")
    print("input folder:",tmp_folder)
    print("output folder:", out_folder)
    print("output file:", out_filename)
    print("********************************************\n")
    subprocess.run(v2e_command)
    shutil.rmtree(tmp_folder)
    stats_record.seek(0)
    stats_record.write(out_folder+out_filename+"created successfully.              /n")

stats_record.close()