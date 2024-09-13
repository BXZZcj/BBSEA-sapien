import os

base_dir = "/home/admin01/Data/BBSEA-sapien/assets/object/ycb/"
for sample in os.listdir(base_dir):
    mtl_f=f"/home/admin01/Data/BBSEA-sapien/assets/object/ycb/{sample}/google_16k/textured.mtl"

    with open(mtl_f, "r") as file:
        lines=file.readlines()

    if len(lines)>=3:
        lines[2]=lines[2].strip() + '\n'

    with open(mtl_f, 'w') as file:
        file.writelines(lines)