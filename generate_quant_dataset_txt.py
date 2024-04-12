import os
from numpy import random as rd


img_dir = r"/home/yuyang/data/make3d/train/image"
txt_file_save_pth = r"dataset.txt"
quant_img_count = 200
lines = [os.path.join(img_dir, name) + "\n" for name in os.listdir(img_dir) if name.lower().endswith("png") or name.lower().endswith("jpg")]
if len(lines) > quant_img_count:
    lines = rd.choice(lines, quant_img_count, replace=False)
lines[-1] = lines[-1].strip("\n")
with open(txt_file_save_pth, "w", encoding="utf-8") as file:
    file.writelines(lines)