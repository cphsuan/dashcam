import json
import os
import sys
import shutil
import glob

import cv2
import pandas as pd
import numpy as np

# ParentPath
parent_path = '/media/hsuan/data/VIL100/'
# json_path = os.listdir(os.path.join(parent_path, "Json/"))
# allFileList = sorted(json_path)

img_path = os.path.join(parent_path, "JPEGImages/0_Road014_Trim004_frames/00000.jpg")
json_path = os.path.join(parent_path, "Json/0_Road014_Trim004_frames/00000.jpg.json")

print(json_path)
with open(json_path) as f:
    data = json.load(f)
print(data)