import base64
import json
import io
import os
import os.path as osp
import shutil
from unicodedata import name
from copy import deepcopy
import cv2
import numpy as np
from tqdm import tqdm

from PIL import Image
from labelme import PY2
from labelme import QT4

BDDSegJsonPath = '/media/hsuan/data/BDD/bdd100k_pan_seg_labels_trainval/bdd100k/labels/pan_seg/polygons/pan_seg_train.json'
with open(BDDSegJsonPath) as f:
    BDDJson = json.load(f)
# print(len(BDDJson)) #7000

classes = ["ego vehicle", "rider", "bicycle", "bus", "car","caravan", "motorcycle", "trailer", "train", "truck"]

labelmejson = {"version": "5.0.1", "flags": {}, "shapes": None, "imagePath": None, "imageData": None, "imageHeight":None, "imageWidth":None} 
labeljson0 = {"label": None, "points": None, "group_id": None, "shape_type": "rectangle","flags": {}}

print("start")
egoVeh = []
for i in tqdm(range(len(BDDJson))):
    imgpath = os.path.join('/media/hsuan/data/BDD/bdd100k_images_10k/bdd100k/images/10k/train/', BDDJson[i]["name"])
    desdir = '/media/hsuan/data/BDD_egoVeh/train/'
    img = cv2.imread(imgpath)
    imgH, imgW, _ = np.shape(img)
    labels = BDDJson[i]["labels"]
    shapes = []
    for j in range(len(labels)):
        
        if labels[j]["category"] in classes:
            labeljson= deepcopy(labeljson0)
            labeljson["label"] = labels[j]["category"]

            x, y, w, h = cv2.boundingRect(np.array(labels[j]["poly2d"][0]["vertices"], dtype='int64'))
            labeljson["points"] = [[x, y],[x+w, y+h]]
            shapes.append(labeljson)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 


    if len(shapes) != 0:
        labelmejson["shapes"] = shapes
        labelmejson["imageHeight"] = imgH
        labelmejson["imageWidth"] = imgW

        shutil.copy(imgpath, desdir)

        labelmejson["imagePath"] = os.path.join(desdir, BDDJson[i]["name"])


        imageData = Image.open(os.path.join(desdir, BDDJson[i]["name"]))
        with io.BytesIO() as f:
            ext = osp.splitext(BDDJson[i]["name"])[1].lower()
            if PY2 and QT4:
                format = 'PNG'
            elif ext in ['.jpg', '.jpeg']:
                format = 'JPEG'
            else:
                format = 'PNG'
            imageData.save(f, format=format)
            f.seek(0)
            image_bytes = f.read()

        imageData=base64.b64encode(image_bytes).decode('utf-8')
        labelmejson["imageData"] = imageData

        if isinstance(labelmejson, bytes):
            labelmejson = str(labelmejson, encoding='utf-8')

        with open(os.path.join(desdir,BDDJson[i]["name"].replace('jpg', 'json')), "w") as f:
            json.dump(labelmejson, f, indent = 4)

    # cv2.imshow("img_out", img)
    # cv2.waitKey(0)



