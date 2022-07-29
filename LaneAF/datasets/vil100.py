import json
import os
import sys
import shutil
import glob

import cv2
import pandas as pd
import numpy as np

# ParentPath
ParentPath = '/media/hsuan/data/VIL100/'
allFileList = os.listdir(os.path.join(ParentPath, "Json/"))
allFileList = sorted(allFileList)
# read Json
LanePath = []
id = []
lane_id = []
attribute = []
occlusion = []
for k in range(len(allFileList)):
    ChildPath = os.path.join(ParentPath, "Json", allFileList[k], '*.json')
    JsonPath = sorted(glob.glob(ChildPath))

    for j in range(len(JsonPath)):
        with open(JsonPath[j]) as f:

            data = json.load(f)
            lane_info = data["annotations"]["lane"]

            for i in range(len(lane_info)):
                LanePath.append('/'.join(JsonPath[j].split('/')[6:8]))
                # LanePath.append(JsonPath[j].split('/')[6:7])
                id.append(lane_info[i]["id"])
                lane_id.append(lane_info[i]["lane_id"])
                attribute.append(lane_info[i]["attribute"])
                occlusion.append(lane_info[i]["occlusion"])

# summary
df = pd.DataFrame((zip(LanePath, id, lane_id, attribute, occlusion)),
                  columns=['LanePath', 'id', 'lane_id', 'attribute', 'occlusion'])
df[['LanePath']] = df[['LanePath']].astype(str)
print(df)

group = df.groupby("attribute")["LanePath"].nunique()
print(group)
# print(group.size().reset_index(name='counts'))
# print(job_group['attribute'].value_counts())
above_13 = df[df["attribute"] == 13]
print(above_13)
# result = pd.value_counts(attribute)
# print(result)



# visualize
image = '4_Road027_Trim006_frames/00150.jpg'
imagepath = os.path.join(ParentPath, "JPEGImages/", image)
print(imagepath)
img = cv2.imread(imagepath)

imagejsonpath = os.path.join(ParentPath, "Json/", (image + ".json"))
print(imagejsonpath)
imgjson = json.load(open(imagejsonpath))
imgjsonlane_info = imgjson["annotations"]["lane"]
lanecolor = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
             (0, 255, 255), (255, 0, 255), (255, 255, 0)]
for i in range(len(lane_info)):
    points = imgjsonlane_info[i]["points"]
    firstpoint = points[0]
    lastpoint = points[-1]

    pts = []

    for point in points:
        pts.append([int(point[0])-10, int(point[1])])
        pts.append([int(point[0])+10, int(point[1])])
        cv2.circle(img, (int(point[0]), int(point[1])), 1, lanecolor[i], 2)

    ## (1) Crop the bounding rect
    pts = np.array(pts)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    # box = [[int(firstpoint[0])-10, int(firstpoint[1])],
    #         [int(firstpoint[0])+10,int(firstpoint[1])],
    #         [int(lastpoint[0])+10, int(lastpoint[1])],
    #         [int(lastpoint[0])-10, int(lastpoint[1])]]
    # box = np.int0(box)
    # cv2.drawContours(img, [box], 0, (255, 0, 0), 1)

    cv2.circle(img, (int(firstpoint[0]), int(
        firstpoint[1])), 1, (255, 255, 0), 2)
    cv2.circle(img, (int(lastpoint[0]), int(
        lastpoint[1])), 1, (255, 0, 255), 2)
    cv2.imshow("img_out", dst2)
    cv2.waitKey(0)

cv2.imshow("img_out", img)
cv2.waitKey(0)
