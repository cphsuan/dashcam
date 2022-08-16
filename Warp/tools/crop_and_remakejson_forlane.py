import base64
import json
import io
import os
import os.path as osp
import argparse
from copy import deepcopy
import cv2

from PIL import Image
from labelme import PY2
from labelme import QT4

parser = argparse.ArgumentParser('Remake Warp Json...')
parser.add_argument('--dataset-WarpIMGdir', type=str , default='/media/hsuan/data/WarpDataset/VIL100/JPEGImages/', help='path to Warp IMG dataset')
parser.add_argument('--output-IMGdir', type=str , default='/media/hsuan/data/WarpDataset/Label/', help='path to output Img')
args = parser.parse_args()

if __name__ == "__main__":
    # ParentPath
    WarpIMGdir = args.dataset_WarpIMGdir
    # allFileList = sorted(os.listdir(os.path.join(parentPath, "JPEGImages")))
    file = "0_Road014_Trim004_frames"
    imgPerFile = os.path.join(WarpIMGdir, file)
    fileExt = r".jpg"
    imgPerFile = sorted([_ for _ in os.listdir(imgPerFile) if _.endswith(fileExt)])
    print(imgPerFile)

    for frameIndex, frame in enumerate(imgPerFile):
        imgPath = os.path.join(WarpIMGdir, file, frame)
        jsonPath = os.path.join(WarpIMGdir, file, frame.replace('jpg', 'json'))
        
        print(imgPath)
        print(jsonPath)

        with open(jsonPath) as f:
            jsonData = json.load(f)
        
        img0 = cv2.imread(imgPath)

        for index, lane in enumerate(jsonData["shapes"]):
            label = lane['label']
            points = lane['points']

            print(int(points[0][1]),int(points[1][1]),int(points[0][0]),int(points[1][0]))
            cropped = img0[ 0 : int(points[1][1]),int(points[0][0]):int(points[1][0])]
            
            lanename = file+'_'+str(index)+'_'+frame
            print(lanename)
            cv2.imwrite(os.path.join(args.output_IMGdir,label,lanename),cropped)
            # cv2.imshow("img", cropped)
            # cv2.waitKey(0)

            # print(jsonData["shapes"])
            # print(len(jsonData["shapes"]))
        # input()