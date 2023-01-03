import base64
import json
import io
import os
import os.path as osp
import argparse
from copy import deepcopy
import cv2
from tqdm import tqdm

from PIL import Image
from labelme import PY2
from labelme import QT4

import shutil

parser = argparse.ArgumentParser('Remake Warp Json...')
parser.add_argument('--dataset-WarpIMGdir', type=str , default='/media/hsuan/data/WarpDataset/VIL100/JPEGImages/', help='path to Warp IMG dataset')
parser.add_argument('--output-IMGdir', type=str , default='/media/hsuan/data/WarpDataset/Label/', help='path to output Img')

args = parser.parse_args()

if __name__ == "__main__":
    # label
    label = '/media/hsuan/data/WarpDataset/label.txt'
    with open(label, encoding='utf-8') as f:
        class_names = f.readlines()
    names = []
    indexs = []
    for data in class_names:
        name,index = data.split(' ')
        names.append(name)
        indexs.append(int(index))
    print(names)

    for name in names:
        if os.path.isdir(os.path.join(args.output_IMGdir,name)):
            shutil.rmtree(os.path.join(args.output_IMGdir,name))

        os.makedirs(os.path.join(args.output_IMGdir,name))

    WarpIMGdir = args.dataset_WarpIMGdir
    allFileList = sorted(os.listdir(WarpIMGdir))

    pbar = tqdm(total=len(allFileList),desc=f'FIle Processing : ' , mininterval=0.3)
    for fileidx, file in enumerate(allFileList):

        imgPerFile = os.path.join(WarpIMGdir, file)
        fileExt = r".jpg"
        imgPerFile = sorted([_ for _ in os.listdir(imgPerFile) if _.endswith(fileExt)])

        for frameIndex, frame in enumerate(imgPerFile):
            imgPath = os.path.join(WarpIMGdir, file, frame)
            jsonPath = os.path.join(WarpIMGdir, file, frame.replace('jpg', 'json'))

            if os.path.isfile(jsonPath):
                with open(jsonPath) as f:
                    jsonData = json.load(f)
                
                img0 = cv2.imread(imgPath)

                for index, lane in enumerate(jsonData["shapes"]):
                    label = lane['label']
                    points = lane['points']

                    cropped = img0[ 0 : int(points[1][1]) if points[1][1] > 0 else 0,int(points[0][0]) if points[0][0] > 0 else 0:int(points[1][0]) if points[1][0] > 0 else 0]
                    # print(os.path.join(args.output_IMGdir,label,file+'_'+str(index)+'_'+frame))
                    cv2.imwrite(os.path.join(args.output_IMGdir,label,file+'_'+str(index)+'_'+frame),cropped)
                    # cv2.imshow("img", cropped)
                    # cv2.waitKey(0)
            
        pbar.update(1)
    pbar.close()
