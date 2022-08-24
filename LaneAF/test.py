import sys
import cv2
import numpy    as np

import asyncio
from argparse import ArgumentParser
sys.path.append(r'/home/hsuan/Thesis/mmdetection/')
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',default='/media/hsuan/data/VIL100/JPEGImages/1_Road018_Trim006_frames/00006.jpg' , help='Image file')
    parser.add_argument('--detconfig',default='mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py' ,help='Config file')
    parser.add_argument('--detcheckpoint',default='mmdetection/work_dirs/latest.pth' ,help='Checkpoint file')
    parser.add_argument('--detout-file', default=None, help='Path to output file')
    parser.add_argument('--detpalette', default='coco', choices=['coco', 'voc', 'citys', 'random'], help='Color palette used for visualization')
    parser.add_argument('--detscore-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def drawBoundingBox(img, bboxs):
    for box in bboxs:
        x1,y1,x2,y2 = (box['x1'], box['y1'], box['x2'], box['y2'])
        label = box['label']
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 6)
        fontFace = cv2.FONT_HERSHEY_COMPLEX
        fontScale = 0.5
        thickness = 1
        labelSize = cv2.getTextSize(label, fontFace, fontScale, thickness)
        _x1 = x1 # bottomleft x of text
        _y1 = y1 # bottomleft y of text
        _x2 = x1+labelSize[0][0] # topright x of text
        _y2 = y1-labelSize[0][1] # topright y of text
        cv2.rectangle(img, (_x1,_y1), (_x2,_y2), (0,255,0), cv2.FILLED) # text background
        cv2.putText(img, label, (x1,y1), fontFace, fontScale, (0,0,0), thickness)
    return img


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.detconfig, args.detcheckpoint, device='cuda:0')
    # test a single image
    result = inference_detector(model, args.img)
    print('result')
    print(result[0])
    print('class')
    print(model.CLASSES)

    img=cv2.imread(args.img)
    print(np.shape(img))
    bboxs = []
    for rec in result[0]:
        x1, y1, x2, y2, score = rec[0], rec[1], rec[2], rec[3], rec[4]

        if score>= args.detscore_thr :
            print(x1, y1, x2, y2)
            box = {'label': model.CLASSES[0], 'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
            bboxs.append(box)
    drawBoundingBox(img, bboxs)

    cv2.imshow("img_out", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    args = parse_args()
    main(args)
