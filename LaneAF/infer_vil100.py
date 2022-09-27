from copy import deepcopy
from distutils.log import error
import sys
import os
import json
from datetime import datetime
from pickle import FALSE, OBJ, TRUE
from statistics import mean
import argparse
import numpy as np
import cv2
from pandas import array
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import SpectralClustering
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import datasets.transforms as tf
from datasets.tusimple import TuSimple, get_lanes_tusimple
from models.dla.pose_dla_dcn import get_pose_net
from models.erfnet.erfnet import ERFNet
from models.enet.ENet import ENet
from utils.affinity_fields import decodeAFs
from utils.metrics import match_multi_class, LaneEval
from utils.visualize import tensor2image, create_viz
from tools.perspective_correction import *
from tools.CustomerClass import *
from tools.detect_tool import *
from tools.camera_correction import *
#for mmdetection
sys.path.append(r'/home/hsuan/Thesis/mmdetection/')
from mmdet.apis import (inference_detector, init_detector)
import pysnooper

parser = argparse.ArgumentParser('Options for inference with LaneAF models in PyTorch...')
parser.add_argument('--input_video', type=str , default='/media/hsuan/data/VIL100/videos/1_Road018_Trim006_frames.avi', help='path to input video')
parser.add_argument('--dataset-dir', type=str , default='/media/hsuan/data/CULane', help='path to dataset')
parser.add_argument('--snapshot', type=str, default='/home/hsuan/Thesis/LaneAF/laneaf-weights/culane-weights/dla34/net_0033.pth', help='path to pre-trained model snapshot')
parser.add_argument('--seed', type=int, default=1 , help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
# for mmdetection
parser.add_argument('--detconfig',default='mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py' ,help='Config file')
parser.add_argument('--detcheckpoint',default='mmdetection/work_dirs/latest.pth' ,help='Checkpoint file')
parser.add_argument('--detout-file', default=None, help='Path to output file')
parser.add_argument('--detpalette', default='coco', choices=['coco', 'voc', 'citys', 'random'], help='Color palette used for visualization')
parser.add_argument('--detscore-thr', type=float, default=0.3, help='bbox score threshold')
args = parser.parse_args()
# check args
if args.input_video is None:
    assert False, 'Path to image not provided!'
if args.snapshot is None:
    assert False, 'Model snapshot not provided!'
# set batch size to 1 for visualization purposes
args.batch_size = 1
# setup cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()

# load args used from training snapshot (if available)
if os.path.exists(os.path.join(os.path.dirname(args.snapshot), 'config.json')):
    with open(os.path.join(os.path.dirname(args.snapshot), 'config.json')) as f:
        json_args = json.load(f)
    # augment infer args with training args for model consistency
    if 'backbone' in json_args.keys():
        args.backbone = json_args['backbone']
    else:
        args.backbone = 'dla34'

# set random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 1}


def LaneAF(img, net):
    """
    LaneAF: Robust Multi-Lane Detection with Affinity Fields
    https://github.com/sel118/LaneAF 
    """
    net.eval()
    img_transforms = transforms.Compose([
        tf.GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
        tf.GroupNormalize(mean=([0.485, 0.456, 0.406], (0, )), std=([0.229, 0.224, 0.225], (1, ))),])
    
    # TODO mask processing #暫時不處理

    input_img, _ = img_transforms((img, img))
    input_img = torch.from_numpy(input_img).permute(2, 0, 1).contiguous().float()
    input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
    input_seg, input_mask, input_af = torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan'))
    if args.cuda:
        input_img = torch.tensor(input_img).cuda()
        input_seg = input_seg.cuda()
        input_mask = input_mask.cuda()
        input_af = input_af.cuda()

    # do the forward pass
    outputs = net(input_img)[-1]

    # convert to arrays
    img = tensor2image(input_img.detach(), np.array(
        [0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
    mask_out = tensor2image(torch.sigmoid(outputs['hm']).repeat(1, 3, 1, 1).detach(),
                            np.array([0.0 for _ in range(3)], dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
    vaf_out = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    haf_out = np.transpose(outputs['haf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))

    # decode AFs to get lane instances
    seg_out = decodeAFs(mask_out[:, :, 0], vaf_out,haf_out, fg_thresh=128, err_thresh=5)

    ###
    if torch.any(torch.isnan(input_seg)):
        # if labels are not available, skip this step
        pass
    else:
        # if test set labels are available
        # re-assign lane IDs to match with ground truth
        seg_out = match_multi_class(seg_out.astype(np.int64), input_seg[0, 0, :, :].detach().cpu().numpy().astype(np.int64))
    img_out = create_viz(img, seg_out.astype(np.uint8), mask_out, vaf_out, haf_out)
    ### LANEAF RESULT VIS ### 把create_viz拿出來
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    img_out_LaneAF = np.ascontiguousarray(img, dtype=np.uint8)
    return seg_out, img_out_LaneAF


def Detection(args, img):
    '''加入 mmdetection 訓練後的結果， 目前只取 ego vehicle'''
    model = init_detector(args.detconfig, args.detcheckpoint, device='cuda:0')
    # test a single image
    result = inference_detector(model, img)
    egobboxs = []
    if len(result[0])> 0 :
        sc_row, _ = np.where(result[0]== np.max(result[0],axis=0)[4]) # the highest score
        x1, y1, x2, y2, score = result[0][sc_row][0][0], result[0][sc_row][0][1], result[0][sc_row][0][2], result[0][sc_row][0][3], result[0][sc_row][0][4]
        
        if score > args.detscore_thr:
            # print(x1, y1, x2, y2)
            box = {'label': model.CLASSES[0], 'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
            egobboxs.append(box)
            drawBoundingBox(img, egobboxs)
            return img, y1, box

    return img, None, None

if __name__ == "__main__":

    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    if args.backbone == 'dla34':
        model = get_pose_net(num_layers=34, heads=heads,head_conv=256, down_ratio=4)
    elif args.backbone == 'erfnet':
        model = ERFNet(heads=heads)
    elif args.backbone == 'enet':
        model = ENet(heads=heads)
    model.load_state_dict(torch.load(args.snapshot), strict=True)
    if args.cuda:
        model.cuda()
    # print(model)
    
    """ preprocessing video  """
    cap = cv2.VideoCapture(args.input_video)
    
    # 寫入檔案
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # 設定參數
    frame_index = 0
    lane_allframe = [] #儲存每一幀的lanes，(包含修改過的)
    tem = [(-1,-1)] #儲存有問題的frameID and 每一幀的lanes(原始)
    slope_diff = 0.15 #線段的斜率相減小於的值，視為同一條線
    lanecolor = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,255,255)]

    while (cap.isOpened()):
        success, frame = cap.read()
        laneframe = LanePerFrame(str(frame_index))

        if not success:
            print("Can't receive frame ",frame_index+1)
            break
        if frame_index == 0: #第一幀為標準幀
            parm = Parm(np.shape(frame)[1],np.shape(frame)[0])
            parm.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            parm.fps = cap.get(cv2.CAP_PROP_FPS)

            ### carema correction (barrel distortion) ###
            frame, cam, distCoeff = camera_correction(frame)

            ### frame processing ###
            img = cv2.resize(frame[int(parm.IMG_org_crop_w):,:, :], (int(parm.IMG_H), int(parm.IMG_W)), interpolation=cv2.INTER_LINEAR)  # 圖片長寬要可以被 32 整除

            ### detect hood ###
            global egoH
            img, egoH, ego_box = Detection(args, img)
            # cv2.imshow("img_out", img)
            # cv2.waitKey(0)
            print(ego_box)
            ### frame processing ###
            img = img.astype(np.float32)/255  # (H, W, 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ### Identify lane lines:LaneAF ###
            seg_out_LaneAF, img_out_LaneAF = LaneAF(img, model)
            ### build centerline and Lane Info (stored in laneframe)  ###
            laneframe = centerline(seg_out_LaneAF,laneframe,ego_box)
            lane_allframe.append(laneframe)   
            ### re-Lane ###
            #第一幀pass
            ### lane vis ###
            for i, laneid in enumerate(lane_allframe[frame_index].laneIDs):
                # print("laneid",laneid.name,laneid.equa[0])
                cols, rows = zip(*laneid.allpoints)
                for r, c in zip(rows, cols):
                    cv2.circle(img_out_LaneAF, (r, c) , 10, lanecolor[i], 1)
            # cv2.imshow("img_out", img_out_LaneAF)
            # cv2.waitKey(0)
            ### Determine the location of the lanes ###
            lane_loc = lane_loc_f(laneframe)
            ### Calculate the vanishing point ###
            Vpoint = vanishing_point(lane_loc)
            ### Perspective transform ###
            img_out, warped, pm= perspective_transform(parm, egoH, Vpoint, lane_loc, img_out_LaneAF)
            ### 建立剪裁參數 ###
            crop_loc(pm[0],pm[1],laneframe,parm)
            ### 剪裁圖片 ###
            cropimage = warped[0:int(parm.IMG_W) , parm.crop[0] : parm.crop[1]]
            img = cv2.hconcat([img_out, cropimage])  # 水平拼接
            videoWrite = cv2.VideoWriter('/home/hsuan/results/result.avi', fourcc, parm.fps, (int(parm.IMG_H+parm.crop[1]-parm.crop[0]), int(parm.IMG_W)))
            videoWrite.write(img)

        elif (frame_index+1) == (parm.frame_count): #最後一幀結束
            print("Stream end. Exiting ...")
            break

        else:
            ### carema correction (barrel distortion) ###
            frame = cv2.undistort(frame,cam,distCoeff)
            ### frame processing ###
            img = frame.astype(np.float32)/255  # (H, W, 3)
            img = cv2.resize(img[int(parm.IMG_org_crop_w):,:, :], (int(parm.IMG_H), int(parm.IMG_W)), interpolation=cv2.INTER_LINEAR)  # 圖片長寬要可以被 32 整除
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ### Identify lane lines:LaneAF ###
            seg_out_LaneAF, img_out_LaneAF = LaneAF(img, model)
            # cv2.imshow("img_out", img_out_LaneAF)
            # cv2.waitKey(0)
            ### build centerline and Lane Info (stored in laneframe) ###
            laneframe = centerline(seg_out_LaneAF,laneframe,ego_box)
            lane_allframe.append(laneframe)
            ### re-Lane ###
            lane_allframe, tem = re_lane(lane_allframe,frame_index,tem, slope_diff)
            ### lane vis ###
            for i, laneid in enumerate(lane_allframe[frame_index].laneIDs):
                # print("laneid",laneid.name,laneid.equa[0])
                cols, rows = zip(*laneid.allpoints)
                for r, c in zip(rows, cols):
                    cv2.circle(img_out_LaneAF, (r, c) , 10, lanecolor[i], 1)
            # cv2.imshow("img_out", img_out_LaneAF)
            # cv2.waitKey(0)
            ### 道路線位置判斷 ###
            lane_loc = lane_loc_f(lane_allframe[frame_index])
            ### 消失點 ###
            Vpoint = vanishing_point(lane_loc)
            ### 透視變換 ###
            img_out, warped, pm= perspective_transform(parm, egoH, Vpoint, lane_loc, img_out_LaneAF)
            ### 剪裁圖片 ###
            cropimage = warped[0:int(parm.IMG_W) , parm.crop[0] : parm.crop[1]]
            img = cv2.hconcat([img_out, cropimage])  # 水平拼接
            
            warped2 = cv2.resize(warped,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_LINEAR)
            # print(np.shape(img))
            # cv2.imshow("img_out", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            videoWrite.write(img)


        print('Done with image {} out of {}...'.format(frame_index+1, int(parm.frame_count)))
        frame_index+=1

    videoWrite.release()

