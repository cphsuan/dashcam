import os
import json
from datetime import datetime
from pickle import FALSE, TRUE
from statistics import mean
import argparse
import numpy as np
import cv2
from pandas import array
import math
from sklearn.metrics import accuracy_score, f1_score
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

parser = argparse.ArgumentParser('Options for inference with LaneAF models in PyTorch...')
parser.add_argument('--input_video', type=str , default='/media/hsuan/data/VIL100/videos/0_Road014_Trim004_frames.avi', help='path to input video')
parser.add_argument('--dataset-dir', type=str , default='/media/hsuan/data/TuSimple', help='path to dataset')
parser.add_argument('--snapshot', type=str, default='/home/hsuan/Thesis/LaneAF/laneaf-weights/tusimple-weights/dla34/net_0025.pth', help='path to pre-trained model snapshot')
parser.add_argument('--seed', type=int, default=1 , help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')

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


class Parm:
    def __init__(self,IMG_H, IMG_W):
        self.IMG_H = IMG_H
        self.IMG_org_crop_H = IMG_H
        self.IMG_W = IMG_W
        self.IMG_org_crop_w = IMG_W

    @property
    def IMG_H(self):
        return self.__IMG_H
    @IMG_H.setter
    def IMG_H(self, value):
        quotient = value / 32
        if quotient %2 == 0:
            self.__IMG_H = quotient*32
        else:
            self.__IMG_H = (quotient-1)*32
    @property
    def IMG_W(self):
        return self.__IMG_W
    @IMG_W.setter
    def IMG_W(self, value):
        quotient = math.floor(value / 32)
        if quotient %2 == 0:
            self.__IMG_W = quotient*32
        else:
            self.__IMG_W = (quotient-1)*32
    @property
    def IMG_org_crop_w(self):
        return self.__IMG_org_crop_w
    @IMG_org_crop_w.setter
    def IMG_org_crop_w(self,value):
        quotient = math.floor(value / 32)
        if quotient %2 == 0:
            self.__IMG_org_crop_w = 0
        else:
            self.__IMG_org_crop_w = value - (quotient-1)*32


def LaneAF(image, net):
    """ LaneAF: Robust Multi-Lane Detection with Affinity Fields
    https://github.com/sel118/LaneAF """
    net.eval()
    # img preprocessing
    img = image.astype(np.float32)/255  # (H, W, 3)
    #img = cv2.resize(img[16:, :, :], (1280, 704), interpolation=cv2.INTER_LINEAR)  # 圖片長寬要可以被 32 整除
    img = cv2.resize(img[int(parm.IMG_org_crop_w):,:, :], (int(parm.IMG_H), int(parm.IMG_W)), interpolation=cv2.INTER_LINEAR)  # 圖片長寬要可以被 32 整除
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_transforms = transforms.Compose([
        tf.GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
        tf.GroupNormalize(mean=([0.485, 0.456, 0.406], (0, )), std=([0.229, 0.224, 0.225], (1, ))),])
    
    # mask processing #暫時不處理

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
    img_out = create_viz(img, seg_out.astype(
        np.uint8), mask_out, vaf_out, haf_out)
    return seg_out, img_out


def preprocessing(seg_out_LaneAF, img_out_LaneAF):
    """ preprocessing image : create centerlane → """
    #####test 建立lane class
    global LaneID
    LaneID = centerline(seg_out_LaneAF)    
    ####斜率判斷
    lane_loc = lane_loc_f(LaneID)
    ## 消失點
    Vpoint = vanishing_point(lane_loc)
    ### 透視變換 ###
    img_out, warped, pm= perspective_transform(parm, LaneID, Vpoint, lane_loc, img_out_LaneAF)

    crop_loc(pm[0],pm[1],LaneID,parm)
    cropimage = warped[0:int(parm.IMG_W) , parm.crop[0] : parm.crop[1]]

    img = cv2.hconcat([img_out, cropimage])  # 水平拼接
    return img

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
    print(model)
    

    """ preprocessing video  """
    cap = cv2.VideoCapture(args.input_video)
    
    # 寫入檔案
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    i = 1
    while (cap.isOpened()):
        success, frame = cap.read()

        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if i == 1:
            parm = Parm(np.shape(frame)[1],np.shape(frame)[0])
            parm.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            parm.fps = cap.get(cv2.CAP_PROP_FPS)

            seg_out_LaneAF, img_out_LaneAF = LaneAF(frame, model)
            img_out = preprocessing(seg_out_LaneAF, img_out_LaneAF)

            ### 建立lane class ###
            global LaneID
            LaneID = centerline(seg_out_LaneAF)    
            ### 斜率判斷 ###
            lane_loc = lane_loc_f(LaneID)
            ### 消失點 ###
            Vpoint = vanishing_point(lane_loc)
            ### 透視變換 ###
            img_out, warped, pm= perspective_transform(parm, LaneID, Vpoint, lane_loc, img_out_LaneAF)
            ### 建立剪裁參數 ###
            crop_loc(pm[0],pm[1],LaneID,parm)
            ### 剪裁圖片 ###
            cropimage = warped[0:int(parm.IMG_W) , parm.crop[0] : parm.crop[1]]
            img = cv2.hconcat([img_out, cropimage])  # 水平拼接
            
            videoWrite = cv2.VideoWriter('/home/hsuan/result.avi', fourcc, parm.fps, (int(parm.IMG_H+parm.crop[1]-parm.crop[0]), int(parm.IMG_W)))
            videoWrite.write(img)


        seg_out_LaneAF, img_out_LaneAF = LaneAF(frame, model)
        
        ### 建立lane class ###
        LaneID = centerline(seg_out_LaneAF)    
        ### 斜率判斷 ###
        lane_loc = lane_loc_f(LaneID)
        ### 消失點 ###
        Vpoint = vanishing_point(lane_loc)
        ### 透視變換 ###
        img_out, warped, pm= perspective_transform(parm, LaneID, Vpoint, lane_loc, img_out_LaneAF)
        ### 剪裁圖片 ###
        cropimage = warped[0:int(parm.IMG_W) , parm.crop[0] : parm.crop[1]]
        img = cv2.hconcat([img_out, cropimage])  # 水平拼接
        
        # cv2.imshow("img_out", img)
        # cv2.waitKey(0)
        videoWrite.write(img)

        print('Done with image {} out of {}...'.format(i, int(parm.frame_count)))
        i+=1

    videoWrite.release()

