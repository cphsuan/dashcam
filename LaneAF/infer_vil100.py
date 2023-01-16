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
import matplotlib.pyplot as plt
from tools.logger import *
import pandas as pd
#for mmdetection
sys.path.append(r'/home/hsuan/Thesis/mmdetection/')
from mmdet.apis import (inference_detector, init_detector)
import pysnooper
import time
#for mmclassification
sys.path.append(r'/home/hsuan/Thesis/mmclassification/')
from mmcls.apis import init_model, inference_model, show_result_pyplot



parser = argparse.ArgumentParser('Options for inference with LaneAF models in PyTorch...')
# 2_Road026_Trim003_frames

# demo video
# 0_Road001_Trim003_frames
# 1_Road014_Trim001_frames
# 2_Road017_Trim001_frames
# 4_Road027_Trim013_frames
# 5_Road001_Trim001_frames
# 6_Road022_Trim001_frames

parser.add_argument('--input_video', type=str , default='2_Road017_Trim001_frames.avi', help='path to input video')
parser.add_argument('--dataset-dir', type=str , default='/media/hsuan/data/CULane', help='path to dataset')
#'/media/hsuan/data/WarpDataset/VIL100/JPEGImages/'
parser.add_argument('--storepic-path', type=str , default='/media/hsuan/data/WarpDataset/VIL100/JPEGImages/', help='path to storepic dataset')
# culane-weights/dla34/net_0033.pth
parser.add_argument('--snapshot', type=str, default='/home/hsuan/Thesis/LaneAF/laneaf-weights/culane-weights/dla34/net_0033.pth', help='path to pre-trained model snapshot')
parser.add_argument('--seed', type=int, default=1 , help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
# for mmdetection
parser.add_argument('--detconfig',default='mmdetection/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py' ,help='Config file')
parser.add_argument('--detcheckpoint',default='mmdetection/work_dirs/latest.pth' ,help='Checkpoint file')
parser.add_argument('--detout-file', default=None, help='Path to output file')
parser.add_argument('--detpalette', default='coco', choices=['coco', 'voc', 'citys', 'random'], help='Color palette used for visualization')
parser.add_argument('--detscore-thr', type=float, default=0.5, help='bbox score threshold')
# for mmclassification
parser.add_argument('--clsconfig',default='mmclassification/wrk_dir/resnet101_warp.py' ,help='Config file')
parser.add_argument('--clscheckpoint',default='mmclassification/wrk_dir/latest.pth' ,help='Checkpoint file')
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
#store pic to do dataset
storepic_path = args.storepic_path+(args.input_video).replace('.avi', '')
if not os.path.isdir(storepic_path):
    os.mkdir(storepic_path)
# set random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 1}

#LaneAF model for lane detection
def LaneAF(img, net):
    """
    LaneAF: Robust Multi-Lane Detection with Affinity Fields
    https://github.com/sel118/LaneAF 
    """
    net.eval()
    img_transforms = transforms.Compose([
        tf.GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
        tf.GroupNormalize(mean=([0.485, 0.456, 0.406], (0, )), std=([0.229, 0.224, 0.225], (1, ))),])
    # TODO mask processing
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
    seg_out = decodeAFs(mask_out[:, :, 0], vaf_out,haf_out, fg_thresh=128, err_thresh=5,viz=False)

    ###
    if torch.any(torch.isnan(input_seg)):
        # if labels are not available, skip this step
        pass
    else:
        # if test set labels are available
        # re-assign lane IDs to match with ground truth
        seg_out = match_multi_class(seg_out.astype(np.int64), input_seg[0, 0, :, :].detach().cpu().numpy().astype(np.int64))

    ### LANEAF RESULT VIS(orginal) ### 
    img_out_LaneAF = np.ascontiguousarray(create_viz(img, seg_out.astype(np.uint8), mask_out, vaf_out, haf_out), dtype=np.uint8)

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    return seg_out, img

def HoodDetection(args, img):
    '''加入 mmdetection 訓練後的結果，取 ego vehicle'''
    objmodel = init_detector(args.detconfig, args.detcheckpoint, device='cuda:0')
    # test a single image
    objresult = inference_detector(objmodel, img)
    y1, egobbox = None, None
    # egohood
    if len(objresult[0])> 0 :
        sc_row, _ = np.where(objresult[0]== np.max(objresult[0],axis=0)[4]) # the highest score
        x1, y1, x2, y2, score = objresult[0][sc_row][0][0], objresult[0][sc_row][0][1], objresult[0][sc_row][0][2], objresult[0][sc_row][0][3], objresult[0][sc_row][0][4]
        
        if score > args.detscore_thr:
            # print(x1, y1, x2, y2)
            egobbox = {'label': objmodel.CLASSES[0], 'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),'score':score}
            # drawBoundingBox(img, egobbox)
            # print(box)
        else:
            y1 = None

    return img, y1, egobbox, objmodel

def ObjDetection(args,objmodel, img):
    objresult = inference_detector(objmodel, img)
    # other objects
    class_res = []
    for i in range(1,len(objresult)):
        if len(objresult[i])>0:
            for j in range(0,len(objresult[i])):
                if objresult[i][j][4]  > args.detscore_thr:
                    class_bbox = {'label': objmodel.CLASSES[i], 'x1': int(objresult[i][j][0]), 'y1': int(objresult[i][j][1]), 'x2': int(objresult[i][j][2]), 'y2': int(objresult[i][j][3])}
                    class_res.append(class_bbox)

    # drawBoundingBox(img, class_res)
    # cv2.imshow("img_out", img)
    # cv2.waitKey(0)
    return img, class_res

def LaneClassification(args,clsmodel, img):
    
    clsresult = inference_model(clsmodel, img)
    print("classification_result",clsresult)
    # show_result_pyplot(clsmodel, img, clsresult)
    return clsresult

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
    cap = cv2.VideoCapture("/media/hsuan/data/VIL100/videos/"+args.input_video)
    
    # write file
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # setting parameter
    frame_index,previous_L,previous_R = 0,0,0
    lane_allframe,previous_Vpoint = [],[] #store all lanes pre frame
    diff = 2 #threshold for determine slope if the same lane
    lanecolor = {1:(0,255,0),2:(0,0,255),3:(255,0,0),\
        4:(255,255,0),5:(255,0,255),6:(0,255,255),\
        7:(125,0,125),8:(0,125,125),9:(125,125,0),\
        10:(125,0,255),11:(0,255,125),12:(255,125,0)}
    change_lane = False
    LaneID_Info = []
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
            # i_index,r2_list,num = [],[],[]
            # for k in np.arange(0, -0.00003, -0.000001):
            #     frame0, _, _ = camera_correction(frame,round(k,6))
            #     # cv2.imwrite("/home/hsuan/canny/"+str(round(abs(k)*100000,1))+".jpg", frame0)
            #     ### frame processing ###
            #     img = cv2.resize(frame0[int(parm.IMG_org_crop_w):,:, :], (int(parm.IMG_H), int(parm.IMG_W)), interpolation=cv2.INTER_LINEAR)  # 圖片長寬要可以被 32 整除

            #     ### detect hood ###
            #     global egoH
            #     img, egoH, ego_box, objmodel = HoodDetection(args, img)
            #     # If there is a hood below the picture
            #     transH=egoH if egoH else parm.IMG_W
            #     ### frame processing ###
            #     img = img.astype(np.float32)/255  # (H, W, 3)
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #     ### Identify lane lines:LaneAF ###
            #     seg_out_LaneAF, img_out_LaneAF = LaneAF(img, model)
            #     ### build centerline and Lane Info (stored in laneframe)  ###
            #     laneframe = centerline(seg_out_LaneAF,laneframe,ego_box)
            #     ### Calculate R^2 to evaluate undistort
            #     r2_ = []
            #     for _, laneid in enumerate(laneframe.laneIDs):
            #         y, x = zip(*laneid.allpoints)
            #         parameter = np.polyfit(x, y, 1) #建立線

            #         f = [np.polyval(parameter, x_value) for x_value in x]
            #         r2_.append(r2_score(y, f))

            #     # print("avg=",sum(r2_)/len(r2_))
            #     i_index.append(k)
            #     r2_list.append(sum(r2_)/len(r2_))
            #     num.append(len(r2_))
            #     laneframe = laneframe.reset((frame_index))
                
            # # df = pd.DataFrame(list(zip(i_index, r2_list,num)), columns =["i_index","r2_list","num"])
            # # with open('/home/hsuan/results2/output.csv', 'w', encoding = 'utf-8-sig') as f:  
            # #     df.to_csv(f)

            # # plt.style.use("ggplot")
            # # fig, ax1 = plt.subplots()
            # # ax2 = ax1.twinx()

            # # ax1.plot(df["i_index"], df["r2_list"],c = 'tab:blue')
            # # ax1.set_ylabel('Average of R2 Score', color='tab:blue')
            # # ax1.set_ylim([min(df["r2_list"])-0.05,1])
            # # ax1.tick_params(axis='y', labelcolor='tab:blue')
            # ##########
            # if len(set(num)) != 1:
            #     max_r2 = r2_list[0]
            #     for i, x in enumerate(num):
            #         if x > num[0]:
            #             break
            #         max_r2 = max(r2_list[i],max_r2)
            #     coef_k = i_index[r2_list.index(max_r2)]
            # else:
            #     max_r2 = max(r2_list)
            #     coef_k = i_index[r2_list.index(max_r2)]
            # #########
            # # ax1.plot(coef_k,max_r2, 'bo')
            # # ax1.annotate("({},{})".format(coef_k,round(max_r2,0)), xy=(coef_k,max_r2), xytext=(15, 0), color='blue', textcoords='offset points')

            # # ax2.plot(df["i_index"], df["num"],c = "g")
            # # ax2.set_ylabel('Num of Lanes', color='g')
            # # ax2.set_ylim([min(df['num'])-1 if min(df['num'])-1 >= 0  else 0,max(df['num'])+1])
            # # ax2.tick_params(axis='y', labelcolor='g')

            # # plt.title("Calibration Coefficient k Estimation", fontsize = 15, fontweight = "bold", y = 1) 
            # # ax1.set_xlabel("Calibration coefficient k", fontweight = "bold")   
            # # ax1.set_xlim(max(df["i_index"]),min(df["i_index"]))

            # # fig.tight_layout()
            # # plt.show()

            ### carema correction (barrel distortion) ###
            coef_k = -0.00002 #-0.000011
            frame, cam, distCoeff = camera_correction(frame,round(coef_k,6))
            ### frame processing ###
            img = cv2.resize(frame[int(parm.IMG_org_crop_w):,:, :], (int(parm.IMG_H), int(parm.IMG_W)), interpolation=cv2.INTER_LINEAR)  # 圖片長寬要可以被 32 整除
            ### detect hood ###
            img, egoH, ego_box, objmodel = HoodDetection(args, img)
            # cv2.imwrite("/home/hsuan/results_img/hoodres/hood_result3.jpg", img)
            # cv2.imshow("img_out", img)
            # cv2.waitKey(0)
            ### If there is a hood below the picture###
            transH=egoH if egoH else parm.IMG_W
            ### frame processing ###
            img = cv2.cvtColor(img.astype(np.float32)/255, cv2.COLOR_BGR2RGB)
            ### Identify lane lines:LaneAF ###
            seg_out_LaneAF, img_out_LaneAF = LaneAF(img, model)

            # cv2.imwrite("/home/hsuan/results_img/lane_result{}.jpg".format(frame_index), img_out_LaneAF)
            ### build centerline and Lane Info (stored in laneframe)  ###
            laneframe = centerline(seg_out_LaneAF,laneframe,ego_box)

            ###calculate angle###
            laneframe = angle_f(laneframe,transH)

            #########################讀vil100 dataset json (for dataset)
            # vil_jsonlist = [_ for _ in os.listdir(os.path.join('/media/hsuan/data/VIL100/Json/{}'.format((args.input_video).replace('.avi','')))) if _.endswith(".json")]
            # vil_jsonlist = sorted(vil_jsonlist)
            # with open(os.path.join('/media/hsuan/data/VIL100/Json/{}/{}'.format((args.input_video).replace('.avi',''),vil_jsonlist[frame_index]))) as f:
            #     vildata = json.load(f)
            # vil_info = vildata['annotations']["lane"]
            # for i, laneid in enumerate(laneframe.laneIDs):
            #     for vil_i, vil_id in enumerate(vil_info):
            #         #build the lines
            #         x, y = zip(*vil_id['points'])
            #         y = [i_y-(np.shape(frame)[0]- np.shape(img)[0]) for i_y in y]
            #         vil_equa = np.polyfit(x, y, 1)

            #         if vil_equa[0] ==1: #垂直線
            #             angle = 90
            #         elif vil_equa[1] <0:
            #             angle = abs((math.atan2( vil_equa[1]-0, 0-vil_equa[1]/vil_equa[0] )/math.pi*180))
            #         else:
            #             angle =180-(math.atan2( vil_equa[1]-0, 0-vil_equa[1]/vil_equa[0] )/math.pi*180)
                    
            #         if vil_i == 0 :
            #             min_angle = abs(laneid.angle-angle)
            #         if abs(laneid.angle-angle) <= min_angle:
            #             min = abs(laneid.angle-angle)
            #             laneid.lanetype = vil_id['attribute']
            #             laneid.occlusion = vil_id['occlusion']
            
            ### re-Lane ###
            for i, laneid in enumerate(laneframe.laneIDs):
                LaneID_Info.append(LaneID_info(laneid.name,5, laneid.angle))
            maxID = len(laneframe.laneIDs)
            laneframe.sort()
            ### lane vis ###
            # for i, laneid in enumerate(laneframe.laneIDs):
            #     # print(laneid.equa,laneid.angle)
            #     cols, rows = zip(*laneid.allpoints)
            #     for r, c in zip(rows, cols):
            #         cv2.circle(img_out_LaneAF, (r, c) , 3, lanecolor[laneid.name], 1)
            #     cv2.imshow("img_out", img_out_LaneAF)
            #     cv2.waitKey(0)
            # cv2.imwrite("/home/hsuan/results_img/lane_result{}.jpg".format(frame_index), img_out_LaneAF)
            ### Determine the location of the lanes and Calculate the vanishing point ###
            try:
                lane_loc = lane_loc_f(laneframe)
                Vpoint = vanishing_point(lane_loc)
            except:
                logging_func(1,args.input_video,frame_index)
                break
            
            laneframe.Vpoint = Vpoint
            lane_allframe.append(laneframe)

            ### Perspective transform ###
            img_out, warped, pm= perspective_transform(parm, transH, Vpoint, lane_loc, img_out_LaneAF)
            # cv2.imshow("warped", warped)
            # cv2.waitKey(0)
            ### setting crop parameter ###
            crop_loc(pm[0],pm[1],laneframe,parm) 
            ### crop pic ###
            cropimage = warped[0:int(parm.IMG_W) , parm.crop[0] : parm.crop[1]]

            ### crop lane ###
            lane_pic = crop_lane(lane_allframe[frame_index],warped,parm,pm)

            #########################讀vil100 dataset json (for dataset)
            # for idx, laneid in enumerate(lane_allframe[frame_index].laneIDs):
            #     img = laneid.img
            #     lanetype = laneid.lanetype
            #     if img != "undefined":
            #         cv2.imwrite(os.path.join('/media/hsuan/data/LaneDataset/{}/{}'.format(str(lanetype),str((args.input_video).replace('.avi',''))+str(frame_index)+'_'+str(idx))+'.jpg'),img)

            ### image Enhancement ###
            clsmodel = init_model(args.clsconfig, args.clscheckpoint, device='cuda:0')
            for idx,laneid in enumerate(lane_allframe[frame_index].laneIDs):
                img = laneid.img
                #Laplace
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
                img_Laplace = cv2.filter2D(img, -1, kernel=kernel)       
                # CLAHE
                imgYUV = cv2.cvtColor(img_Laplace, cv2.COLOR_BGR2YCrCb)
                channelsYUV = cv2.split(imgYUV)
                t = channelsYUV[0]
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
                p= clahe.apply(t)
                channels = cv2.merge([p,channelsYUV[1],channelsYUV[2]])
                img_CLAHE = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)

                clsresult = LaneClassification(args,clsmodel, img_CLAHE)
                print(clsresult)
                laneid.lanetype = clsresult
                # cv2.imshow("Orginal", img)
                cv2.imshow("CLAHE_result", img_CLAHE)
                cv2.waitKey(0)
            ### store pic to do dataset
            # cv2.imwrite(os.path.join(storepic_path,'{0:05d}'.format(frame_index)+'.jpg'),cropimage)

            img = cv2.hconcat([img_out, cropimage])  # 水平拼接
            #int(parm.IMG_H+parm.crop[1]-parm.crop[0])
            videoWrite = cv2.VideoWriter('/home/hsuan/results2/'+args.input_video, fourcc, parm.fps, (int(parm.IMG_H+parm.crop[1]-parm.crop[0]), int(parm.IMG_W)))
            videoWrite.write(img)

        elif (frame_index+1) == (parm.frame_count): #最後一幀結束
            print("Stream end. Exiting ...")
            break

        else:
            ### carema correction (barrel distortion) ###
            frame = cv2.undistort(frame,cam,distCoeff)
            ### frame processing ###
            img = cv2.resize(frame[int(parm.IMG_org_crop_w):,:, :], (int(parm.IMG_H), int(parm.IMG_W)), interpolation=cv2.INTER_LINEAR)  # 圖片長寬要可以被 32 整除
            ### detect hood ### pass
            ### detect objects ###
            # img, class_res = ObjDetection(args,objmodel, img)
            ### Identify lane lines:LaneAF ###
            img = cv2.cvtColor(img.astype(np.float32)/255, cv2.COLOR_BGR2RGB)
            seg_out_LaneAF, img_out_LaneAF = LaneAF(img, model)
            ### build centerline and Lane Info (stored in laneframe) ###
            laneframe = centerline(seg_out_LaneAF,laneframe,ego_box)
            ###calculate angle###
            laneframe = angle_f(laneframe,transH)

            #########################讀vil100 dataset json (for dataset)
            # vil_jsonlist = [_ for _ in os.listdir(os.path.join('/media/hsuan/data/VIL100/Json/{}'.format((args.input_video).replace('.avi','')))) if _.endswith(".json")]
            # vil_jsonlist = sorted(vil_jsonlist)
            # with open(os.path.join('/media/hsuan/data/VIL100/Json/{}/{}'.format((args.input_video).replace('.avi',''),vil_jsonlist[frame_index]))) as f:
            #     vildata = json.load(f)
            # vil_info = vildata['annotations']["lane"]
            # for i, laneid in enumerate(laneframe.laneIDs):
            #     for vil_i, vil_id in enumerate(vil_info):
            #         #build the lines
            #         x, y = zip(*vil_id['points'])
            #         y = [i_y-(np.shape(frame)[0]- np.shape(img)[0]) for i_y in y]
            #         vil_equa = np.polyfit(x, y, 1)

            #         if vil_equa[0] ==1: #垂直線
            #             angle = 90
            #         elif vil_equa[1] <0:
            #             angle = abs((math.atan2( vil_equa[1]-0, 0-vil_equa[1]/vil_equa[0] )/math.pi*180))
            #         else:
            #             angle =180-(math.atan2( vil_equa[1]-0, 0-vil_equa[1]/vil_equa[0] )/math.pi*180)
                    
            #         if vil_i == 0 :
            #             min_angle = abs(laneid.angle-angle)
            #         if abs(laneid.angle-angle) <= min_angle:
            #             min = abs(laneid.angle-angle)
            #             laneid.lanetype = vil_id['attribute']
            #             laneid.occlusion = vil_id['occlusion']

            ### re-Lane ###
            laneframe.sort()
            laneframe,LaneID_Info, maxID= re_lane_angle(laneframe,LaneID_Info,maxID)
            ### vis
            # for i, laneid in enumerate(laneframe.laneIDs):
            #     print(laneid.equa,laneid.angle)
            #     print(len(laneid.allpoints))
            #     cols, rows = zip(*laneid.allpoints)
            #     for r, c in zip(rows, cols):
            #         cv2.circle(img_out_LaneAF, (r, c) , 3, lanecolor[laneid.name], 1)
            #     cv2.imshow("img_out", img_out_LaneAF)
            #     cv2.waitKey(0)
            #     cv2.imwrite("/home/hsuan/results_img/lane_result{}.jpg".format(frame_index), img_out_LaneAF)
            ###  Determine the location of the lanes and Calculate the vanishing point###
            try:
                lane_loc = lane_loc_f(laneframe)
                Vpoint = vanishing_point(lane_loc)
            except:
                logging_func(1,args.input_video,frame_index)
                break
            laneframe.Vpoint = Vpoint
            lane_allframe.append(laneframe)

            ### Perspective transform ###
            img_out, warped, pm= perspective_transform(parm, transH, Vpoint, lane_loc, img_out_LaneAF)
            ### crop pic ###
            cropimage = warped[0:int(parm.IMG_W) , parm.crop[0] : parm.crop[1]]
            ### crop lane ###
            lane_pic = crop_lane(lane_allframe[frame_index],warped,parm,pm)
            
            #########################讀vil100 dataset json (for dataset)
            # for idx, laneid in enumerate(lane_allframe[frame_index].laneIDs):
            #     img = laneid.img
            #     lanetype = laneid.lanetype
            #     if img != "undefined":
            #         cv2.imwrite(os.path.join('/media/hsuan/data/LaneDataset/{}/{}'.format(str(lanetype),str((args.input_video).replace('.avi',''))+str(frame_index)+'_'+str(idx))+'.jpg'),img)

            ### image Enhancement ###
            for idx,laneid in enumerate(lane_allframe[frame_index].laneIDs):
                img = laneid.img
                #Laplace
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
                img_Laplace = cv2.filter2D(img, -1, kernel=kernel)       
                # CLAHE
                imgYUV = cv2.cvtColor(img_Laplace, cv2.COLOR_BGR2YCrCb)
                channelsYUV = cv2.split(imgYUV)
                t = channelsYUV[0]
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
                p= clahe.apply(t)
                channels = cv2.merge([p,channelsYUV[1],channelsYUV[2]])
                img_CLAHE = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
                clsresult = LaneClassification(args,clsmodel, img_CLAHE)
                laneid.lanetype = clsresult
                cv2.imshow("CLAHE_result", img_CLAHE)
                cv2.waitKey(0)

            img = cv2.hconcat([img_out, cropimage])  # 水平拼接
            # print(np.shape(img))
            # input()
            img = cv2.rectangle(img, (40, 15), (550, 50), (0, 0, 0), -1)
            img = cv2.putText(img, ("frame: " + str(frame_index+1))+"  changelane: "+str(change_lane), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,cv2.LINE_AA)
            # for i, laneid in enumerate(lane_allframe[frame_index].laneIDs):
            #     img = cv2.putText(img, ("laneID:" + str(laneid.name) +" Slope:"+ str(round(laneid.equa[0],3))), (int(laneid.hor_x), 500+20*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,cv2.LINE_AA)
            # # print(np.shape(img))
            # cv2.imshow("img_out", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            videoWrite.write(img)


        print('Done with image {} out of {}...'.format(frame_index, int(parm.frame_count)))
        frame_index+=1

    videoWrite.release()

