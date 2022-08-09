import json
import os

import argparse
import cv2
import pandas as pd
import numpy as np
from copy import deepcopy

import sys
sys.path.append(r'/home/hsuan/Thesis/')
from LaneAF.tools.perspective_correction import *
from LaneAF.tools.CustomerClass import *


parser = argparse.ArgumentParser('MakeWarpDataset...')
parser.add_argument('--dataset-dir', type=str , default='/media/hsuan/data/VIL100/', help='path to dataset')
parser.add_argument('--output-IMGdir', type=str , default='/media/hsuan/data/WarpDataset/VIL100/JPEGImages/', help='path to output IMG dataset')
args = parser.parse_args()


if __name__ == "__main__":
    # ParentPath
    parentPath = args.dataset_dir
    # allFileList = sorted(os.listdir(os.path.join(parentPath, "JPEGImages")))
    file = "0_Road014_Trim004_frames"
    outputIMGpath = os.path.join(args.output_IMGdir, file)
    if not os.path.isdir(outputIMGpath):
        os.makedirs(outputIMGpath)

    #參數
    lane_allframe = [] #儲存每一幀的lanes，(包含修改過的)
    tem = [(-1,-1)] #儲存有問題的frameID and 每一幀的lanes(原始)
    slope_diff = 0.08 #線段的斜率相減小於的值，視為同一條線

    # for fileIndex, file in enumerate(allFileList):
    # subpath
    imgPerFile = sorted(os.listdir(os.path.join(parentPath, "JPEGImages", file)))
    for frameIndex, frame in enumerate(imgPerFile):

        imgPath = os.path.join(parentPath, "JPEGImages", file, frame)
        jsonPath = os.path.join(parentPath, "Json", file, (frame+".json"))

        img0 = cv2.imread(imgPath)
        img = deepcopy(img0)

        with open(jsonPath) as f:
            jsonData = json.load(f)

        frame_info = jsonData["info"]
        lane_info = jsonData["annotations"]["lane"]

        laneframe = LanePerFrame(str(frameIndex))

        print("processing...frame:{}...".format(frameIndex))
        if frameIndex ==0:
            parm = ParmGT(np.shape(img)[1],np.shape(img)[0])
            ### 建立lane class ###
            for i in range(len(lane_info)):

                points_list = lane_info[i]["points"]
                # for point in points_list:
                #     cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), thickness)

                x_scale, y_scale = zip(*points_list)
                equa = np.polyfit(x_scale, y_scale, 1) #建立線

                eachLane = Lane(("LaneID_"+ str(lane_info[i]["lane_id"])), equa, points_list)
                eachLane.lanetype = lane_info[i]["attribute"]
                laneframe.add_lane(eachLane)
            
            lane_allframe.append(laneframe)
            ### re-Lane ###
            #第一幀pass
            ### 道路線位置判斷 ###
            lane_loc = lane_loc_f(laneframe)
            ### 消失點 ###
            Vpoint = vanishing_point(lane_loc)
            ### 透視變換 ###
            img_out, warped, pm= perspective_transform(parm, Vpoint, lane_loc, img)
            ### 建立剪裁參數 ###
            crop_loc(pm[0],pm[1],laneframe, parm)
            ### 剪裁圖片 ###
            cropimage = warped[0:int(parm.IMG_W) , parm.crop[0] : parm.crop[1]]
            img_out = cv2.hconcat([img_out, cropimage])  # 水平拼接

        elif (frameIndex+1) == (len(imgPerFile)): #最後一幀結束
            print("Stream end. Exiting ...")
            break

        else:
            ### 建立lane class ###
            for i in range(len(lane_info)):

                points_list = lane_info[i]["points"]
                # for point in points_list:
                #     cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), thickness)

                x_scale, y_scale = zip(*points_list)
                equa = np.polyfit(x_scale, y_scale, 1) #建立線

                eachLane = Lane(("LaneID_"+ str(lane_info[i]["lane_id"])), equa, points_list)
                eachLane.lanetype = lane_info[i]["attribute"]
                laneframe.add_lane(eachLane)
            
            lane_allframe.append(laneframe)
            ### re-Lane ###
            lane_allframe, tem = re_lane(lane_allframe,frameIndex,tem, slope_diff)
            print("frame:{} result...".format(frameIndex))
            for i, laneid in enumerate(lane_allframe[frameIndex].laneIDs):
                print("laneid",laneid.name,laneid.equa[0])
            #第一幀pass
            ### 道路線位置判斷 ###
            lane_loc = lane_loc_f(lane_allframe[frameIndex])
            ### 消失點 ###
            Vpoint = vanishing_point(lane_loc)
            ### 透視變換 ###
            img_out, warped, pm= perspective_transform(parm, Vpoint, lane_loc, img)
            ### 剪裁圖片 ###
            cropimage = warped[0:int(parm.IMG_W) , parm.crop[0] : parm.crop[1]]
            img_out = cv2.hconcat([img_out, cropimage])  # 水平拼接
        
        cv2.imwrite(os.path.join(outputIMGpath,frame),cropimage)
        # cv2.imshow("img_out", img_out)
        # cv2.imshow("cropimage", cropimage)
        # cv2.imshow("img", img_out)
        # cv2.waitKey(0)