from enum import unique
from cv2 import sort
import numpy as np
import cv2
from copy import deepcopy
from typing import Dict, List
import math
from sklearn.metrics import r2_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import pysnooper
import statistics
#@pysnooper.snoop()
class Lane:
    '''
    It's used to build each lane of per frame.
    name: LaneID_1, LaneID_2, LaneID_3...
    equa: equa[0] = slope, equa[1] = intercept build the line
    allpoints: allpoints of per frame(magnification)
    lanetype: #TODO lane classification
    '''
    def __init__(self,lane_name,equa, allpoints):
        self.name = lane_name
        self.equa = equa #中心線
        self.allpoints = allpoints #所有點(放大)
        self.lanetype = "undefined"
        self.hor_x = "undefined"
        self.angle = "undefined"
    
    def __str__(self):
        return f"Name is {self.name}, Equa is {self.equa}, Hor_x is {self.hor_x}, Angle is {self.angle}"

    @property
    def min_axis_x(self):
        return self.allpoints.min(axis=0)

    @property
    def max_axis_x(self):
        return self.allpoints.max(axis=0)

class LaneID_info:
    def __init__(self,lane_name,freq, angle):
        self.name = lane_name
        self.freq = freq #出現次數 #5 遞減
        self.angle = angle 
    
    def __str__(self):
        return f"Name is {self.name}, LaneID_info freq is {self.freq},LaneID_infoAngle is {self.angle}"

def centerline(seg_out,laneframe, ego_box):
    """
    Use the polyfit function to build centerlines.
    :return: laneframe: all lanes Info in this frame.
    """
    ID = np.delete(np.unique(seg_out), [0]) #number of lanes
    checkslope, checkintercept, id= [],[],1

    for i in ID:
        cal_seg = seg_out.copy()
        cal_seg[cal_seg != i] = 0
        AllPoints = []

        for row in range(np.shape(cal_seg)[0]): # search each row
            for col, v in enumerate(cal_seg[row]):
                if (v == i) :
                    # if the point is in the ego vehicle, it represents a misjudgment. pass it!
                    if ego_box != None and col*8 >= ego_box['x1'] and col*8 <= ego_box['x2'] and row*8 >= ego_box['y1'] and row*8 <= ego_box['y2'] :
                        continue
                    AllPoints.append([row*8, col*8])
        if len(AllPoints) < 20: # the lane is too small
            print("the lane is too small")
            continue

        #build the lines
        y, x = zip(*AllPoints)
        equa = np.polyfit(x, y, 1)

        if len(set(x)) <=6 and len(set(y)) >=15:
            equa = [1,-statistics.median(list(set(x)))]

        if abs(equa[0]) < 0.001:
            print("NEW!!!?????")
            # laneframe.add_lane(Lane(id,equa, np.array(AllPoints)))
            # id +=1
            input()
            continue

        #Check for similar lines
        if len(checkslope) != 0 :
            slope,error = [], 0.26

            if len(AllPoints) <= 80:
                error += 0.4
                
            for idx, s in enumerate(checkslope):
                if s/equa[0] <= 1+error and s/equa[0] >= 1-error and checkintercept[idx]/equa[1] <= 1+error and checkintercept[idx]/equa[1] >= 1-error: #beta0
                        slope.append(s)

            if len(slope) != 0 :
                min_x = min(slope)
                testlane = laneframe.laneIDs[checkslope.index(min_x)]

                comb_AllPoints = np.concatenate((testlane.allpoints, AllPoints))

                y_s, x_s = zip(*comb_AllPoints)
                comb_equa = np.polyfit(x_s, y_s, 1)

                if len(set(x_s)) <=5 and len(set(y_s)) >=15:
                    comb_equa = [1,-statistics.median(list(set(x_s)))]

                laneframe.laneIDs[checkslope.index(min_x)].equa = comb_equa
                laneframe.laneIDs[checkslope.index(min_x)].allpoints = comb_AllPoints
                continue

        checkslope.append(equa[0])
        checkintercept.append(equa[1])
        #add lane to laneframe
        laneframe.add_lane(Lane(id,equa, np.array(AllPoints)))
        id +=1

    return laneframe

def vanishing_point(lane_loc: Dict)-> List:
    """
    Calculate the vanishing point through the intersection of left_near_center and right_near_center
    """
    P_diff = np.polysub(lane_loc["left_near_center"].equa, lane_loc["right_near_center"].equa)
    Vx = np.roots(P_diff)
    Vy = np.polyval(lane_loc["left_near_center"].equa, Vx)
    Vpoint = np.append(Vx, Vy)
    return Vpoint

def angle_f(laneframe,transH):

    for id in laneframe.laneIDs:
        if id.equa[0] ==1: #垂直線
            angle = 90
        elif id.equa[1] <0:
            angle = abs((math.atan2( id.equa[1]-0, 0-id.equa[1]/id.equa[0] )/math.pi*180))
        else:
            angle =180-(math.atan2( id.equa[1]-0, 0-id.equa[1]/id.equa[0] )/math.pi*180)

        hor_x = int((transH - id.equa[1]) / id.equa[0]) 
        id.hor_x = hor_x
        id.angle = angle
    
    return laneframe

def lane_loc_f(laneframe):
    """
    Determine the location of the lanes.
    only judge by the slope.
    #TODO judge in another way: curve lane:https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8542714
    """
    left_loc, right_loc,center_loc =[], [], []

    for id in laneframe.laneIDs:

        angle = id.angle

        if angle > 70 and angle <110:
            center_loc.append(id)

        elif angle <70:
            left_loc.append(id)

        else :
            right_loc.append(id)

    left_loc.sort(key=lambda x: x.angle)
    right_loc.sort(key=lambda x: x.angle)

    lane_loc = {"leftmost":left_loc[0],"left_near_center":left_loc[-1],"right_near_center":right_loc[0],"rightmost":right_loc[-1],"center_loc":center_loc}
    
    return lane_loc

def corresponding_coordinates(poss,M):
    """透視變換 每個座標對應"""
    pos2 = []
    for pos in poss:
        u = pos[0]
        v = pos[1]
        x = (M[0][0]*u+M[0][1]*v+M[0][2])/(M[2][0]*u+M[2][1]*v+M[2][2])
        y = (M[1][0]*u+M[1][1]*v+M[1][2])/(M[2][0]*u+M[2][1]*v+M[2][2])

        pos2.append((int(x/2), int(y/2)))
    return pos2

def perspective_transform(parm, transH, Vpoint: List, lane_loc: Dict, img_out):
    """
    Convert to a bird's-eye view with perspective transformation
    """
    # 計算投影Y軸
    ProjY = int((transH-Vpoint[1])*0.15+Vpoint[1]) #0.25
    # 取 left_near_center right_near_center Vertical line
    lane1x_u = int((ProjY - lane_loc["left_near_center"].equa[1]) / lane_loc["left_near_center"].equa[0]) #右上
    lane2x_u = int((ProjY - lane_loc["right_near_center"].equa[1]) / lane_loc["right_near_center"].equa[0]) #左上
    lane1x_d = int((transH - lane_loc["left_near_center"].equa[1]) / lane_loc["left_near_center"].equa[0]) # 左下
    lane2x_d = int((transH - lane_loc["right_near_center"].equa[1]) / lane_loc["right_near_center"].equa[0]) #右下
    # 原點
    srcPts = [(lane2x_u, int(ProjY)),(lane1x_u, int(ProjY)),(lane2x_d, int(transH)),(lane1x_d, int(transH))] #(左上 右上 左下 右下 )
    
    img_out2 = img_out.copy()
    # 投影點
    dstPts = [(lane2x_d, 0),(lane1x_d, 0),(lane2x_d, parm.IMG_W), (lane1x_d, parm.IMG_W) ]
    # 透視變換矩陣
    M = cv2.getPerspectiveTransform(np.float32(srcPts), np.float32(dstPts)+1000)
    warped = cv2.warpPerspective(img_out2, M, (5000, 5000), flags=cv2.INTER_LINEAR)
    warped = cv2.resize(warped, (2500, 2500))
    # horizontal line
    # cv2.line(img_out2, (0, int(Vpoint[1])), (int(parm.IMG_H), int(Vpoint[1])),color=(0, 0, 255), thickness=3)
    # Vertical line
    # cv2.line(img_out2, (int(Vpoint[0]), int(Vpoint[1])), (int(Vpoint[0]), parm.IMG_W),color=(0, 0, 255), thickness=3)
    # ProjY horizontal line
    # cv2.line(img_out2, (0, ProjY), (int(parm.IMG_H), ProjY),color=(0, 255, 0), thickness=3)
    # cv2.line(img_out2, (0, int(transH)), (int(parm.IMG_H), int(transH)),color=(255, 255, 0), thickness=3)
    # Lane Line
    # cv2.line(img_out2, (lane2x_d, int(transH)), (int(Vpoint[0]), int(Vpoint[1])), (0, 0, 255), 5) #左
    # cv2.line(img_out2, (lane1x_d, int(transH)), (int(Vpoint[0]), int(Vpoint[1])), (0, 0, 255), 5) #右
    # src points
    # for i, pt in enumerate(srcPts):
    #     cv2.circle(img_out2, pt, 10, (255, 0, 0), -1)
    #     cv2.putText(img_out2,"P"+str(i+1),(pt[0]+10,pt[1]+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2)

    # cv2.circle(img_out2,(int(Vpoint[0]),int(Vpoint[1])), 10, (0, 255, 255), -1)
    # cv2.imshow("img_out", img_out2)
    # cv2.waitKey(0)

    # dst points
    # Transform the points
    # transformed = corresponding_coordinates(np.float32(srcPts) , M)
    # for i, pt in enumerate(transformed):
    #     cv2.circle(warped, pt, 10, (255, 0, 0), -1)

    # cv2.imwrite('/home/hsuan/results_img/orginal.jpg',img_out2)
    return img_out2, warped, (ProjY,M)

def crop_loc(ProjY,M,laneframe,parm):
    '''計算擷取位置'''
    crop_x =[]
    for id in (laneframe.laneIDs):
        x = int((ProjY - id.equa[1]) / id.equa[0]) #計算lane 在投影Y軸上的X值
        dst= corresponding_coordinates(np.float32([(x, ProjY)]),M)
        crop_x.append(dst[0][0])

    parm.crop = (int(min(crop_x) -400) if int(min(crop_x) -400) > 0 else 0, int(max(crop_x)+300))

def crop_lane(laneframe,warped,parm,pm):
    '''裁剪車道線'''
    lane_pic = []
    for i, laneid in enumerate(laneframe.laneIDs):
        cols, rows = zip(*laneid.allpoints)
        orginal_ps= list(zip(rows,cols))
        transformed = corresponding_coordinates(np.float32(orginal_ps) , pm[1])
        lanepoints= []
        for pt in transformed:
            if pt[1]>=0 and pt[1]<=int(parm.IMG_W) and pt[0]>= parm.crop[0] and pt[0] <=parm.crop[1]:
                lanepoints.append(pt)
                # cv2.circle(warped, pt, 10, (255, 0, 0), -1)
        trans_rows, trans_cols = zip(*lanepoints)
        medianpts = int((max(set(trans_rows))+min(set(trans_rows)))/2)
        w = 50
        # cv2.line(warped, (medianpts-w, 0), (medianpts-w, parm.IMG_W),color=(0, 0, 255), thickness=3)
        # cv2.line(warped, (medianpts+w, 0), (medianpts+w, parm.IMG_W),color=(0, 0, 255), thickness=3)
        # cv2.line(warped, (medianpts, 0), (medianpts, parm.IMG_W),color=(0, 255, 0), thickness=3)
        # cv2.line(warped, (0, 950), (int(parm.IMG_H), 950),color=(0, 255, 0), thickness=3)
        croplane = warped[0:950 , medianpts-w : medianpts+w]
        lane_pic.append(croplane)
    return lane_pic

def arr_type(laneframe,type):
    """product array """ #TODO
    arr = []
    if type == 4: #arr_type(LaneID_Info,4)
        for i, laneid in enumerate(laneframe):
            arr.append(laneid.name)
    elif type == 5:
        for i, laneid in enumerate(laneframe):
            arr.append(laneid.freq)
    elif type == 6:
        for i, laneid in enumerate(laneframe):
            arr.append(laneid.angle)
    else:
        for i, laneid in enumerate(laneframe.laneIDs):
            if type ==1:
                arr.append(laneid.equa[0])
            elif type ==2:
                arr.append(laneid.hor_x)
            elif type ==3:
                arr.append(laneid.angle)
            else:
                pass
    arr = np.array(arr)
    # if type == 3:
    #     arr = sorted(arr, key=lambda x:x[0])
    # else:
    #     arr = np.array(arr)
    return arr

def re_lane_angle(laneframe,LaneID_Info,maxID):
    nowframe = deepcopy(laneframe)
    allID = arr_type(LaneID_Info,4)
    allID_angle = arr_type(LaneID_Info,6)

    assigned = [False for _ in range(len(nowframe))]
    assigned_allID = [False for _ in range(len(allID))]

    C = np.zeros((len(allID_angle), len(nowframe)))
    
    for id_n, laneid_prev_angle in enumerate(allID_angle): #r
        for id_p, laneid_now in enumerate(nowframe.laneIDs): #c
            C[id_n, id_p]= abs(laneid_prev_angle - laneid_now.angle)

    # optimal linear assignment (Hungarian)
    row_ind, col_ind = linear_sum_assignment(C)

    for r, c in zip(row_ind, col_ind):
        if C[r, c] >20:
            continue
        nowframe.laneIDs[c].name = allID[r]
        assigned[c] = True
        assigned_allID[r] = True

        for i, laneID in enumerate(LaneID_Info): 
                if laneID.name == allID[r]:
                    laneID.angle = nowframe.laneIDs[c].angle
                    laneID.freq = 5
                continue

    for i, laneid_now in enumerate(nowframe.laneIDs):
        if assigned[i]:
            pass
        else:
            laneid_now.name = maxID + 1
            LaneID_Info.append(LaneID_info(maxID + 1 , 5 , laneid_now.angle))
            maxID += 1
    
    for idx,assigned_id in enumerate(assigned_allID):
        if assigned_id == False:
            LaneID_Info[idx].freq -= 1

    check_allID_freq = arr_type(LaneID_Info,5)
    for idx in range(len(check_allID_freq)-1,-1,-1):
        # print("i=",idx)
        if check_allID_freq[idx] == 0:
                del LaneID_Info[idx]
                # print(LaneID_Info)
                # input()

    # if frame_index >=23:
    #     print(assigned_allID)
    #     for i, laneID in enumerate(LaneID_Info): 
    #         print("info",laneID.name,laneID.freq,laneID.angle)
    #     for i, laneid in enumerate(lane_allframe[frame_index].laneIDs):
    #         print(laneid.equa,laneid.angle)
    #     print(C)
    #     print(row_ind, col_ind)
        # input()
    return nowframe,LaneID_Info,maxID


# def re_lane(lane_allframe,frame_index,slope_diff,change_lane,LaneID):
#     '''
#     Re-id the lanes for each frame
#     '''
#     nowframe, prevframe = deepcopy(lane_allframe[frame_index]), deepcopy(lane_allframe[frame_index-1])
#     prev_assigned = LaneID[1]
#     add_id = max(LaneID[0])
#     # print("prev_num_lane",prev_num_lane)
#     # print("LaneID[0]",LaneID[0])
#     if len(prev_assigned)> len(nowframe):
#         assigned = [False for _ in range(0,len(LaneID[0]))]
#         assigned_row = [False for _ in range(0,len(LaneID[0]))]
#     else:
#         assigned = [False for _ in range(0,len(nowframe))]
#         assigned_row = [False for _ in range(0,len(LaneID[0]))]
#     # print(assigned)
#     C = np.Inf*np.ones((len(prevframe), len(assigned)), dtype=np.float64)
#     if change_lane != False:
#         for id_n, laneid_prev in enumerate(prevframe.laneIDs):
#             for id_p, laneid_now in enumerate(nowframe.laneIDs):
#                 C[id_n, id_p]= abs(laneid_prev.hor_x - laneid_now.hor_x)
#     else:
#         for id_n, laneid_prev in enumerate(prevframe.laneIDs):
#             for id_p, laneid_now in enumerate(nowframe.laneIDs):
#                 C[id_n, id_p]= abs(laneid_prev.angle - laneid_now.angle)

#     # assign clusters to lane (in acsending order of error)
#     if len(prevframe) < len(LaneID[1]):
#         patch = np.Inf*np.ones(len(assigned), dtype=np.float64)
#         for idx,i in enumerate(LaneID[1]):
#             if i == False:
#                 C = np.insert(C,idx,[patch],axis= 0)
#     # print("C",C)
#     row_ind, col_ind = np.unravel_index(np.argsort(C, axis=None), C.shape)
#     for r, c in zip(row_ind, col_ind):
#         if math.isinf(C[r,c]) == False:
#             if False not in assigned_row:
#                 break
#             if C[r, c] >= slope_diff+7 and change_lane == False:
#                 break
#             if change_lane != False and C[r, c] >=200 :
#                 break
#             if assigned[c] or assigned_row[r]:
#                 continue
#             assigned[c] = True
#             assigned_row[r] = True
#             # update best lane match with current pixel
#             nowframe.laneIDs[c].name = LaneID[0][r]

#     if False in assigned:
#         # print("assigned",assigned)
#         for i, x in enumerate(assigned):
#             if x == False:
#                 if i > len(nowframe)-1:
#                     break
#                 else:
#                     if i+1 > len(LaneID[0]) or assigned_row[i]==True:

#                         assigned[i] = True
#                         nowframe.laneIDs[i].name = add_id+1
#                         add_id+=1
#                     else:
#                         nowframe.laneIDs[i].name = LaneID[0][i]
#                         assigned[i] = True
            
#     new_LaneID = arr_type(nowframe,3)

#     if len(new_LaneID) != len(set(new_LaneID)):
#         print("#320 error")
#         input()
#     lackidx = [new_LaneID.index(lackvalue) for lackvalue in new_LaneID if lackvalue not in LaneID[0]]  

#     for idx in lackidx:
#         if idx ==0:
#             LaneID[0].insert(0,new_LaneID[idx])
#         else:
#             LaneID[0].insert(LaneID[0].index(new_LaneID[idx-1])+1,new_LaneID[idx])

#     print("new_LaneID",new_LaneID)
#     print("LaneID[0]",LaneID[0])

#     # for idx in range(0,len(new_LaneID)-1):
#     #     prev = LaneID[0].index(new_LaneID[idx])
#     #     next = LaneID[0].index(new_LaneID[idx+1])
#     #     if prev > next:
#     #         LaneID[0][prev], LaneID[0][next] = new_LaneID[idx+1], new_LaneID[idx]

#     # new_LaneID = arr_type(nowframe,3)
#     # print("new_LaneID",new_LaneID)
#     # print("LaneID[0]",LaneID[0])

#     assigned = [False for _ in range(0,len(LaneID[0]))]
#     for idx in new_LaneID:
#         assigned[LaneID[0].index(idx)]  = True
    
#     LaneID = [LaneID[0],assigned]
#     lane_allframe[frame_index] = nowframe
    
#     # if frame_index>87:
#     #     print(lane_allframe[frame_index])
#     #     input()
    
#     return lane_allframe,LaneID

# def re_lane_f(A_laneframe,B_laneframe,prob_laneID,slope_diff):
#     # A_laneframe = lane_allframe[frame_index-1]
#     # B_laneframe = nowframe
#     for i, laneid_prev in enumerate(A_laneframe.laneIDs):
#         # print("laneid_prev",laneid_prev,'\n','B_laneframe',B_laneframe.laneIDs[i])
#         if (abs(laneid_prev.angle - B_laneframe.laneIDs[i].angle) <= slope_diff):
#             B_laneframe.laneIDs[i].name = laneid_prev.name
#             continue
#         elif (i+1) < len(B_laneframe): #如果不是最後一條道路線  
#             #+1(跳過自己) 先檢查後面的線
#             for j in range(i+1 ,len(B_laneframe)):
#                 if(abs(laneid_prev.angle - B_laneframe.laneIDs[j].angle) <= slope_diff):
#                     tem_lane = deepcopy(B_laneframe.laneIDs[i]) #暫時儲存要交換的道路線
#                     B_laneframe.laneIDs[j].name = laneid_prev.name
#                     B_laneframe.laneIDs[i] = deepcopy(B_laneframe.laneIDs[j])
#                     B_laneframe.laneIDs[j] = deepcopy(tem_lane)
#                     break
#             else: #如果沒找到，找問題線
#                 if prob_laneID: 
#                     for prob_index, prob in enumerate(prob_laneID):
#                         if(abs(laneid_prev.equa[0] - prob.equa[0]) <= slope_diff):
#                             B_laneframe.laneIDs[i].name = "prob_"+ B_laneframe.laneIDs[i].name
#                             prob_laneID.append(B_laneframe.laneIDs[i])

#                             prob.name = laneid_prev.name
#                             B_laneframe.laneIDs[i] = deepcopy(prob)
#                             del prob_laneID[prob_index]
#                             break
#                 else:
#                     # 如果都沒找到
#                     B_laneframe.laneIDs[i].name = "prob_"+ B_laneframe.laneIDs[i].name
#                     prob_laneID.append(B_laneframe.laneIDs[i])

#         else:#最後一條直接檢查問題線
#             if prob_laneID: 
#                 for prob_index, prob in enumerate(prob_laneID):
#                     if(abs(laneid_prev.angle - prob.angle) <= slope_diff):
#                         B_laneframe.laneIDs[i].name = "prob_"+ B_laneframe.laneIDs[i].name
#                         prob_laneID.append(B_laneframe.laneIDs[i])

#                         prob.name = laneid_prev.name
#                         B_laneframe.laneIDs[i] = deepcopy(prob)
#                         del prob_laneID[prob_index]
#                         break
#             else:
#                 # 如果都沒找到，把這條加到問題線裡# 
                
#                 B_laneframe.laneIDs[i].name = "prob_"+ B_laneframe.laneIDs[i].name
#                 prob_laneID.append(B_laneframe.laneIDs[i])
    
#     return A_laneframe, B_laneframe
# def re_lane_V2(lane_allframe,frame_index,tem, slope_diff,change_lane):
#     '''
#     Re-id the lanes for each frame
#     '''
#     nowframe, prevframe = deepcopy(lane_allframe[frame_index]), deepcopy(lane_allframe[frame_index-1])
#     prob_laneID = []
#     prob_frameids, prob_frames = zip(*tem)  #prob_frame

#     """ 如果現在這一幀跟前一幀抓到的道路線數一樣，用前一幀比對這一幀"""
#     if len(nowframe) == len(prevframe):
#         assigned = [False for _ in range(0,len(prevframe.laneIDs))]
#         # print("assigned",assigned)
#         C = np.Inf*np.ones((len(prevframe), len(nowframe)), dtype=np.float64)
#         if change_lane == "NEW":
#             for id_n, laneid_now in enumerate(prevframe.laneIDs):
#                 for id_p, laneid_prev in enumerate(nowframe.laneIDs):
#                     C[id_n, id_p]= abs(laneid_prev.hor_x - laneid_now.hor_x)
#         else:
#             for id_n, laneid_now in enumerate(prevframe.laneIDs):
#                 for id_p, laneid_prev in enumerate(nowframe.laneIDs):
#                     C[id_n, id_p]= abs(laneid_prev.angle - laneid_now.angle)
#         # print(C)
#         # assign clusters to lane (in acsending order of error)
#         row_ind, col_ind = np.unravel_index(np.argsort(C, axis=None), C.shape)
#         for r, c in zip(row_ind, col_ind):
#             # print("C[r, c]",C[r, c],r,c)
#             if C[r, c] >= slope_diff:
#                 if change_lane == True or "NEW":
#                     pass
#                 else:
#                     break
#             if assigned[c]:
#                 continue
#             assigned[c] = True
#             # update best lane match with current pixel
#             if nowframe.laneIDs[c].name != prevframe.laneIDs[r].name:
#                 nowframe.laneIDs[c].name = prevframe.laneIDs[r].name
#         print("assigned",assigned)
#         # input()
#         print("prevframe",prevframe,'\n','nowframe',nowframe)

#         # for c, cluster in enumerate(prevframe.laneIDs):
#         #     if len(cluster) == 0:
#         #         continue
#         #     # if not assigned[c]:

#         # _ ,nowframe = re_lane_f(prevframe,nowframe,prob_laneID, slope_diff)
#     elif len(nowframe) > len(prevframe):
#         print("now:",nowframe)
#         print("prev:",prevframe)
#         print("Q1")
#         input()
#         """ 如果現在這一幀(4)比前一幀(3)抓到的道路線數多，則用前一幀比對這一幀 """
#         tem.append((frame_index ,deepcopy(lane_allframe[frame_index]))) #Record the problem frame
#         if frame_index-1 in prob_frameids: #檢驗前一幀是不是也有相同問題
#             prob_prevframe = deepcopy(prob_frames[prob_frameids.index(frame_index-1)])
#             prob_prevframe.sort()

#             prob_prevframe_equa = arr_type(prob_prevframe,1)
#             nowframe_equa = arr_type(nowframe,1)
            
#             if len(prob_prevframe_equa) == len(nowframe_equa):
#                 cal_equa = nowframe_equa - prob_prevframe_equa

#                 if (np.array(abs(cal_equa))<= 0.5).all() :
#                     prevframe_equa = arr_type(prevframe,1)
#                     diff_idx = np.where(np.in1d(prob_prevframe_equa, prevframe_equa) == False)[0]
                    
#                     j = 0
#                     for i in diff_idx:
#                         prob_prevframe.laneIDs[int(i)].name = "LaneID_" + str(len(prevframe_equa)+j+1)
#                         prevframe.add_lane(prob_prevframe.laneIDs[int(i)])
#                         j = j+1

#                     # _ ,nowframe = re_lane_f(prevframe,nowframe,prob_laneID,0.5)
#                 else:
#                     #TODO 換道問題 導致 斜率都變了QQ
#                     diff_idx = np.where((np.array(abs(cal_equa))<= slope_diff) == False)[0]
#                     print(diff_idx)
#                     input()
#             else:
#                 #TODO 抓到的線長度不一樣
#                 assert("error2! 遇到了再處理")

#         else:
#             #把多的線列為有問題線
#             for k in range(len(lane_allframe[frame_index-1]) ,len(lane_allframe[frame_index])):
#                 nowframe.laneIDs[k].name = "prob_"+ lane_allframe[frame_index].laneIDs[k].name
#                 prob_laneID.append(nowframe.laneIDs[k])
#             del nowframe.laneIDs[len(lane_allframe[frame_index-1]):]

#             _ ,nowframe = re_lane_f(prevframe,nowframe,prob_laneID,slope_diff)

#     else:
#         # print("now:",nowframe)
#         # print("prev:",prevframe)
#         # print("Q2")
#         # input()
#         """ 如果現在這一幀(3)比前一幀(4)抓到的道路線數少，則用這一幀比對前一幀 """
#         tem.append((frame_index ,deepcopy(lane_allframe[frame_index]))) #Record the problem frame
#         assigned = [False for _ in range(0,len(prevframe.laneIDs))]
#         print("assigned",assigned)
#         C = np.Inf*np.ones((len(prevframe), len(nowframe)), dtype=np.float64)
#         if change_lane == "NEW":
#             for id_n, laneid_now in enumerate(prevframe.laneIDs):
#                 for id_p, laneid_prev in enumerate(nowframe.laneIDs):
#                     C[id_n, id_p]= abs(laneid_prev.hor_x - laneid_now.hor_x)
#         else:
#             for id_n, laneid_now in enumerate(prevframe.laneIDs):
#                 for id_p, laneid_prev in enumerate(nowframe.laneIDs):
#                     C[id_n, id_p]= abs(laneid_prev.angle - laneid_now.angle)
        
#         # print(C)
#         # assign clusters to lane (in acsending order of error)
#         row_ind, col_ind = np.unravel_index(np.argsort(C, axis=None), C.shape)
#         for r, c in zip(row_ind, col_ind):
#             # print("C[r, c]",C[r, c],r,c)
#             if C[r, c] >= slope_diff:
#                 if change_lane == True or "NEW":
#                     pass
#                 else:
#                     break
#             if assigned[c]:
#                 continue
#             assigned[c] = True
#             # update best lane match with current pixel
#             if nowframe.laneIDs[c].name != prevframe.laneIDs[r].name:
#                 nowframe.laneIDs[c].name = prevframe.laneIDs[r].name
        
        
#         # if frame_index-1 in prob_frameids: #檢驗前一幀是不是也有相同問題 #TODO 
#         #     assert False, 'error3! 遇到了再處理'

#         # #把多的線列為有問題線 
#         # for k in range(len(lane_allframe[frame_index]) ,len(lane_allframe[frame_index-1])):
#         #     prevframe.laneIDs[k].name = "prob_"+ lane_allframe[frame_index-1].laneIDs[k].name
#         #     prob_laneID.append(prevframe.laneIDs[k])
#         # del prevframe.laneIDs[len(lane_allframe[frame_index]):]

#         # nowframe,_ = re_lane_f(nowframe,prevframe,prob_laneID,slope_diff)

#     #移除無法配對到的問題線
#     for idx, laneid in enumerate(nowframe.laneIDs):
#         print("laneid",laneid.name,laneid.equa[0])
#         if "prob" in laneid.name:
#             if change_lane ==True: #TODO
#                 nowframe.laneIDs[idx].name = laneid.name.replace("prob_","")
#             else:
#                 del nowframe.laneIDs[idx]

#     lane_allframe[frame_index] = nowframe

#     return lane_allframe,tem