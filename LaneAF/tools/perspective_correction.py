import numpy as np
import cv2
from copy import deepcopy
from typing import Dict, List
import pysnooper
#@pysnooper.snoop()
class Lane:
    '''
    It's used to build each lane of per frame.
    name: LaneID_1, LaneID_2, LaneID_3...
    equa: equa[0] = slope, equa[1] = intercept build the line
    centerpoints: center point of each row of per frame.(no magnification)
    allpoints: allpoints of per frame(magnification)
    lanetype: #TODO
    '''
    def __init__(self,lane_name,equa,centerpoints, allpoints):
        self.name = lane_name
        self.equa = equa #中心線
        self.centerpoints = centerpoints
        self.allpoints = allpoints #所有點(已經放大)
        self.lanetype = "undefined"
    
    def __str__(self):
        return f"Name is {self.name}, Equa is {self.equa}"

    @property
    def min_axis_x(self):
        return self.allpoints.min(axis=0)

    @property
    def max_axis_x(self):
        return self.allpoints.max(axis=0)

def combline_f(testlane,comb_equa,comb_centerpoints,AllPoints,updated_lane_allpoints):
    '''the func is to update infro for centerline.'''
    testlane.equa = comb_equa
    testlane.centerpoints = comb_centerpoints
    comb_allpoints = np.concatenate((testlane.allpoints,np.array(AllPoints)*8))
    comb_allpoints = comb_allpoints[comb_allpoints[:,0].argsort()]
    updated_lane_allpoints.allpoints = comb_allpoints

def centerline(seg_out,laneframe, ego_box):
    """
    Use the polyfit function to build centerlines.
    :return: laneframe: all lanes Info in this frame.
    """
    ID = np.delete(np.unique(seg_out), [0]) #number of lanes
    checkslope = []
    for i in ID:
        cal_seg = seg_out.copy()
        cal_seg[cal_seg != i] = 0
        CenterPoints = []
        AllPoints = []

        for row in range(np.shape(cal_seg)[0]): # search each row
            col_loc = []
            for j, v in enumerate(cal_seg[row]):
                if (v == i) :
                    # if the point is in the ego vehicle, it represents a misjudgment. pass it!
                    if ego_box != None and j*8 >= ego_box['x1'] and j*8 <= ego_box['x2'] and row*8 >= ego_box['y1'] and row*8 <= ego_box['y2'] :
                        continue
                    col_loc.append(j)
                    AllPoints.append([row, j])

            if (np.isnan(np.median(col_loc)) == False):
                CenterPoints.append([row, int(np.median(col_loc))]) #CenterPoints : get the median point for row

        if len(AllPoints) < 20: # the lane is too small
            print("the lane is too small")
            continue

        #build the centerlines
        y, x = zip(*CenterPoints)

        x_scale = [i*8 for i in x] #Enlarge to original size
        y_scale = [i*8 for i in y]
        equa = np.polyfit(x_scale, y_scale, 1)
        #Check for similar lines(slope)
        if len(checkslope) != 0 :

            x = [x for x in checkslope if abs(x-equa[0]) < 0.15] #threshold: 0.1
            if len(x) != 0:
                testlane = laneframe.laneIDs[checkslope.index(x)]
                comb_centerpoints = np.concatenate((testlane.centerpoints,CenterPoints))
                comb_centerpoints = comb_centerpoints[comb_centerpoints[:,0].argsort()]

                y_p, x_p = zip(*comb_centerpoints)
                comb_x_scale = [i*8 for i in x_p] #放大成原尺寸
                comb_y_scale = [i*8 for i in y_p]
                comb_equa = np.polyfit(comb_x_scale, comb_y_scale, 1) #建立線

                #Check if the new centerline's slope is between the two lines' slope. True means they are the same lane.
                if (abs(equa[0] - comb_equa[0]) <= 0.1 and abs(comb_equa[0] - testlane.equa[0]) <= 0.1):
                    combline_f(testlane,comb_equa,comb_centerpoints,AllPoints,laneframe.laneIDs[checkslope.index(x)])
                    continue
                elif testlane.equa[0] <= equa[0]:
                    if equa[0] - comb_equa[0] >= 0 and comb_equa[0] - testlane.equa[0] >= 0: #0_Road031_Trim003_frames
                        combline_f(testlane,comb_equa,comb_centerpoints,AllPoints,laneframe.laneIDs[checkslope.index(x)])
                        continue
                else:
                    if comb_equa[0] - testlane.equa[0] <=0 and comb_equa[0] - equa[0] >=0:  #0_Road014_Trim005_frames
                        combline_f(testlane,comb_equa,comb_centerpoints,AllPoints,laneframe.laneIDs[checkslope.index(x)])
                        continue

        checkslope.append(equa[0])
        #add lane to laneframe
        laneframe.add_lane(Lane(("LaneID_"+str(i)),equa,np.array(CenterPoints), np.array(AllPoints)*8))
    
    return laneframe

def lane_loc_f(laneframe) -> Dict:
    """
    Determine the location of the lanes.
    only judge by the slope.
    #TODO judge in another way
    """
    left_loc=[]
    right_loc=[]

    for id in laneframe.laneIDs:

        slope = id.equa[0]

        if slope < 0: #因為opencv座標關係
            left_loc.append(id)#.name,id.equa])
        else :
            right_loc.append(id)#.name,id.equa])

    left_loc.sort(key=lambda x: x.equa[0])        
    right_loc.sort(key=lambda x: x.equa[0])

    lane_loc = {"leftmost":left_loc[-1],"left_near_center":left_loc[0],"right_near_center":right_loc[-1],"rightmost":right_loc[0]}
    return lane_loc

def vanishing_point(lane_loc: Dict)-> List:
    """
    Calculate the vanishing point through the intersection of left_near_center and right_near_center
    """
    P_diff = np.polysub(lane_loc["left_near_center"].equa, lane_loc["right_near_center"].equa)
    Vx = np.roots(P_diff)
    Vy = np.polyval(lane_loc["left_near_center"].equa, Vx)
    Vpoint = np.append(Vx, Vy)
    return Vpoint

def corresponding_coordinates(pos,M):
    """透視變換 每個座標對應"""
    u = pos[0]
    v = pos[1]
    x = (M[0][0]*u+M[0][1]*v+M[0][2])/(M[2][0]*u+M[2][1]*v+M[2][2])
    y = (M[1][0]*u+M[1][1]*v+M[1][2])/(M[2][0]*u+M[2][1]*v+M[2][2])
    return (int(x), int(y))

def perspective_transform(parm, egoH, Vpoint: List, lane_loc: Dict, img_out):
    """
    Convert to a bird's-eye view with perspective transformation
    """
    transH = parm.IMG_W
    if egoH: # If there is a hood below the picture
        transH = egoH

    # 計算投影Y軸
    ProjY = int((transH-Vpoint[1])*0.25+Vpoint[1])
    # 取 left_near_center right_near_center Vertical line
    lane1x_u = int((ProjY - lane_loc["left_near_center"].equa[1]) / lane_loc["left_near_center"].equa[0]) 
    lane2x_u = int((ProjY - lane_loc["right_near_center"].equa[1]) / lane_loc["right_near_center"].equa[0]) 
    lane1x_d = int((transH - lane_loc["left_near_center"].equa[1]) / lane_loc["left_near_center"].equa[0]) 
    lane2x_d = int((transH - lane_loc["right_near_center"].equa[1]) / lane_loc["right_near_center"].equa[0])
    # 原點
    srcPts = np.float32([(lane1x_u, int(ProjY)),(lane2x_u, int(ProjY)),(lane1x_d, int(transH)), (lane2x_d, int(transH))]) #(左上 右上 右下 左下)
    
    img_out2 = img_out.copy()
    # 投影點
    dstPts = np.float32([(lane1x_d, 0), (lane2x_d, 0),(lane1x_d, parm.IMG_W), (lane2x_d, parm.IMG_W)])
    # 透視變換矩陣
    M = cv2.getPerspectiveTransform(srcPts, dstPts+100)
    warped = cv2.warpPerspective(img_out2, M, (5000, 5000), flags=cv2.INTER_LINEAR)
    
    # horizontal line
    cv2.line(img_out2, (0, int(Vpoint[1])), (int(parm.IMG_H), int(Vpoint[1])),color=(0, 0, 255), thickness=3)
    # Vertical line
    cv2.line(img_out2, (int(Vpoint[0]), int(Vpoint[1])), (int(Vpoint[0]), parm.IMG_W),color=(0, 0, 255), thickness=3)
    # ProjY horizontal line
    cv2.line(img_out2, (0, ProjY), (int(parm.IMG_H), ProjY),color=(0, 255, 0), thickness=3)
    cv2.line(img_out2, (0, int(transH)), (int(parm.IMG_H), int(transH)),color=(255, 255, 0), thickness=3)
    # src points
    cv2.circle(img_out2, (lane1x_u, int(ProjY)),10, (255, 0, 0), 4)
    cv2.circle(img_out2, (lane2x_u, int(ProjY)),10, (255, 0, 0), 4)
    cv2.circle(img_out2, (lane1x_d, int(transH)),10, (255, 0, 0), 4)
    cv2.circle(img_out2, (lane2x_d, int(transH)),10, (255, 0, 0), 4)
    return img_out2, warped, (ProjY,M)

def crop_loc(ProjY,M,laneframe,parm):
    '''計算擷取位置'''
    crop_x =[]
    for id in (laneframe.laneIDs):
        x = int((ProjY - id.equa[1]) / id.equa[0]) #計算lane 在投影Y軸上的X值
        dst= corresponding_coordinates((int(x), int(ProjY)),M)
        crop_x.append(dst[0])
    parm.crop = (round(min(crop_x)-200,-2) if round(min(crop_x)-200,-2) > 0 else 0, round(max(crop_x)+200,-2))

def re_lane_f(A_laneframe,B_laneframe,prob_laneID,slope_diff):
    # A_laneframe = lane_allframe[frame_index-1]
    # B_laneframe = nowframe
    for i, laneid_prev in enumerate(A_laneframe.laneIDs):
        if (abs(laneid_prev.equa[0] - B_laneframe.laneIDs[i].equa[0]) <= slope_diff):
            B_laneframe.laneIDs[i].name = laneid_prev.name
            continue
        elif (i+1) < len(B_laneframe): #如果不是最後一條道路線  
            #+1(跳過自己) 先檢查後面的線
            for j in range(i+1 ,len(B_laneframe)):
                if(abs(laneid_prev.equa[0] - B_laneframe.laneIDs[j].equa[0]) <= slope_diff):
                    tem_lane = B_laneframe.laneIDs[i] #暫時儲存要交換的道路線
                    B_laneframe.laneIDs[j].name = laneid_prev.name
                    B_laneframe.laneIDs[i] = B_laneframe.laneIDs[j]
                    B_laneframe.laneIDs[j] = tem_lane
                    break
                if (j == (len((B_laneframe))-1)): #判斷是最後一條線，檢查問題線
                    if len(prob_laneID) != 0:
                        for prob_index, prob in enumerate(prob_laneID):
                            if(abs(laneid_prev.equa[0] - prob.equa[0]) <= slope_diff):
                                B_laneframe.laneIDs[i].name = "prob_"+ B_laneframe.laneIDs[i].name
                                prob_laneID.append(B_laneframe.laneIDs[i])
                                prob.name = laneid_prev.name
                                B_laneframe.laneIDs[i] = prob
                                del prob_laneID[prob_index]
                                break
                        else:
                            # 如果都沒找到
                            input()
                            B_laneframe.laneIDs[i].name = "prob_"+ B_laneframe.laneIDs[i].name
                            prob_laneID.append(B_laneframe.laneIDs[i])
                            break
        else:#最後一條直接檢查問題線
            if len(prob_laneID) != 0:
                for prob_index, prob in enumerate(prob_laneID):
                    if(abs(laneid_prev.equa[0] - prob.equa[0]) <= slope_diff):
                        B_laneframe.laneIDs[i].name = "prob_"+ B_laneframe.laneIDs[i].name
                        prob_laneID.append(B_laneframe.laneIDs[i])
                        prob.name = laneid_prev.name
                        B_laneframe.laneIDs[i] = prob
                        del prob_laneID[prob_index]
                        break
                else:
                    # 如果都沒找到，把這條加到問題線裡
                    input()
                    B_laneframe.laneIDs[i].name = "prob_"+ B_laneframe.laneIDs[i].name
                    prob_laneID.append(B_laneframe.laneIDs[i])
                    break
    
    return A_laneframe, B_laneframe



def re_lane(lane_allframe,frame_index,tem, slope_diff):
    '''
    Re-id the lanes for each frame
    '''
    prob_laneID = []
    nowframe = deepcopy(lane_allframe[frame_index])
    prevframe = deepcopy(lane_allframe[frame_index-1])
    """ 如果現在這一幀跟前一幀抓到的道路線數一樣，用前一幀比對這一幀"""
    if len(nowframe) == len(prevframe):
        _ ,nowframe = re_lane_f(prevframe,nowframe,prob_laneID, slope_diff)

    elif len(nowframe) > len(prevframe):
        """ 如果現在這一幀比前一幀抓到的道路線數多，則用前一幀比對這一幀 """
        prob_frameids, b = zip(*tem) #檢驗前一幀是不是也有相同問題 #TODO
        if frame_index-1 and frame_index-2 in prob_frameids:
            assert False, 'error2! 遇到了再處理'
        
        tem.append((frame_index ,deepcopy(lane_allframe[frame_index]))) #Record the problem frame
        a, b = zip(*tem)

        #把多的線列為有問題線 
        for k in range(len(lane_allframe[frame_index-1]) ,len(lane_allframe[frame_index])):
            nowframe.laneIDs[k].name = "prob_"+ lane_allframe[frame_index].laneIDs[k].name
            prob_laneID.append(nowframe.laneIDs[k])
            del nowframe.laneIDs[k]

        _ ,nowframe = re_lane_f(prevframe,nowframe,prob_laneID,slope_diff)

    else:
        """ 如果現在這一幀比前一幀抓到的道路線數少，則用這一幀比對前一幀 """
        prob_frameids, b = zip(*tem) #檢驗前一幀是不是也有相同問題 #TODO
        if frame_index-1 and frame_index-2 in prob_frameids:
            assert False, 'error3! 遇到了再處理'
        
        tem.append((frame_index ,deepcopy(lane_allframe[frame_index]))) #Record the problem frame
        a, b = zip(*tem)

        #把多的線列為有問題線 
        for k in range(len(lane_allframe[frame_index]) ,len(lane_allframe[frame_index-1])):
            prevframe.laneIDs[k].name = "prob_"+ lane_allframe[frame_index-1].laneIDs[k].name
            prob_laneID.append(prevframe.laneIDs[k])
            del prevframe.laneIDs[k]

        nowframe,_ = re_lane_f(prevframe,prevframe,prob_laneID,slope_diff)

    #移除無法配對到的問題線
    for index, laneid in enumerate(nowframe.laneIDs):
        # print("laneid",laneid.name,laneid.equa[0])
        if "prob" in  laneid.name:
            del nowframe.laneIDs[index]

    lane_allframe[frame_index] = nowframe

    return lane_allframe,tem