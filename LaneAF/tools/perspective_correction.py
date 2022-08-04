import numpy as np
import cv2
from copy import deepcopy

class Lane:
    def __init__(self,lane_name,equa,allpoints):
        self.name = lane_name
        self.equa = equa #中心線
        self.allpoints = allpoints #所有點(已經放大) 
    
    def __str__(self):
        return f"Name is {self.name}, Equa is {self.equa}"

    @property
    def min_axis_x(self):
        return self.allpoints.min(axis=0)

    @property
    def max_axis_x(self):
        return self.allpoints.max(axis=0)

def centerline(seg_out,laneframe):
    """建立中心線"""
    ID = np.delete(np.unique(seg_out), [0]) #有幾條道路線

    for i in ID:
        cal_seg = seg_out.copy()
        cal_seg[cal_seg != i] = 0
        CenterPoints = []
        AllPoints = []

        for row in range(np.shape(cal_seg)[0]): #每條row去找
            col_loc = []
            for j, v in enumerate(cal_seg[row]):
                if (v == i):
                    col_loc.append(j)
                    AllPoints.append([j,row])

            if (np.isnan(np.median(col_loc)) == False):
                CenterPoints.append([row, int(np.median(col_loc))]) #row的中心位置

        #建立中心線
        y, x = zip(*CenterPoints)

        x_scale = [i*8 for i in x] #放大成原尺寸
        y_scale = [i*8 for i in y]
        equa = np.polyfit(x_scale, y_scale, 1) #建立線

        #建立lane class
        laneframe.add_lane(Lane(("LaneID_"+str(i)),equa,np.array(AllPoints)*8))
    
    return laneframe

def lane_loc_f(laneframe):
    """道路線相對位置"""
    left_loc=[]
    right_loc=[]

    for id in laneframe.laneIDs:

        slope = id.equa[0]

        if slope < 0: #因為opencv座標關係
            left_loc.append([id.name,id.equa])
        else :
            right_loc.append([id.name,id.equa])

    lane_loc = {"leftmost":min(left_loc),"left_near_center":max(left_loc),"right_near_center":min(right_loc),"rightmost":max(right_loc)}
    return lane_loc

def vanishing_point(lane_loc):
    """ 找消失點(left_near_center,right_near_center) """
    P_diff = np.polysub(lane_loc["left_near_center"][1], lane_loc["right_near_center"][1])
    Vx = np.roots(P_diff)
    Vy = np.polyval(lane_loc["left_near_center"][1], Vx)
    Vpoint = np.append(Vx, Vy)
    return Vpoint

def corresponding_coordinates(pos,M):
    """透視變換 每個座標對應"""
    u = pos[0]
    v = pos[1]
    x = (M[0][0]*u+M[0][1]*v+M[0][2])/(M[2][0]*u+M[2][1]*v+M[2][2])
    y = (M[1][0]*u+M[1][1]*v+M[1][2])/(M[2][0]*u+M[2][1]*v+M[2][2])
    return (int(x), int(y))

def perspective_transform(parm, Vpoint, lane_loc, img_out):
    """ 透視變換 """
    # 計算投影Y軸
    ProjY = int((parm.IMG_W-Vpoint[1])*0.25+Vpoint[1])
    # 取 left_near_center right_near_center Vertical line
    lane1x_u = int((ProjY - lane_loc["left_near_center"][1][1]) / lane_loc["left_near_center"][1][0]) 
    lane2x_u = int((ProjY - lane_loc["right_near_center"][1][1]) / lane_loc["right_near_center"][1][0]) 
    lane1x_d = int((parm.IMG_W - lane_loc["left_near_center"][1][1]) / lane_loc["left_near_center"][1][0]) 
    lane2x_d = int((parm.IMG_W - lane_loc["right_near_center"][1][1]) / lane_loc["right_near_center"][1][0])
    # 原點
    srcPts = np.float32([(lane1x_u, int(ProjY)),(lane2x_u, int(ProjY)),(lane1x_d, parm.IMG_W), (lane2x_d, parm.IMG_W)]) #(左上 右上 右下 左下)
    # src points
    img_out2 = img_out.copy()
    # cv2.circle(img_out2, (lane1x_u, int(ProjY)),10, (255, 255, 0), 4)
    # cv2.circle(img_out2, (lane2x_u, int(ProjY)),10, (255, 255, 0), 4)
    # cv2.circle(img_out2, (lane1x_d, IMG_W),10, (255, 255, 0), 4)
    # cv2.circle(img_out2, (lane2x_d, IMG_W),10, (255, 255, 0), 4)
    # 投影點
    dstPts = np.float32([(lane1x_d, 0), (lane2x_d, 0),(lane1x_d, parm.IMG_W), (lane2x_d, parm.IMG_W)])
    # 透視變換矩陣
    M = cv2.getPerspectiveTransform(srcPts, dstPts+100)
    warped = cv2.warpPerspective(img_out2, M, (5000, 5000), flags=cv2.INTER_LINEAR)
    
    img_out2 = img_out.copy()
    # horizontal line
    cv2.line(img_out2, (0, int(Vpoint[1])), (int(parm.IMG_H), int(Vpoint[1])),color=(0, 0, 255), thickness=3)
    # Vertical line
    cv2.line(img_out2, (int(Vpoint[0]), int(Vpoint[1])), (int(Vpoint[0]), parm.IMG_W),color=(0, 0, 255), thickness=3)
    # ProjY horizontal line
    cv2.line(img_out2, (0, ProjY), (int(parm.IMG_H), ProjY),color=(0, 255, 0), thickness=3)
    # src points
    cv2.circle(img_out2, (lane1x_u, int(ProjY)),10, (255, 0, 0), 4)
    cv2.circle(img_out2, (lane2x_u, int(ProjY)),10, (255, 0, 0), 4)
    cv2.circle(img_out2, (lane1x_d, int(parm.IMG_W)),10, (255, 0, 0), 4)
    cv2.circle(img_out2, (lane2x_d, int(parm.IMG_W)),10, (255, 0, 0), 4)
    return img_out2, warped, (ProjY,M)

def crop_loc(ProjY,M,laneframe,parm):
    #計算擷取位置
    crop_x =[]
    for id in (laneframe.laneIDs):
        x = int((ProjY - id.equa[1]) / id.equa[0]) #計算lane 在投影Y軸上的X值
        dst= corresponding_coordinates((int(x), int(ProjY)),M)
        crop_x.append(dst[0])
    parm.crop = (round(min(crop_x)-200,-2), round(max(crop_x)+200,-2))

def re_lane(lane_allframe,frame_index,tem, slope_diff):
    """ 如果現在這一幀跟前一幀抓到的道路線數一樣 """
    if len(lane_allframe[frame_index]) == len(lane_allframe[frame_index-1]):
        for i, laneid in enumerate(lane_allframe[frame_index].laneIDs):
            if (laneid.name == lane_allframe[frame_index-1].laneIDs[i].name) & (abs(laneid.equa[0] -lane_allframe[frame_index-1].laneIDs[i].equa[0]) <= slope_diff):
                # print('第{}幀的道路線{}，前一幀的道路線{}'.format(frame_index,laneid.name,lane_allframe[frame_index-1].laneIDs[i].name))
                # print("new",laneid.equa[0])
                # print("prev",lane_allframe[frame_index-1].laneIDs[i].equa[0])
                pass
            else:
                assert False, 'error! 遇到了再處理'
            """ 如果現在這一幀比前一幀抓到的道路線數多，則用前一幀比對這一幀 """
    elif len(lane_allframe[frame_index]) > len(lane_allframe[frame_index-1]):
        # print('第{}幀有{}條道路線，前一幀有{}條道路線'.format(frame_index,len(lane_allframe[frame_index]),len(lane_allframe[frame_index-1])))
        a, b = zip(*tem) #檢驗前一幀是不是也有相同問題
        if frame_index-1 in a:
            assert False, 'error! 遇到了再處理'
        
        tem.append((frame_index ,deepcopy(lane_allframe[frame_index]))) #紀錄問題幀
        a, b = zip(*tem)

        prob_laneID = []
        #把多的線列為有問題線
        for k in range(len(lane_allframe[frame_index-1]) ,len(lane_allframe[frame_index])):
            lane_allframe[frame_index].laneIDs[k].name = "prob_"+ lane_allframe[frame_index].laneIDs[k].name
            prob_laneID.append(lane_allframe[frame_index].laneIDs[k])
            del lane_allframe[frame_index].laneIDs[k]
        
        #判斷
        for i, laneid_prev in enumerate(lane_allframe[frame_index-1].laneIDs):

            if (laneid_prev.name == lane_allframe[frame_index].laneIDs[i].name) & (abs(laneid_prev.equa[0] - lane_allframe[frame_index].laneIDs[i].equa[0]) <= slope_diff):
                pass
            else:
                # print(laneid_prev.name)
                lane_allframe[frame_index].laneIDs[i].name = "prob_"+ lane_allframe[frame_index].laneIDs[i].name
                prob_laneID.append(lane_allframe[frame_index].laneIDs[i])
                #+1(跳過自己) 檢查後面的線
                for j in range(i+1 ,len(lane_allframe[frame_index-1])):
                    if(abs(laneid_prev.equa[0] - lane_allframe[frame_index].laneIDs[j].equa[0]) <= slope_diff):
                        assert False, 'error! 遇到了再處理'
                
                #再檢查問題線
                for prob_index, prob in enumerate(prob_laneID):
                    if(abs(laneid_prev.equa[0] - prob.equa[0]) <= slope_diff):
                        del prob_laneID[prob_index]
                        prob.name = laneid_prev.name
                        lane_allframe[frame_index].add_lane(prob)
                        break

                #移除問題線
                for index, laneid_prev in enumerate(lane_allframe[frame_index].laneIDs):
                    if "prob" in  laneid_prev.name:
                        del lane_allframe[frame_index].laneIDs[index]
                
        """如果現在這一幀比前一幀抓到的道路線數少"""
    else:
        print('第{}幀有{}條道路線，前一幀有{}條道路線'.format(frame_index,len(lane_allframe[frame_index]),len(lane_allframe[frame_index-1])))
        pass

    return lane_allframe,tem