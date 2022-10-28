import math
from copy import deepcopy

from operator import attrgetter
class LanePerFrame:
    """
    All lanes Info in this frame.
    name: Lane_Frame1, Lane_Frame2, Lane_Frame3...
    laneIDs: store all lanes Info from Class Lane, its name: LaneID_1, LaneID_2, LaneID_3...
    """
    def __init__(self, frameid):
        self.name = "Lane_Frame"+frameid
        self.Vpoint = "undefined"
        self.laneIDs = list()
    
    def add_lane(self, laneID):
        self.laneIDs.append(laneID)

    def __str__(self):
        return f"LaneFrame_Name is {self.name},VPoint is {self.Vpoint} ,laneID name is {[i.name for i in self.laneIDs]}, laneID equa is {[i.equa[0] for i in self.laneIDs]}, Hor_x is {[i.hor_x for i in self.laneIDs]}, Angle is {[i.angle for i in self.laneIDs]}"

    def __len__(self):
        return len(self.laneIDs)
    
    def sort(self):
        self.laneIDs.sort(key=attrgetter('hor_x'))
        # sorted(self.laneIDs , key=lambda x: x.hor_x)

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

class ParmGT:
    def __init__(self,IMG_H, IMG_W):
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W

    # @property
    # def IMG_H(self):
    #     return self.__IMG_H
    
    # @property
    # def IMG_W(self):
    #     return self.__IMG_W
    