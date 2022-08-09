import math

class LanePerFrame:
    def __init__(self, frameid):
        self.name = "Lane_Frame"+frameid
        self.laneIDs = list()
    
    def add_lane(self, laneID):
        self.laneIDs.append(laneID)

    def __str__(self):
        return f"LaneFrame_Name is {self.name}, laneIDs is {self.laneIDs}"

    def __len__(self):
        return len(self.laneIDs)

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
    