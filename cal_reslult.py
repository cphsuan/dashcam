import os
from subprocess import call
import sys
import logging
import json 
if __name__ == "__main__":
    ID1, ID2, ID3, ID4, ID5, ID6 =0,0,0,0,0,0
    video = '6_Road022_Trim001_frames'
    allfile = [_ for _ in os.listdir(os.path.join('/media/hsuan/data/VIL100/Json/',video)) if _.endswith(".json")]
    allfile = sorted(allfile)
    for idx, name in enumerate(allfile):
        with open(os.path.join('/media/hsuan/data/VIL100/Json/',video,name)) as f:
            data = json.load(f)
            for laneid in data["annotations"]["lane"]:
                if laneid['id'] ==1:
                    ID1 +=1
                elif laneid['id'] ==2:
                    ID2 +=1
                elif laneid['id'] ==3:
                    ID3 +=1
                elif laneid['id'] ==4:
                    ID4 +=1
                elif laneid['id'] ==5:
                    ID5 +=1
                else:
                    ID6 +=1
    print(ID1, ID2, ID3, ID4, ID5, ID6)
