import os
from subprocess import call
import sys
import logging
if __name__ == "__main__":
    allfile = [_ for _ in os.listdir("/media/hsuan/data/VIL100/videos") if _.endswith(".avi")]
    allfile = sorted(allfile)
    sys.path.append("/home/hsuan/Thesis/LaneAF/")

    script_descriptor = open("/home/hsuan/Thesis/LaneAF/infer_vil100.py")
    a_script = script_descriptor.read()
    sys.argv = ["infer_vil100.py", "--input_video='2_Road017_Trim001_frames.avi'"]
    for idx, i in enumerate(allfile):
        print("idx",idx,"video=",i)
        if idx > 91:
            sys.argv[1] = '--input_video='+i
            exec(a_script)
            # input()
