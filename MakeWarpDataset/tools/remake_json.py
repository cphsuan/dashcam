import base64
import json
import io
import os
import os.path as osp
import argparse
from copy import deepcopy

from PIL import Image
from labelme.logger import logger
from labelme import PY2
from labelme import QT4

parser = argparse.ArgumentParser('Remake Warp Json...')
parser.add_argument('--dataset-WarpIMGdir', type=str , default='/media/hsuan/data/WarpDataset/VIL100/JPEGImages/', help='path to Warp IMG dataset')
parser.add_argument('--output-Jsondir', type=str , default='/media/hsuan/data/WarpDataset/VIL100/Json/', help='path to output Json')
args = parser.parse_args()


if __name__ == "__main__":
    # ParentPath
    WarpIMGdir = args.dataset_WarpIMGdir
    # allFileList = sorted(os.listdir(os.path.join(parentPath, "JPEGImages")))
    file = "0_Road014_Trim004_frames"
    imgPerFile = sorted(os.listdir(os.path.join(WarpIMGdir, file)))
    print(imgPerFile)

    for frameIndex, frame in enumerate(imgPerFile):
        if frameIndex == 0:
            referjsonPath = os.path.join(args.output_Jsondir, file, frame.replace('jpg', 'json'))

            with open(referjsonPath) as f:
                jsonData = json.load(f)
            referJson = deepcopy(jsonData)
            imagePath = referJson["imagePath"].replace(frame,"")

        else:
            jsonPath = os.path.join(args.output_Jsondir, file, frame.replace('jpg', 'json'))
            referJson["imagePath"] = imagePath+frame
            print(imagePath+frame)
            imageData = Image.open(os.path.join(WarpIMGdir, file, frame))

            with io.BytesIO() as f:
                ext = osp.splitext(frame)[1].lower()
                if PY2 and QT4:
                    format = 'PNG'
                elif ext in ['.jpg', '.jpeg']:
                    format = 'JPEG'
                else:
                    format = 'PNG'
                imageData.save(f, format=format)
                f.seek(0)
                image_bytes = f.read()

            imageData=base64.b64encode(image_bytes).decode('utf-8')
            referJson["imageData"] = imageData
            
            if isinstance(referJson, bytes):
                referJson = str(referJson, encoding='utf-8')
             
            with open(jsonPath, 'w') as wf:
                json.dump(referJson, wf, indent=4)


