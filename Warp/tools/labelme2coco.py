import os
import json
import numpy as np
import glob
import shutil
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
np.random.seed(41)

#0為背景
# classname_to_id = ["single white solid", "single white dotted"]
classname_to_id = ["ego vehicle", "rider", "bicycle", "bus", "car","caravan", "motorcycle", "trailer", "train", "truck"]

class_dic = dict()
for index,value in enumerate(classname_to_id):
  class_dic[value] = index+1


class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    # 由json文件建立COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 新增類別
    def _init_categories(self):
        for k, v in class_dic.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 新增COCO的image
    def _image(self, obj, path):
        image = {}
        from labelme import utils
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # 建立COCO的annotation
    def _annotation(self, shape):
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(class_dic[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 讀取json文件，輸出json
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 對應COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

parser = argparse.ArgumentParser('Labelme to Coco...')
parser.add_argument('--labelme-path', type=str , default='/media/hsuan/data/BDD_egoVeh/train', help='path to Warp IMG dataset')
# "/media/hsuan/data/WarpDataset/VIL100/JPEGImages/0_Road014_Trim004_frames"
parser.add_argument('--output-cocodir', type=str , default='/home/hsuan/Thesis/mmdetection/data/', help='path to output Img')
args = parser.parse_args()


if __name__ == '__main__':

    labelme_path = args.labelme_path
    saved_coco_path = args.output_cocodir
    # 建立文件
    if not os.path.exists("%scoco/annotations/"%saved_coco_path):
        os.makedirs("%scoco/annotations/"%saved_coco_path)
    if not os.path.exists("%scoco/train2017/"%saved_coco_path):
        os.makedirs("%scoco/train2017"%saved_coco_path)
    if not os.path.exists("%scoco/val2017/"%saved_coco_path):
        os.makedirs("%scoco/val2017"%saved_coco_path)
    # 取得所有json
    json_list_path = glob.glob(labelme_path + "/*.json")
    # 切分資料集比例
    train_path, val_path = train_test_split(json_list_path, test_size=0.12)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    # 把訓練集轉成coco json
    print("process train dataset...")
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json'%saved_coco_path)
    for file in tqdm(train_path, desc='train pics:'):
        shutil.copy(file.replace("json","jpg"),"%scoco/train2017/"%saved_coco_path)

    # 把驗證集轉成coco json
    print("process val dataset...")
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json'%saved_coco_path)
    for file in tqdm(val_path, desc='val pics:'):
        shutil.copy(file.replace("json","jpg"),"%scoco/val2017/"%saved_coco_path)