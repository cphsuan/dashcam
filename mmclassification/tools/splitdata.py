import os
from shutil import copy, rmtree
import random
from tqdm import tqdm
import numpy as np

def makedir(path):
    if os.path.exists(path):
        rmtree(path)
    os.makedirs(path)

def splitdataset(init_dataset, new_dataset,datasets, split_rate):
    random.seed(0)
    classes_name = [name for name in os.listdir(init_dataset)]
    makedir(new_dataset)
    for i in datasets:
        dir = os.path.join(new_dataset, i)
        makedir(dir)
        for cla in classes_name:
            makedir(os.path.join(dir, cla))

    meta_set = os.path.join(new_dataset, "meta")
    makedir(meta_set)
    
    for cla in classes_name:
        class_path = os.path.join(init_dataset, cla)

        img_set = os.listdir(class_path)
        num = len(img_set)
        train,test = np.split(img_set, [int((1-split_rate) * len(img_set))])
        # train, val, test = np.split(img_set, [int((1-split_rate*2) * len(img_set)), int((1-split_rate) * len(img_set))])

        pbar = tqdm(total=num,desc=f'Class : ' + cla, mininterval=0.3)
        for img in img_set:
            pbar.update(1)
            if img in train:
                init_img = os.path.join(class_path, img)
                new_img = os.path.join(new_dataset, "train", cla)
                copy(init_img, new_img)
            else:
                init_img = os.path.join(class_path, img)
                new_img = os.path.join(new_dataset, "test", cla)
                copy(init_img, new_img)
        
        pbar.close()


def get_info(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    names = []
    indexs = []
    for data in class_names:
        name,index = data.split(' ')
        names.append(name)
        indexs.append(int(index))
        
    return names,indexs

def get_annotation(new_dataset_path ,label, datasets):

    meta_dir = os.path.join(new_dataset_path, "meta")
    classes, indexs = get_info(label)

    for dataset in datasets:
        txt_file = open(meta_dir+ '/'+ dataset + '.txt', 'w')
        print(txt_file)
        datasets_path_ = os.path.join(new_dataset_path, dataset)
        classes_name   = os.listdir(datasets_path_)

        for name in classes_name:
            print(name)
            print(classes)
            if name not in classes:
                continue
            cls_id = indexs[classes.index(name)]

            images_path = os.path.join(datasets_path_, name)
            images_name = os.listdir(images_path)
            for photo_name in images_name:
                _, postfix = os.path.splitext(photo_name)
                if postfix not in ['.jpg', '.png', '.jpeg','.JPG', '.PNG', '.JPEG']:
                    continue
                txt_file.write('%s'%(os.path.join(name, photo_name)) + ' ' + str(cls_id))
                txt_file.write('\n')
        txt_file.close()

if __name__ == '__main__':
    init_dataset_path = '/media/hsuan/data/LaneDataset/'
    new_dataset_path = '/home/hsuan/Thesis/mmclassification/data'
    label = '/media/hsuan/data/label.txt'
    split_rate = 0.2
    datasets = ["train", "val", "test"]
    
    print("=====Start to Split Datasets======")
    splitdataset(init_dataset_path, new_dataset_path ,datasets, split_rate)
    print("=====Start to Generate Annotations======")
    get_annotation(new_dataset_path, label, datasets)
    print("=====Finished!======")
