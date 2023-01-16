import mmcv
import argparse
from mmcls.datasets import build_dataset
from mmcls.core.evaluation import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='calculate precision and recall for each class')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    pred = mmcv.load("./result.pkl")['pred_label']
    matrix = confusion_matrix(pred, dataset.get_gt_labels())
    print('confusion_matrix:', matrix)
    print(dataset.get_gt_labels())
    pred = mmcv.load("./result.pkl")['class_scores']

    #AP
    ap = precision(pred, dataset.get_gt_labels(),thrs=0.5)
    print("AP=0.5:",ap)
    ap = precision(pred, dataset.get_gt_labels(),thrs=0.75)
    print("AP=0.75:",ap)
    ap = precision(pred, dataset.get_gt_labels(),thrs=0.)
    print("AP=0.:",ap)
    #AR
    ar = recall(pred, dataset.get_gt_labels(),thrs=0.)
    print("AR=0.:",ar)
    #F1
    f1 = f1_score(pred, dataset.get_gt_labels(),thrs=0.)
    print("f1=0.:",f1)
    #confusion matix
    matrix = calculate_confusion_matrix(pred, dataset.get_gt_labels())
    name = ["0","single_white_solid","single_white_dotted","single_yellow_solid","single_yellow_dotted","double_white_solid","double_yellow_solid","double_yellow_dotted","double_white_solid_dotted","double_white_dotted_solid"]


    fig = plt.figure(figsize=(9, 9),dpi=81)
    ax1 = fig.add_subplot(1, 1, 1) 

    for (j,i),label in np.ndenumerate(matrix[1:10, 1:10]):
        print(j,i)
        ax1.text(i,j,label,color='black', ha='center',va='center')

    im = ax1.imshow(matrix[1:10, 1:10], cmap=plt.cm.Blues)
    ax1.set_yticklabels(name,fontsize=10)
    ax1.set_xticklabels(name,fontsize=10)
    plt.xticks(rotation=90)
    ax1.set_title('The Confusion Matrix of Lane Classification Model')
    fig.colorbar(im)
    plt.show()
if __name__ == '__main__':
    main()