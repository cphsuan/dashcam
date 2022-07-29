import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser('jpg to avi in python...')
parser.add_argument('--dataset-dir', type=str, default='/media/hsuan/data/VIL100/JPEGImages', help='path to dataset')
parser.add_argument('--output-dir', type=str, default='/media/hsuan/data/VIL100/videos', help='path to output_dataset')
print("file Function: convert .jpeg to H.264 video files")

args = parser.parse_args()

fps = 10
size = (1920, 1080) #需要轉為視訊的圖片的尺寸

def jpg2avi(filename):
    path = args.dataset_dir + '/' + filename + '/'
    filelist = os.listdir(path)
    filelist = sorted(os.listdir(path))
    video = cv2.VideoWriter(args.output_dir + '/' + filename +'.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
        
    for item in filelist:
        if item.endswith('.jpg'):
            item = path + item
            img = cv2.imread(item)
            img=cv2.resize(img,size)
            video.write(img)

    video.release()


if __name__ == '__main__':
    i = 0
    for filename in sorted(os.listdir(args.dataset_dir)):
        i +=1
        print('Done with video {} out of {}...'.format(i, len(os.listdir(args.dataset_dir))))
        jpg2avi(filename)
        
    print('finished!')
    cv2.destroyAllWindows()