from tools.perspective_correction import *
import numpy as np
import cv2
import os

dataset = 5
piclist = [_ for _ in os.listdir(os.path.join('/media/hsuan/data/LaneDataset/{}').format(str(dataset))) if _.endswith(".jpg")]
print(piclist)

for pic in piclist:
    imglist = []
    img = cv2.imread(os.path.join('/media/hsuan/data/LaneDataset/{}/{}'.format(str(dataset),pic)))
    imglist.append(img)
    print("piclen:",len(piclist),"pic:",piclist.index(pic)+1)
    for i in range(-50,0,25):
        auglist_0 = augmentation(1,imglist,thed= i)
        auglist_1 = deepcopy(auglist_0)

        for img in auglist_0:
            if i != 0 :
                cv2.imwrite(os.path.join('/media/hsuan/data/LaneDataset/{}/{}'.format(str(dataset),str((pic).replace('.jpg',''))+'_'+str(i)+'.jpg')),img)
                # cv2.imshow("img_out", img)
                # cv2.waitKey(0)

            #Laplace
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
            img_Laplace = cv2.filter2D(img, -1, kernel=kernel)
            # cv2.imwrite(os.path.join('/media/hsuan/data/LaneDataset/{}/{}'.format(str(dataset),str((pic).replace('.jpg',''))+'_'+str(i)+'_'+'L'+'.jpg')),img_Laplace)
            auglist_1.append(img_Laplace)
            # CLAHE
            imgYUV = cv2.cvtColor(img_Laplace, cv2.COLOR_BGR2YCrCb)
            channelsYUV = cv2.split(imgYUV)
            t = channelsYUV[0]
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
            p= clahe.apply(t)
            channels = cv2.merge([p,channelsYUV[1],channelsYUV[2]])
            img_CLAHE = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
            cv2.imwrite(os.path.join('/media/hsuan/data/LaneDataset/{}/{}'.format(str(dataset),str((pic).replace('.jpg',''))+'_'+str(i)+'_'+'C'+'.jpg')),img_CLAHE)
            auglist_1.append(img_CLAHE)

        # auglist_2 = augmentation(2,auglist_1,thed=7)
        # for idx, img in enumerate(auglist_2):
        #     cv2.imwrite(os.path.join('/media/hsuan/data/LaneDataset/{}/{}'.format(str(dataset),str((pic).replace('.jpg',''))+'_'+str(i)+'_7_'+str(idx)+'.jpg')),img)
            # cv2.imshow("img_out", img)
            # cv2.waitKey(0)


        # input()

