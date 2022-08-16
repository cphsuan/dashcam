# import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import pyplot as plt
frog_bgr = cv2.imread('/media/hsuan/data/VIL100/JPEGImages/0_Road029_Trim001_frames/00054.jpg')
# cv2.imshow("img_out", frog_bgr)
# cv2.waitKey(0)
plt.imshow(frog_bgr)
plt.show()
print("continue")
frog_rgb = cv2.cvtColor(frog_bgr, cv2.COLOR_BGR2RGB)
rectangle = (0,  360, 960, 120)
mask = np.zeros(frog_rgb.shape[:2], dtype=np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
# 执行GrabCut
cv2.grabCut(frog_rgb, 
            mask, # 掩模
            rectangle, # 矩形
            bgdModel,  # 背景的临时数组
            fgdModel,  # 前景的临时数组
            50,  # 迭代次数
            cv2.GC_INIT_WITH_RECT) # 使用定义的矩形初始化
# 执行完GrabCut，mask已然发生了变化，感兴趣的小伙伴可以自行打印查看
# 创建掩模，将背景部分设为0，前景部分设为1
mask_2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# 除去背景
frog_rgb_nobg = frog_rgb*mask_2[:,:,np.newaxis]
# 显示图像
# cv2.imshow("img_out", frog_rgb_nobg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(frog_rgb_nobg)
plt.show()
