import cv2
import numpy as np
import matplotlib.pyplot as plt
from LaneAF.tools.camera_correction import *
from tkinter import *
from PIL import Image, ImageTk
import pandas as pd
# canny 算法
# def canny_edge(img, g_kernel, g_dev, lth, hth, show_img=False, save_img=False):
# 	img_gaussian = cv2.GaussianBlur(img, (g_kernel, g_kernel), g_dev)
# 	img_edge = cv2.Canny(img_gaussian, lth, hth)
# 	if save_img:
# 		cv2.imwrite("./img_edge.jpg", img_edge)
# 	if show_img:
# 		cv2.imshow("img_edge", img_edge)
# 		cv2.waitKey(-1)
# 	return img_edge


def auto_canny(image, sigma=0.1):
	# 計算單通道像素強度的中位數
	v = np.median(image)

	# 選擇合適的lower和upper值，然後應用它們
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	return edged

if __name__=="__main__":
    img0 = cv2.imread("/home/hsuan/distort2.png")#distort.png #00048.jpg
    # img0 = cv2.resize(img0, (833,468), interpolation=cv2.INTER_AREA)
    # img0, cam, distCoeff = camera_correction(img0,-0.0001) #-0.0001
    # cv2.imshow("original img", img0)
    # cv2.waitKey(-1)

    numLine = []
    avgLineLength = []
    avgheight = []
    i_index = []

    for i in np.arange(0, -0.00003, -0.000001):
        img0 = cv2.imread("/home/hsuan/00048.jpg")
        img0, cam, distCoeff = camera_correction(img0,round(i,6)) #-0.0001

        gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        auto = auto_canny(blurred)

        # HoughLinesP函数是概率直线检测，注意区分HoughLines函数
        lines = cv2.HoughLinesP(auto, 1, np.pi/180, 200, minLineLength=50,maxLineGap=10)

        if lines is None:
            lines1, allheight, allLength = 0, 0, 0
            i_index.append(i)
            numLine.append(0)
            avgheight.append(0)
            avgLineLength.append(0)

        else:

            lines1 = lines[:,0,:]

            allLength = 0
            for x1,y1,x2,y2 in lines1:
                # allheight += abs(y2-y1)
                # maxlength = (abs(x2-x1)**2+abs(y2-y1)**2)**0.5 if (abs(x2-x1)**2+abs(y2-y1)**2)**0.5 > maxlength else maxlength

                allLength += (abs(x2-x1)**2+abs(y2-y1)**2)**0.5
                cv2.line(img0,(x1,y1),(x2,y2),(0,255,0),2)
                
            cv2.imwrite("/home/hsuan/canny/"+str(round(abs(i)*100000,1))+".jpg", img0)
            # cv2.imshow("original img", img0)
            # cv2.waitKey(-1)

            i_index.append(i)
            numLine.append(len(lines1))
            avgLineLength.append(allLength/len(lines1))

    df = pd.DataFrame(list(zip(i_index, numLine,avgLineLength)), columns =["i_index","numLine","avgLineLength"])
    print(df)

    plt.style.use("ggplot") 
    plt.plot(df["i_index"], df["avgLineLength"],c = "r") 
    # plt.plot(df["i_index"], df["numLine"],c = "g")
    plt.plot(i_index[avgLineLength.index(max(avgLineLength))],max(avgLineLength), 'ro')
    plt.annotate("({},{})".format(i_index[avgLineLength.index(max(avgLineLength))],round(max(avgLineLength),0)), xy=(i_index[avgLineLength.index(max(avgLineLength))],max(avgLineLength)), xytext=(15, 0), textcoords='offset points')

    plt.title("Calibration Coefficient k Estimation", fontsize = 15, fontweight = "bold", y = 1) 
    plt.xlabel("Calibration coefficient k", fontweight = "bold")  
    plt.ylabel("Average Length of Line", fontweight = "bold")  
    plt.xlim(max(df["i_index"]),min(df["i_index"]))

    plt.show()
    # cv2.imshow("original img", img0)
    # cv2.waitKey(-1)

    # root = Tk()
    # root.title('oxxo.studio')
    # root.geometry('1000x800')
    # a = StringVar()   # 定義文字變數
    # a.set('0,0')

    # cv2img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGBA)
    # current_image = Image.fromarray(cv2img0)
    # imgtk = ImageTk.PhotoImage(image=current_image)

    # label = Label(root, image=imgtk, textvariable=a)
    # label.pack()

    # def correction_config(img0):
    #     img1, cam, distCoeff = camera_correction(img0)
    #     cv2img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGBA)
    #     curimg1 = Image.fromarray(cv2img1)
    #     imgtk1 = ImageTk.PhotoImage(image=curimg1)
    #     return imgtk1

    # def show(e):
    #     a.set(f'{scale_h.get()}')
    #     imgtk1 = correction_config(img0)
    #     label.configure(image=imgtk1) 


    # scale_h = Scale(root, from_=-1, to=0, orient='horizontal',resolution=0.01, command=show)  # 改變時執行 show
    # scale_h.pack()

    # root.mainloop()



    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ret, output1 = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    # img_edge = canny_edge(output1, 11, 0.0, 100, 180, show_img=False)
    # cv2.imwrite("/home/hsuan/canny3.jpg", output1)
	
    # cv2.imshow("img_edge", output1)
    # cv2.waitKey(-1)
