import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
from functools import partial
import cv2
from scipy.optimize import curve_fit
#import matplotlib.pyplot as plt
import pycpd as pypd

import os
import matplotlib.pyplot as plt
import time

#PCD algorithm visualization
def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0], 512-X[:, 1], color='red', label='Target',s=5)
    ax.scatter(Y[:, 0], 512-Y[:, 1], color='blue', label='Source',s=5)

    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
             fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

if __name__ == "__main__":
    time1 = time.time()
    #load the segmentation module (cocoCascade.py and cascademaskrcnn.pth).
    config_file = './cocoCascader100.py'
    checkpoint_file = './cascademaskrcnn.pth'
    model = init_detector(config_file, checkpoint_file)  # or device='cuda:0'

    #The file includes the axial slices of PE on the nearest distance to the DDP.
    path_photox = './result/0.jpg'  # 所有photo所在的文件夹目录

    imgX = mmcv.imread(path_photox)
    resultX = inference_detector(model, imgX)
    tempX = np.zeros((512, 512))
    for i in range(len(resultX[1])):
        for j in range(len(resultX[1][i])):
            tempX = resultX[1][i][j] + tempX

    #Threshold segmentation was used to extract the contour of anterior chest.
    colorX = np.zeros((512, 512, 3))
    colorX = np.array(colorX, dtype='uint8')
    colorX[:, :, 0] = tempX[:, :]
    colorX[:, :, 1] = tempX[:, :]
    colorX[:, :, 2] = tempX[:, :]
    gray = cv2.cvtColor(colorX, cv2.COLOR_BGR2GRAY)
    cv2.ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgX, contours, -1, (0, 0, 255), 3)
    cv2.imshow("result", imgX)
    # cv2.waitKey(0)

    posX1 = []
    posX2 = []
    for i in range(len(contours)):
        for j in range(contours[i].shape[0]):
            posX1.append(contours[i][j, 0, 0])
            posX2.append(contours[i][j, 0, 1])

    X = np.zeros((len(posX1), 2))
    X[:, 0] = posX1[:]
    X[:, 1] = posX2[:]

    #The file includes the axial slices of the healthy thoracic cage.
    path_photo = './test1/'  
    files_list = os.listdir(path_photo)
    q1 = 10000
    dff = 1
    for a in range(len(files_list)):
        imgY = mmcv.imread(path_photo + files_list[a])
        resultY = inference_detector(model, imgY)
        tempY = np.zeros((512, 512))
        for i in range(len(resultY[1])):
            for j in range(len(resultY[1][i])):
                tempY = resultY[1][i][j] + tempY
        colorY = np.zeros((512, 512, 3))
        colorY = np.array(colorY, dtype='uint8')
        colorY[:, :, 0] = tempY[:, :]
        colorY[:, :, 1] = tempY[:, :]
        colorY[:, :, 2] = tempY[:, :]
        grayY = cv2.cvtColor(colorY, cv2.COLOR_BGR2GRAY)
        cv2.ret, binaryY = cv2.threshold(grayY, 0, 255, cv2.THRESH_BINARY)

        contoursY, hierarchy = cv2.findContours(binaryY, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgY, contoursY, -1, (0, 0, 255), 3)
        cv2.imshow("result", imgY)
        # cv2.waitKey(0)

        posY1 = []
        posY2 = []
        for i in range(len(contoursY)):
            for j in range(contoursY[i].shape[0]):
                posY1.append(contoursY[i][j, 0, 0])
                posY2.append(contoursY[i][j, 0, 1])
        Y = np.zeros((len(posY1), 2))
        Y[:, 0] = posY1[:]
        Y[:, 1] = posY2[:]
        q1 = 0
        name = path_photo + files_list[a]
        reg = pypd.RigidRegistration(**{'X': Y, 'Y': X})
        # fig = plt.figure()
        # fig.add_axes([0, 0, 1, 1])
        # callback = partial(visualize, ax=fig.axes[0])
        # reg.register(callback)
        reg.register()
        if (q1 > reg.q):
            q1 = reg.q
            name = files_list[a]
            Ytemp = Y
            Y = Ytemp


        reg = pypd.RigidRegistration(**{'X': Y, 'Y': X})

        reg.register()
        data, (s, r, t) = reg.register()

        print('s:', s)
        print('R:\n', r)
        print('t:\n', t)

        xmin = np.argmin(Y, axis=0)
        xmax = np.argmax(Y, axis=0)
        if (Y[xmin[0], 1] < Y[xmax[0], 1]):
            ymin = Y[xmax[0], 1]
        else:
            ymin = Y[xmin[0], 1]

        print(xmax, '\n')
        print(xmin, '\n')
        extractX1 = []
        extractY1 = []
        for i in range(Y.shape[0]):
            if (Y[i, 1] < ymin):
                extractX1.append(Y[i, 0])
                extractY1.append(Y[i, 1])

        Y1 = np.zeros((len(extractX1), 2))
        for i in range(len(extractX1)):
            Y1[i, 0] = (extractX1[i] - t[0]) / s
            Y1[i, 1] = (extractY1[i] - t[1]) / s

            # Y1[i,0] = Y[i,0] / s - t[0]
            # Y1[i, 1] = Y[i, 1] / s - t[1]
        Y2 = Y1.T
        Ypre1 = np.linalg.solve(r.T, Y2)
        Ypre = Ypre1.T
        extractX = []
        extractY = []
        xmin = np.argmin(X, axis=0)
        xmax = np.argmax(X, axis=0)
        for i in range(len(extractX1)):
            if (X[xmin[0], 0] < Ypre[i, 0]):
                if (X[xmax[0], 0] > Ypre[i, 0]):
                    extractX.append(Ypre[i, 0])
                    extractY.append(Ypre[i, 1])

        # reg = pypd.AffineRegistration(**{'X':Y,'Y':X})
        # data,(s,r,t) = reg.register()
        '''
        xmin = np.argmin(X, axis=0)
        xmax = np.argmax(X, axis=0)
        if (X[xmin[0], 1] < X[xmax[0], 1]):
            ymin = X[xmax[0], 1]
        else:
            ymin = X[xmin[0], 1]

        print(xmax, '\n')
        print(xmin, '\n')
        extractX = []
        extractY = []
        for i in range(Ypre.shape[0]):
            if (Ypre[i, 1] < ymin):
                extractX.append(Ypre[i, 0])
                extractY.append(Ypre[i, 1])
        '''

        '''
        # resultData = np.zeros()

        fig = plt.figure()
        plt.scatter(extractX[:], extractY[:])
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
        # fitEllipse = []
        # for i in range(len(extractX)):
        #    fitEllipse.append((extractX[i],extractY[i]))
        fitEllipse = np.zeros((len(extractX),2),dtype='int')
        fitEllipse[:,0] = extractX[:]
        fitEllipse[:,1] = extractY[:]
        fitdata = np.zeros((512,512,3),dtype='uint8')
        for i in range(len(extractX)):
            fitdata[int(extractX[i]), int(extractY[i]),0] = 255
            fitdata[int(extractX[i]), int(extractY[i]),1] = 0
            fitdata[int(extractX[i]), int(extractY[i]),2] = 0

        #ell = cv2.fitEllipse(fitEllipse)
        ell = cv2.minEnclosingCircle(fitEllipse)
        cv2.ellipse(fitdata,ell,(0,0,255),2)
        cv2.imshow("fit",fitdata)
        cv2.waitKey(0)
        with open('Ytest.txt', 'w') as f:
            for i in range(len(extractX)):
                f.write(str(extractX[i]) + '\t' + str(extractY[i]) + '\n')
        with open('datatest.txt', 'w') as f1:
            for i in range(data.shape[0]):
                f1.write(str(data[i,0]) + '\t' + str(data[i,1]) + '\n')
        '''
        points = []
        ex = []
        ey = []
        sort1 = np.argsort(extractX)
        for i in range(len(sort1)):
            ex.append(extractX[sort1[i]])
            ey.append(extractY[sort1[i]])
        for i in range(len(extractY)):
            points.append((int(ex[i]), int(ey[i])))

        p = tuple(points)
        degree = 3
        curve = fitting.approximate_curve(p, degree, ctrlpts_size=6)

        # Prepare points
        evalpts = np.array(curve.evalpts)
        pts = np.array(p)
        time2 = time.time()
        print(time2 - time1)

        # Plot points together on the same graph
        fig = plt.figure()
        plt.plot(evalpts[:, 0], 512 - evalpts[:, 1])

        plt.scatter(pts[:, 0], 512 - pts[:, 1], color="red", s=5)

        plt.scatter(X[:, 0], 512 - X[:, 1], color="blue", s=5)

        # plt.scatter(Y[:, 0], Y[:, 1])
        # plt.scatter(data[:, 0], data[:, 1])
        # f = plt.gcf() 
        # plt.imsave('./nonimg3/' + files_listx[aa][:-4] + 'u' + files_list[a][:-4] + 'q' + str(reg.q) + '.png')
        # f.savefig('./nonimg2/' + files_listx[aa][:-4] + 'u' + files_list[a][:-4] + 'q' + str(reg.q) + '.png')
        # f.clear() 

