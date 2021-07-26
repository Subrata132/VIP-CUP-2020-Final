import numpy as np
import cv2
import matplotlib.pyplot as plt

def open_label(filename):

    file=open(filename)
    data=file.read()
    file.close()
    cars=data.split('\n')

    return cars

def label_convert(car,img_shape):

    W,H,_=img_shape
    (x1, y1, w, h) = car.split(' ')[2:6]
    x1=float(x1)/W
    y1=float(y1)/H
    w=float(w)/W
    h=float(h)/H
    box=[0, x1+w/2,y1+h/2,w,h]
    
    return box

def converter(imgDir,lblDir,imgs,lbls):
    
    for num in range(len(imgs)):
        init_File=lblDir+lbls[num]
        img=cv2.imread(imgDir+imgs[num],1)

        box=[]
        init_Cars=open_label(init_File)
        for i in range(len(init_Cars)-1):
            initBox=label_convert(init_Cars[i],img.shape)
            box.append(initBox)
            
        np.savetxt('./results/converted_labels/'+lbls[num], box,fmt='%1.4f')