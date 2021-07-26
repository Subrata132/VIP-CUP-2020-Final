import cv2
import numpy as np
import glob
from os import listdir
import re
from tqdm import tqdm

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def create_video(imgDir,lblDir):

    imgs=sorted_alphanumeric(listdir(imgDir))
    lbls=sorted_alphanumeric(listdir(lblDir))


    img_array=[]
    for j in range(len(imgs)):

            
        img=cv2.imread('./dataset/test/images/'+imgs[j],1)
        file=open(lblDir+lbls[j])
        data=file.read()
        file.close()
        cars=data.split('\n')
        size=1920
        
        
        for i in range(len(cars)-1):
            car=cars[i]
            car=car.split(' ')
            x=int(car[2])
            y=int(car[3])
            w=int(car[4])
            h=int(car[5])
            x1,y1,x2,y2=x,y,x+w,y+h
            img=cv2.rectangle(img,(x2,y2),(x1,y1),(0,0,255),2)
            img=cv2.putText(img,str("{:.3f}".format(float(car[1]))),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,255,0), 2)
            #img=cv2.resize(img,((int(size),int(size))))
            img=cv2.putText(img,imgs[j],(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,255,0), 2)
        
        img=cv2.resize(img,((int(size),int(size))))
        nemo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width, layers = img.shape
        
        img_array.append(img)

    size = (1920,1920)
    out = cv2.VideoWriter('./results/video/video_output.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()