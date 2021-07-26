import numpy as np
from os import listdir 
import re
from numpy import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import tensorflow as tf
import tensorflow.keras.backend as K
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import skimage.measure
import time
import os
import glob
import argparse
from tqdm import tqdm

from stuffs.make_video import create_video

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
        try:
            for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
                print(e)

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def open_label(filename):
    file=open(filename)
    data=file.read()
    file.close()
    cars=data.split('\n')

    return cars

def get_box(car):
    
    (x, y, w, h) = car
    x=float(x)
    y=float(y)
    w=float(w)
    h=float(h)
    box=[x-w/2,y-h/2,x+w/2,y+h/2]
    box=box
    
    return box

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
    w2, h2 = box2[2]-box2[0], box2[3]-box2[1]
    
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3  
        
def do_nms(boxes,th):
    boxes=np.array(boxes)
    sort_list=(np.argsort(boxes[...,4]))
    for i in range(len(sort_list)):
        index_i=sort_list[i]

        if boxes[index_i,5]==0:
            continue
        else:
            for j in range (i+1,len(sort_list)):
                index_j=sort_list[j]
                if bbox_iou(boxes[index_i,0:4], boxes[index_j,0:4]) >= th:
                    boxes[index_i,5]=0

    boxes2=[]
    for i in range (len(boxes)):
        if boxes[i,5]!=0:
            boxes2.append(boxes[i])  
    boxes2=np.array(boxes2)
    return boxes2  

def make_str(num):
    if len(str(num))==1:
        num2='00'+str(num)
    elif len(str(num))==2:
        num2='0'+str(num)    
    else:
        num2=str(num)      
    return num2

def calc_iou(box1, box2):
    
    xi1 = np.max([box1[0], box2[0]])
    yi1 = np.max([box1[1], box2[1]])
    xi2 = np.min([box1[2], box2[2]])
    yi2 = np.min([box1[3],box2[3]])
    inter_width = max((yi2-yi1),0)
    inter_height = max((xi2-xi1),0)
    inter_area = max((yi2-yi1),0) * max((xi2-xi1),0)

    box1_area = (box1[3]-box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3]-box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area/union_area
    return iou   

def dist(midpoint_x,midpoint_y):
    ddd=np.sqrt((midpoint_x-1/2)*(midpoint_x-1/2)+(midpoint_y-1/2)*(midpoint_y-1/2))
    return ddd

def eliminate_box(box):
    box=np.array(box)
    box2=[]
    for i in range(len(box)):
        px=(box[i,0]+box[i,2])/2
        py=(box[i,1]+box[i,3])/2
        w=(box[i,2]-box[i,0])
        h=(box[i,3]-box[i,1])
        if (dist(px,py)<0.4) | (w<0.2) | (h<0.2) :
            if  (py>0.17) | (px<0.7) :    
                if  (py<0.36) | (px>0.094) | (h<0.06) :
                    if  (py<0.81) | (px>0.295) | (h/w<1.2) :
            
                        box2.append(box[i])
        
    box2=np.array(box2)
    return box2



def resize_box(box):
    box=np.array(box)
    box2=[]
    for i in range(len(box)):
        mpx=(box[i,0]+box[i,2])/2
        mpy=(box[i,1]+box[i,3])/2
        w=box[i,2]-box[i,0]
        h=box[i,3]-box[i,1]
        if (dist(mpx,mpy)>0.4) & (mpy<0.08):
            temp=np.zeros((6))
            min_wh=min(w,h)
            temp[0]=mpx-(min_wh*1.1)/2
            temp[1]=mpy-(min_wh*0.8)/2
            
            
            temp[2]=mpx+(min_wh*1.1)/2
            temp[3]=mpy+(min_wh*0.8)/2
            temp[4]=box[i,4]
            temp[5]=box[i,5]
            box2.append(temp)
        else:
            box2.append(box[i])
        
    box2=np.array(box2)
    return box2


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
    
def camname_range(imgs):
    names=[]

    for img in imgs:

        img=img[:-4]
        img=img[::-1]
        xx=''
        num=''
        flag=1
        for j in range(len(img)):
            if img[j].isnumeric():
                num=num+img[j]
                
            else:
                flag=j
                break
        num=num[::-1]
        xx=img[::-1][0:-flag]
        #print(xx)

        names.append((xx,num))
    camnameparts=[ii for ii,numnum in names]
    camnames=list(set(camnameparts))
    camrange=[]
    for camname in camnames:
        start=1e10
        end=-20
        zfill_len=5000
        for name in names:
            cam_name,num=name
            zfill_len=min(zfill_len,len(num))
            num1=int(num)
            if camname==cam_name:
                start=min(start,num1)
                end=max(end,num1)
        camrange.append((start,end+1,zfill_len))
        
    return camnames,camrange

oldFiles=glob.glob('./results/labels/*')
for f in oldFiles:
    os.remove(f)

testImgLoc='./dataset/test/images/'
imgs=sorted_alphanumeric(listdir(testImgLoc))
camnames,camrange=camname_range(imgs)
BOX=3
mot_thold=1
thresh_highs=[0.5,0.0025,0.08]
thresh_lows=[0.1,0.00025,0.005]

size=640
grids=[20,40,80]
mot_tholds=[2,0.01,0.01]
anchors=[[[53,40], [62,77], [116,120]], [[24,37], [33,21], [35,58]], [[12,16], [16,27], [24,14]]]

model=load_model('./saved_model/fishNet.h5')

parser = argparse.ArgumentParser()
parser.add_argument("--extension",type=str,default='.jpg')
args = parser.parse_args()
ext=args.extension
for camnum,cam in enumerate(camnames):
    
    start,end,zfill_len=camrange[camnum]
    for num in tqdm(range(start,end)):
        saved_file_name='./results/labels/'+cam+str(num).zfill(zfill_len)

        previous_frame=testImgLoc+cam+str(max(num-1,start)).zfill(zfill_len)+ext
        current_frame=testImgLoc+cam+str(num).zfill(zfill_len)+ext

        imgloc=current_frame
        img=Image.open(imgloc)
        width,height=img.size
        img=img.resize((size,size))
        image=img
        img=(np.array(img,dtype=np.float32))/255.0

        img = expand_dims(img, 0)
        boxes=[]

        netout_3 = model.predict(img)

        #print("Model decode Time : %s seconds ---" % (time.time() - start_time1))
        start_time2 = time.time()

        ###############################

        c1=cv2.imread(previous_frame)
        c2=cv2.imread(current_frame)



        c1_r=cv2.resize(c1,(size,size))
        c2_r=cv2.resize(c2,(size,size))
        cc=cv2.subtract(c1_r,c2_r)
        gray_image = cv2.cvtColor(cc, cv2.COLOR_BGR2GRAY)

        ###############################

        for ii in range(3):

            grid=grids[ii]
            netout=netout_3[ii]
            netout=netout.reshape((grid,grid,BOX,-1))
            ANCHORS=anchors[ii]

            #print(netout.shape)

            grid_h,grid_w=grid,grid

            objectness = _sigmoid(netout[...,4])
            #objectness.shape

            ########################### (3 tunable param)

            sect=int(size/grid)
            block_mean=skimage.measure.block_reduce(gray_image, (sect,sect), np.mean)

            motion_mask=block_mean>mot_tholds[ii]

            th_mat=[]
            th_mat=thresh_highs[ii]-(thresh_highs[ii]-thresh_lows[ii])*motion_mask

            th_mat_expand = np.repeat(th_mat[..., np.newaxis], 3, axis=-1)

            mask=objectness>=th_mat_expand


            ###########################


            #print(mask.shape)

            GRID_H, GRID_W=grid,grid
            cell_x =(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), ( GRID_H, GRID_W, 1, 1)))
            cell_x=tf.cast(cell_x,tf.float32)
            cell_y = tf.transpose(cell_x, (1,0,2,3))
            cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [ 1, 1, 3, 1])

            pred_xy = ((_sigmoid(netout[..., :2]) + cell_grid)/grid)
            pred_wh = ((tf.exp(netout[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2]))/size)

            box_mins=pred_xy-pred_wh/2
            box_maxes=pred_xy+pred_wh/2


            b0=tf.boolean_mask(tf.reshape(box_mins[..., 0],mask.shape),mask)
            b1=tf.boolean_mask(tf.reshape(box_mins[..., 1],mask.shape),mask)
            b2=tf.boolean_mask(tf.reshape(box_maxes[..., 0],mask.shape),mask)
            b3=tf.boolean_mask(tf.reshape(box_maxes[..., 1],mask.shape),mask)
            b4=tf.boolean_mask(objectness,mask)
            b5=tf.boolean_mask(objectness,mask)

            _boxes =  tf.stack([b0,b1,b2,b3,b4,b5])
            _boxes=tf.transpose(_boxes, (1,0))
            _boxes=_boxes.numpy()

            boxes.append(_boxes)


        box0=np.array(boxes[0])
        box1=np.array(boxes[1])
        box2=np.array(boxes[2])

        BOXES=[]
        BOXES[:len(box0)]=box0
        BOXES[len(box0):len(box1)]=box1
        BOXES[len(box0)+len(box1):]=box2

        BOXES_elim=eliminate_box(BOXES)
        BOXES_=resize_box(BOXES_elim)

        if len(BOXES_)>0:
            
            boxes2=do_nms(BOXES_,0.4)
        else:
            boxes2=[]

        outCars=[]

        for i in range(len(boxes2)):
            outcar=[]
            outcar.append('vehicle')
            outcar.append(round(boxes2[i][4],2))
            outcar.append(int((boxes2[i][0])*width))
            outcar.append(int((boxes2[i][1])*height))
            outcar.append(int((boxes2[i][2]-boxes2[i][0])*width))
            outcar.append(int((boxes2[i][3]-boxes2[i][1])*height))    
            outCars.append(outcar)

        fileName=  saved_file_name+'.txt'
        with open(fileName,'w') as f:
            for l in range(len(outCars)):
                s=' '
                s=s.join([str(elem) for elem in outCars[l]])
                s=s+'\n'
                f.write(s)
            f.close()


imgDir='./dataset/test/images/'
lblDir='./results/labels/'
create_video(imgDir,lblDir)
