import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir
from stuffs.label_convert import converter
import os
import glob 

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

def corner_to_box(car):

    (x1,y1,x2,y2)=car

    x1=float(x1)
    y1=float(y1)
    x2=float(x2)
    y2=float(y2)
    box=[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]
    box=box
    
    return box
    

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

def shift_box_iou(box1,box2):
    
    box1[0],box1[1],box1[2],box1[3]=0,0,abs(box1[2]-box1[0]),abs(box1[3]-box1[1])
    box2[0],box2[1],box2[2],box2[3]=0,0,abs(box2[2]-box2[0]),abs(box2[3]-box2[1])

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

    

def make_str(num):
    if len(str(num))==1:
        num2='00'+str(num)
    elif len(str(num))==2:
        num2='0'+str(num)
    else:
        num2=str(num)
        
    return num2

def dist_car(car1,car2):
    
    (x1, y1, w1, h1) = car1
    (x2, y2, w2, h2) = car2
    x1=float(x1)
    y1=float(y1)
    x2=float(x2)
    y2=float(y2)
    
    dist=np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    
    return dist

col_jetaa=[[204  , 0,   0],
 [  0   ,0 ,255],
 [255 ,255 ,  0],
 [0 ,255,   0],
 [ 0 ,255 ,0],
 [204 ,255,  51],
 [0 , 255  , 255],
 [  0 ,204 ,255],
 [0  , 0 ,  255],
 [  125 ,155 ,0],
 [ 255 ,153 ,  0],
 [0 ,0 ,  155],
 [102 ,0 ,153],
 [  0 ,153 ,255],
 [  0 ,102 ,255],
 [255 ,153 ,  0],
 [  0 ,  0 ,204],
 [  0 ,  0 ,255],
 [255 ,  0,   0],
 [ 51 ,255 ,204],
 [204 ,255,  51],
 [0 , 255  , 255],
 [  0 ,204 ,255],
 [153  , 0 ,  0],
  [255 ,  0,   0]]


priLblDir='./results/labels/'
lblDir='./results/converted_labels/'

oldFiles=glob.glob('./results/converted_labels/*')
for f in oldFiles:
    os.remove(f)

imgDir='./dataset/test/images/'
imgs=sorted(listdir(imgDir))
prilbls=sorted(listdir(priLblDir))
converter(imgDir,priLblDir,imgs,prilbls)
lbls=sorted(listdir(lblDir))

size=1024

max_car=400
fp_count=0

G_vec=[]
G_mat=np.zeros((max_car,4))
max_num=0

init_File=lblDir+lbls[0]



init_Cars=open_label(init_File)
for i in range(len(init_Cars)-1):
    initBox=get_box(init_Cars[i].split(' ')[1:5])
    G_mat[i,:]=initBox

temp_len=len(init_Cars)-1
    
img_array=[]
blank=np.zeros((size,size,3),np.uint8)
lost_car=np.zeros((max_car,6))
cnt=0
for num in range (1,len(imgs)):
    
    prev_Cars=[]
    prev_Cars=G_mat[0:temp_len,:]

    current_File=lblDir+lbls[num]

    
    current_Cars=open_label(current_File)
    
    img=cv2.imread(imgDir+imgs[num])
    
    img=cv2.resize(img,((int(size),int(size))))

    pair_mat=np.zeros((len(current_Cars)-1,temp_len))
    
    for i in range(len(current_Cars)-1):
        currentBox=get_box(current_Cars[i].split(' ')[1:5])
        allIOU=[]
        
        for j in range(temp_len):
            predBox=prev_Cars[j,:]
            iou=calc_iou(currentBox,predBox)
            allIOU.append(iou)
            pair_mat[i,j]=round(iou,3)

    pairs=[]
    
    for i in range(len(current_Cars)-1):
        pos_r=np.argmax(pair_mat[i,:])
        for j in range(len(prev_Cars)):
            if j!=pos_r:
                pair_mat[i,j]=0
        
        
    
    
    for i in range(len(prev_Cars)):
        pos=np.argmax(pair_mat[:,i])
        if np.max(pair_mat[:,i])>0:
            pairs.append([i,pos])
    pairs_array=np.array(pairs)
    
    for i in range(len(pairs_array)):

        
        p_car=prev_Cars[pairs_array[i,0]]
        c_car=current_Cars[pairs_array[i,1]].split(' ')

        px=int((float(p_car[0])*size+float(p_car[2])*size)/2)
        py=int((float(p_car[1])*size+float(p_car[3])*size)/2)

        cx=int(float(c_car[1])*size)
        cy=int(float(c_car[2])*size)
        cw=int(float(c_car[3])*size)
        ch=int(float(c_car[4])*size)
        cx1,cy1,cx2,cy2=cx-cw/2,cy-ch/2,cx+cw/2,cy+cw/2
        

        #img=cv2.rectangle(img,(int(cx2),int(cy2)),(int(cx1),int(cy1)),col_jetaa[pairs_array[i,0]%len(col_jetaa)],2)
        img=cv2.putText(img,str(pairs_array[i,0]),(int(cx),int(cy)),cv2.FONT_HERSHEY_SIMPLEX,1, col_jetaa[pairs_array[i,0]%len(col_jetaa)], 3)
        blank=cv2.line(blank,(int(px),int(py)),(int(cx),int(cy)),col_jetaa[pairs_array[i,0]%len(col_jetaa)],3)
        img=cv2.add(img,blank)
        
    
    img=cv2.putText(img,str(num),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,255), 3)
    nemo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array.append(img)

    
    FP_mat=[]
    for i in range (len(current_Cars)-1):
        a= i in (pairs_array[:,1])
        if a==False:
            FP_mat.append(i)
                    
    
    for i in range (len(prev_Cars)):
        ckk=i in (pairs_array[:,0])
        if (prev_Cars[i,0]!=0) & (ckk==False):
            lost_car[i][0:4]=prev_Cars[i]
            lost_car[i][4]=i
            lost_car[i][5]=num
            cnt=cnt+1

    G_mat=np.zeros((max_car,4))
    for i in range(len(pairs_array)): 
        currentBox=get_box(current_Cars[pairs_array[i,1]].split(' ')[1:5])
        G_mat[pairs_array[i,0],:]=currentBox
    
    
    max_num_c=np.max(pairs_array[:,0])
    if max_num_c>max_num:
        max_num=max_num_c
        
    found_car=0
    if len(FP_mat)>0:
        
        for k in range (len(FP_mat)):
            flagg=0
            for los in range(len(lost_car)):
                if (num-lost_car[los][5])<4:
                    if dist_car(current_Cars[FP_mat[k]].split(' ')[1:5],corner_to_box(lost_car[los][0:4]))<0.05:
                        G_mat[int(lost_car[los][4]),:]=get_box(current_Cars[FP_mat[k]].split(' ')[1:5])
                        flagg=1
                        px,py=corner_to_box(lost_car[los][0:4])[0]*size,corner_to_box(lost_car[los][0:4])[1]*size
                        cx,cy=float(current_Cars[FP_mat[k]].split(' ')[1])*size,float(current_Cars[FP_mat[k]].split(' ')[2])*size
                        
                        blank=cv2.line(blank,(int(px),int(py)),(int(cx),int(cy)),col_jetaa[int(lost_car[los][4])%len(col_jetaa)],3)
                        img=cv2.add(img,blank)
                        found_car=found_car+1
            
            if flagg==0:
                G_mat[max_num+1+k,:]=get_box(current_Cars[FP_mat[k]].split(' ')[1:5])
                    

    fp_count=fp_count+len(FP_mat)
    
    temp_len=max_num+1+len(FP_mat)-found_car
    

    
    
height,width,layers = img.shape
sizes = (width,height)
out = cv2.VideoWriter('./results/video/tracked.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 3, sizes)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()