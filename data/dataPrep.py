import os
import os
from PIL import Image 
import argparse
from tqdm import tqdm


def create_txt_file(select,train):

                #Define directories
    #=====================================================================================#
    #curdir=os.getcwd()
    if train==True:
        dayImg='./dataset/'+select+'/day/images/'
        dayLbs='./dataset/'+select+'/day/labels/'
        nightImg='./dataset/'+select+'/night/images/'
        nightLbs='./dataset/'+select+'/night/labels/'

        imgDirs=[dayImg,nightImg]
        lblDirs=[dayLbs,nightLbs]

        fileName=select+'_data.txt'
    else:
        testImg='./dataset/'+select+'/images/'
        testLbs='./dataset/'+select+'/labels/'

        imgDirs=[testImg]
        lblDirs=[testLbs]

        fileName=select+'_data.txt'

    #=====================================================================================#

    with open(fileName, 'w') as f:
        for xx in range(len(imgDirs)):
            labeldir=lblDirs[xx]
            imgdir=imgDirs[xx]
            labellist=sorted(os.listdir(labeldir))
            for item in tqdm(labellist):
                imgloc=imgdir+'/'+item[:-3]+'jpg'
                labelloc=labeldir+'/'+item
                f.write("%s " % imgloc)
                im=Image.open(imgloc)
                h,w=im.size
                with open(labelloc, 'r') as file:
                    boxes = file.readlines()
                    for box in boxes:
                        box=box.strip()
                        #print(box)
                        box=box.split(' ')
                        xcen=float(box[1])*w
                        ycen=float(box[2])*h
                        bw=float(box[3])*w
                        bh=float(box[4])*h
                        xmin=str(int(xcen-bw/2))
                        ymin=str(int(ycen-bh/2))
                        xmax=str(int(xcen+bw/2))
                        ymax=str(int(ycen+bh/2))
                        box=[xmin,ymin,xmax,ymax,box[0]]
                        box=','.join(box)
                        f.write(" %s " % box)
                    f.write("\n")
