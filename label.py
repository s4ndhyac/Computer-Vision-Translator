from PIL import Image, ImageOps
from os.path import join
import os
from shutil import move
from glob import glob
import numpy as np
from processing import square_pad
import cv2

def moveImg(fromDir,toDir,delete=False):
    dirs = glob(fromDir+"/*")
    for dir in dirs:
        destination = dir.split('/')[-2]
        if not os.path.exists(destination):
            os.makedirs(destination)
        #print("moving direcotry %s to %s" % (dir,destination))
        #os.rename(dir, destination)
        for filename in os.listdir(dir):
            move(join(dir,filename),join(destination,filename))
        if delete:
            os.rmdir(dir)

def preprocessImg(dir,size=300):
    """
    Load all files from a directory 
    Rescale them to size x size (as per argument)
    If mask set to true, will remove transparency
    Turn to gray
    and append them in a 4d array
    
    Return 4d array of images and vector of labels
    """
    nbImage = len(glob(dir+"/*.jpeg"))
    dirname = dir.split('/')[-2]
    labels = np.array([dirname for _ in range(nbImage)])
    images = np.empty((nbImage,size,size,3))
    
    for i,infile in enumerate(glob(dir+"/*.jpeg")):
        #file, ext = os.path.splitext(infile)
        #print(infile)
        with Image.open(infile).convert('RGB') as image:
            #img = image.copy()
            img = np.array(image)
            img = img[:, :, ::-1].copy()  

        dim = None
        (h, w) = img.shape[:2]
        # calculate the ratio of the width and construct the
        # dimensions
        r = size / float(w)
        dim = (size, int(h * r))

        # resize the image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        images[i,:,:,:] = resized
    
    return images, labels