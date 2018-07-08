# -*- coding: utf-8 -*-
"""
Created on Sat May 12 13:59:37 2018

@author: josher
"""

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2
import glob
import os 

ia.seed(1)
DirectoryToAugment='/Images/'
files=glob.glob(DirectoryToAugment+'*.png')

#Prepare numpy array
images=np.zeros([len(files),512,512,3],dtype=np.uint8)
#Load images into numpy array
for i in range(len(files)):
    images[i,:,:,:]=cv2.imread(files[i])

#Set up augmentor
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips

    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale=(0.5, 1.5),
        translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
        rotate=(-30, 30)
    )
], random_order=True) # apply augmenters in random order
Augmented_Directory=DirectoryToAugment[:-1]+'_Out'
os.mkdir(Augmented_Directory)
#Run on all files in directory multiple times
for i in range(20):
    images_aug = seq.augment_images(images)
    for j in range(len(files)):
        cv2.imwrite(files[j].replace(Augmented_Directory,Augmented_Directory)[:-4]+'_'+str(i)+'.png',images_aug[j,:,:,:])
