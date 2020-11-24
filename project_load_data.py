#!/usr/bin/env python
# coding: utf-8

# In[8]:


import sys
sys.path.append('../../Python')
import cv2
import os
import numpy as np
from util import *
import matplotlib.pyplot as plt

print('--> loading data ...')
sys.stdout.flush()
def load_images_from_folder(folder,imageset):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            imageset.append(img)
    return imageset


images = []
for folder in os.listdir('C:\\Users\\lyum\\MA490DeepLearning\\gestures'):
    load_images_from_folder('C:\\Users\\lyum\\MA490DeepLearning\\gestures\\' + folder,images)


images = np.array(images)
images.shape


labels = []
for i in range(1,12):
    for j in range(1200):
        labels.append(i)


labels = np.array(labels)
labels.shape




