# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:34:52 2022

@author: Mmr Sagar
"""

import os
import random 
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 

from PIL import Image

from poreUtils import *

root_dir = 'D:\sagar\data'

samples = os.listdir(root_dir)


roi_paths = []

for s in samples:
    sample_path = os.path.join(root_dir, s, 'roi')
    #print(sample_path)
    if os.path.exists(sample_path):
        fpath = glob(sample_path+'\*')
        for f in fpath:
            roi_paths.append(f)
        
        
  
globVol = []


for r in tqdm(random.sample(roi_paths, 50)):
    for aSlice in random.sample(os.listdir(r), 5):
        im = Image.open(os.path.join(r, aSlice))
        imarray = np.array(im)
        #imarray = norm8bit(imarray)
        globVol.append(imarray)
        
globVol = np.asarray(globVol) 


plt.hist(globVol.flat, bins=100)
plt.show()

# Cumulative histogram 

from scipy import stats

result = stats.cumfreq(globVol, numbins=100)

x = result.lowerlimit + np.linspace(0, result.binsize*result.cumcount.size, result.cumcount.size)


plt.bar(x, result.cumcount, width=result.binsize)

plt.show()


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

norCumcount = NormalizeData(result.cumcount)

plt.bar(x, norCumcount, width=result.binsize)


plt.show()


# Use clip before normalization 
np.clip(a, 0.0005, 0.003)

# Then threshold should work 

# If cliping is done then usingnormalization with min and max value should be enough 