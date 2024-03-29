# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:06:05 2022

@author: Mmr Sagar
PhD Student | AG Alves 
MPI for Multidisciplinary Sciences, Germany 


Effect of threshold value 

"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import random 
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from tqdm import tqdm 

from PIL import Image

from proUtils import utils

# Only taking the rois into the consideration 

root_dir = 'E:\\Data\\sam_data\\new'

samples = os.listdir(root_dir)
#samples = ['MD_1264_B1_1_Z3.3mm_corr_phrt']

roi_paths = []

for s in samples:
    sample_path = os.path.join(root_dir, s, 'roi')
    #print(sample_path)
    if os.path.exists(sample_path):
        fpath = glob(sample_path+'\*')
        for f in fpath:
            roi_paths.append(f)
        
        
# Taking randomly selected slices into a volume   
globVol = []

# Number of randomly selcted rois for the calculation 
nbOfvolumes = 30
# Number of slices to be considered for the global volume 
nbOfslices = 10

selected_volumes = random.sample(roi_paths, 30)

for r in tqdm(selected_volumes):
    for aSlice in random.sample(os.listdir(r), 10):
        im = Image.open(os.path.join(r, aSlice))
        imarray = np.array(im)
        #imarray = norm8bit(imarray)
        globVol.append(imarray)
        
globVol = np.asarray(globVol) 

# New rois are already clipped 
#globVol = np.clip(globVol, 0.0005, 0.003)

globVol_8bit = utils.norm8bit(globVol)

voxelCount = []
thRange = range(0, 180)

for i in thRange:
    th_vol = globVol_8bit < i
    voxelCount.append(np.count_nonzero(th_vol))

sns.set(rc={'figure.figsize':(16,10)})
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
markersize = 3
linewidth = 2

    
plt.plot(list(thRange), voxelCount, '-')
plt.xlabel('threshold')
plt.ylabel('number of voxel')
#plt.savefig('thresholdVsnbVox.svg')
plt.show()