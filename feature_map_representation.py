# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:40:57 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import numpy as np
from PIL import Image
from scipy import ndimage as nd

from tqdm import tqdm


from proUtils import utils

import matplotlib.pyplot as plt
from glob import glob

from pore_features import PoreFeatures


# data_dir
roi_dir = 'D:\\VILI\\Lorenzo\\BLEO\\MD_1264_A6_1_aligned\\roi\\'


all_rois = glob(roi_dir + '*')

bin_th = 70

tissue2vol_ratio = []
for aroi in tqdm(all_rois):
    roi = utils.load_roi(aroi, check_blank=False)
    roi = nd.median_filter(roi, size=2)

    th_vol = roi > bin_th
    th_vol = nd.binary_fill_holes(th_vol, np.ones((3,3,3)))
    th_vol = nd.binary_closing(th_vol, np.ones((2,2,2)))
    
    ratio = np.count_nonzero(th_vol)/np.size(th_vol)
    tissue2vol_ratio.append(ratio)


slice_dir = 'D:\\VILI\\Lorenzo\\BLEO\\MD_1264_A6_1_aligned\\slices\\slice_2151.tif'
im = Image.open(slice_dir)
slice_img = np.array(im)
plt.imshow(slice_img)


root_dir = 'D:\\VILI\\Lorenzo\\BLEO\\MD_1264_A6_1_aligned\\'
pf = PoreFeatures(root_dir=root_dir)
pf.load_files()

pf.load_porespy_features(feature='volume', mode='mean')
pf.draw_features_map(imarray=slice_img, bkg_transp=0.8)

fileName = 'mean_volume_featuremap.tif'

pf.save_features_map(fileName=fileName)
