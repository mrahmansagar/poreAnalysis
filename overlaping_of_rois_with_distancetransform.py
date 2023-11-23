# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:04:45 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""
import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image 
from scipy import ndimage as nd
from glob import glob
from proUtils import utils


# Sample dir
scan_dir = 'E:\\projects\\AirGANs\\data\\MD_1264_A6_1_Z3.3mm_corr_phrt\\slices\\'
roi_dir = 'E:\\Data\\sam_data\\new\\MD_1264_A6_1_Z3.3mm_corr_phrt\\roi\\'

# volume dimension 
Z, Y, X = 1700, 3681, 3681

# Reading the original volume with clipping and converted to 8bit 
tiffs = os.listdir(scan_dir)

vol = np.empty(shape=(Z, Y, X), dtype=np.uint8)
for i, fname in enumerate(tqdm(tiffs)):
    im = Image.open(os.path.join(scan_dir, fname))
    imarray = np.array(im)
    imarray = np.clip(imarray, 0.0005, 0.003)
    imarray = utils.norm8bit(imarray)
    vol[i, :, :] = imarray
    

# Available ROIs
selected_rois = glob(roi_dir +'*')

# creating a roi dictionary where key is the depth and value is the available 
# rois in that depth range
# filtering rois according to the z stack 
grouped_rois = {}
for r in selected_rois:
    grp = r.split('\\')[-1].split('x')[0]
    if grp in grouped_rois:
            grouped_rois[grp].append(r)
    else:
        grouped_rois[grp] = [r]
        

sel_roi_lr = 200
sel_roi_hr = 500


roi_key = str(sel_roi_lr)+'-'+str(sel_roi_hr) 

# Binarization value for distance transform  
threshold = 40 

# 
dist_thrs = 2

# roi dimension 
x, y, z = 300, 300, 300

# 3d mask for distance transform and size same to the total volume but depth is equal to roi depth 
roi_mask = np.empty(shape=(z, X,Y))
roi_mask[:, :, :] = np.nan

for r in tqdm(grouped_rois[roi_key]):

    roi = utils.load_roi(r) 
    th_roi = roi > threshold
    th_roi = nd.binary_closing(th_roi, np.ones((3,3,3)))
    dis3d = nd.distance_transform_edt(th_roi)
    
    dis3d[dis3d < dist_thrs] = np.nan

    ly = int(r.split('\\')[-1:][0].split('x')[1].split('-')[0])
    hy = int(r.split('\\')[-1:][0].split('x')[1].split('-')[1])
    
    lx = int(r.split('\\')[-1:][0].split('x')[2].split('-')[0])
    hx = int(r.split('\\')[-1:][0].split('x')[2].split('-')[1])
    
    roi_mask[:, ly:hy, lx:hx] = dis3d[:, :, :]
    

sliceNo = 250 # with respect to whole volume
slice_at_roi = sliceNo - sel_roi_lr

vis_xl = 950
vis_xh = 3050

vis_yl = 900
vis_yh = 3600

plt.figure(figsize=(25,9))

plt.imshow(vol[sliceNo, vis_xl:vis_xh,  vis_yl:vis_yh], cmap='gray')
#cbar = plt.colorbar(orientation='horizontal')
#cbar.set_label('Title (Unit)')

plt.imshow(roi_mask[slice_at_roi, vis_xl:vis_xh,  vis_yl:vis_yh], cmap=plt.cm.jet, alpha=0.7)
cbar = plt.colorbar()
cbar.set_label('Title (Unit)')
#plt.savefig('test.png', dpi=100)
plt.show() 