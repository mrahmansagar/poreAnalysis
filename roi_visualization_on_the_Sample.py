# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:14:29 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image 

from proUtils import utils


# Sample dir
scan_dir = 'E:\\projects\\AirGANs\\data\\MD_1264_A6_1_Z3.3mm_corr_phrt\\slices'
roi_dir = 'E:\\Data\\sam_data\\new\\MD_1264_A6_1_Z3.3mm_corr_phrt\\roi'

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
selected_rois = os.listdir(roi_dir)

# creating a roi dictionary where key is the depth and value is the available 
# rois in that depth range
# filtering rois according to the z stack 
grouped_rois = {}
for r in selected_rois:
    grp = r.split('x')[0]
    if grp in grouped_rois:
            grouped_rois[grp].append(r)
    else:
        grouped_rois[grp] = [r]
 
# roi_range 
roi_key = '200-500'

slice_number = 300

fig, ax = plt.subplots(figsize=(16,9))
for aroi in grouped_rois[roi_key]:
    roi_xrange = aroi.split('x')[2]
    x1 = int(roi_xrange.split('-')[0])
    x2 = int(roi_xrange.split('-')[1])
    roi_yrange = aroi.split('x')[1]
    y1 = int(roi_yrange.split('-')[0])
    y2 = int(roi_yrange.split('-')[1])
    

    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', lw=1)
    ax.add_patch(rect)

# plt.subplots_adjust(hspace=0.1, wspace=0.1)
ax.imshow(vol[slice_number, :, :], cmap='gray')
plt.show()
        