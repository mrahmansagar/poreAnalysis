# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:53:26 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


from proUtils import utils
from proUtils.notebook import vis

# Path to the slices 
data_dir = 'E:\\Data\\sam_data\\new\\MD_1264_B10_Z0.0mm\\slices'
tiffs = os.listdir(data_dir)

# range of newly scanned files 
new_air = 0.0002642
new_paraffin = 0.0009087

# range from the already analyzed files 
old_air = 0.0002605
old_paraffin = 0.0009999


# Reading all the slices in a Volume 
vol = np.empty(shape=(1700, 3657, 3657), dtype=np.uint8)
for i, fname in enumerate(tqdm(tiffs)):
    im = Image.open(os.path.join(data_dir, fname))
    imarray = np.array(im)
    # if new scan then use this to map the values to match the old data range 
    imarray = utils.map_values_to_range(imarray, new_air, new_paraffin, old_air, old_paraffin)
    imarray = np.clip(imarray, 0.0005, 0.003)
    imarray = utils.norm8bit(imarray)
    vol[i, :, :] = imarray
    
vis.interactive_visualize(vol, cmap='gray')
    
xdim = vol.shape[1]
ydim = vol.shape[2]
zdim = vol.shape[0]

# Offset for ROI volume
# Y Range
rowStartingOffset = 700 
rowEndingOffset = ydim - 2300

# X Range 
colStartingOffset = 0  
colEndingOffset = xdim - 3500

roi_vol = vol[:, rowStartingOffset:xdim-rowEndingOffset, colStartingOffset:ydim-colEndingOffset]

#Ploting the middle slice 
plt.figure(figsize=(16,9))
plt.imshow(roi_vol[850, :, :], cmap='gray')
plt.show()

print(roi_vol.shape)

# define the parameter for cliping the roi 
# size of 3D cube 
cube_size = 300
# step size in each direction
step_size = 200

# finding the range for the loop 
depth = roi_vol.shape[0]
# depth = 1200
row = roi_vol.shape[1]
col = roi_vol.shape[2]

print(list(range(0, depth, step_size)))
print(list(range(0, row-step_size, step_size)))
print(list(range(0, col-step_size, step_size)))

del vol 
        
save_dir = os.path.join('D:\\sam_control_new', data_dir.split('\\')[4], 'slices')
print(save_dir)

minmax = {}
all_roi = []
mins = []
maxs = []

depthList = list(range(0, depth, step_size))

for d in tqdm(depthList[:-1]):
    for r in range(0, row-step_size, step_size):
        for c in range(0, col-step_size, step_size):
            cube = roi_vol[d:d+cube_size, r:r+cube_size, c:c+cube_size]
            
            roiName = f'{d}-{d+cube_size}x{rowStartingOffset+r}-{rowStartingOffset+r+cube_size}x{colStartingOffset+c}-{colStartingOffset+c+cube_size}'
            all_roi.append(roiName)
            mins.append(cube.min())
            maxs.append(cube.max())
            
            pathName = f'{os.path.dirname(save_dir)}\\tiles\\{d}-{d+cube_size}x{rowStartingOffset+r}-{rowStartingOffset+r+cube_size}x{colStartingOffset+c}-{colStartingOffset+c+cube_size}'
            utils.saveSlices(cube, pathName)

            

minList = []
for i in range(len(mins)):
    minList.append(mins[i].tolist())
    
maxList = []
for i in range(len(maxs)):
    maxList.append(maxs[i].tolist())
    

minmax = {}
minmax['roi'] = all_roi
minmax['min'] = minList
minmax['max'] = maxList
minmax['coordinate'] = ['depth', 'row', 'col']
minmax['rowstart'] = [rowStartingOffset]
minmax['rowend'] = [rowEndingOffset]
minmax['colstart'] = [colStartingOffset]
minmax['colend'] = [colEndingOffset]

import json
jsonString = json.dumps(minmax)
jsonFile = open(save_dir.split('slices')[0] + 'tiles.json', "w")
jsonFile.write(jsonString)
jsonFile.close()