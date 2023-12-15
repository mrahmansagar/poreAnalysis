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

from skimage.draw import disk, rectangle

from tqdm.notebook import tqdm


from proUtils import utils

import matplotlib.pyplot as plt
from glob import glob

# data_dir
data_dir = 'D:\\VILI\\Lorenzo\\MD_1264_A13_1_aligned\\roi\\'
slice_dir = 'D:\\VILI\\Lorenzo\\MD_1264_A13_1_aligned\\slices\\slice_2313.tif'

im = Image.open(slice_dir)
slice_img = np.array(im)
plt.imshow(slice_img)

all_tiles = glob(data_dir + '*')

bin_th = 70

tissue2vol_ratio = []
for tile in tqdm(all_tiles):
    roi = utils.load_roi(tile, check_blank=False)
    roi = nd.median_filter(roi, size=2)

    th_vol = roi > bin_th
    th_vol = nd.binary_fill_holes(th_vol, np.ones((3,3,3)))
    th_vol = nd.binary_closing(th_vol, np.ones((2,2,2)))
    
    ratio = np.count_nonzero(th_vol)/np.size(th_vol)
    tissue2vol_ratio.append(ratio)
    
# Normalize the ratio to the range [0, 1] for colormap scaling
normalized_ratios =  [(value - min(tissue2vol_ratio)) / (max(tissue2vol_ratio) - min(tissue2vol_ratio)) for value in tissue2vol_ratio]
# Apply a colormap (e.g., 'viridis' for a perceptually uniform colormap)
colormap = plt.get_cmap('jet')
colors_list = colormap(normalized_ratios)

# Create a color-mapped pore mask
ratio_colored = np.zeros(shape=slice_img.shape + (4,), dtype=np.float32)

# Assuming slice_img.shape is (height, width)
height, width = slice_img.shape[:2]


for i, (tile, ratio) in enumerate(zip(all_tiles, tissue2vol_ratio)):
    tile_cords = tile.split('\\')[-1]
    xrange = tile_cords.split('x')[1]
    x = int(xrange.split('-')[0])
    yrange = tile_cords.split('x')[2]
    y = int(yrange.split('-')[0])

    #generate rec coordinates
    r, c = rectangle((x,y), (x+200, y+200))
    # Assign RGBA values to the corresponding region in ratio_colored
    ratio_colored[r, c, 0] = colors_list[2][0]  # Red channel
    ratio_colored[r, c, 1] = colors_list[2][1]  # Green channel
    ratio_colored[r, c, 2] = colors_list[2][2]  # Blue channel
    ratio_colored[r, c, 3] = colors_list[2][3] - 0.9  # Alpha channel
    
    # Generate disk coordinates
    rr, cc = disk((x + 100, y + 100), int(150 * ratio), shape=(height, width))

    # Assign RGBA values to the corresponding region in ratio_colored
    ratio_colored[rr, cc, 0] = colors_list[i][0]  # Red channel
    ratio_colored[rr, cc, 1] = colors_list[i][1]  # Green channel
    ratio_colored[rr, cc, 2] = colors_list[i][2]  # Blue channel
    ratio_colored[rr, cc, 3] = colors_list[i][3] #* 255  # Alpha channel

ratio_colored = (ratio_colored * 255).astype(np.uint8)    

# plt.imshow(slice_img, alpha=0.5)
plt.imshow(ratio_colored)
# 
plt.show()

from skimage import io
io.imsave(f'D:\\VILI\\Lorenzo\\MD_1264_A13_1_aligned\\tissue2vol_onlycircles_ratio_bin_th{bin_th}.tif', ratio_colored )