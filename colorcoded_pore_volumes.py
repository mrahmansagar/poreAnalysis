# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:11:28 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import numpy as np
from scipy import ndimage as nd
from skimage.measure import label
from skimage.segmentation import watershed

from tqdm import tqdm

import porespy as ps
from porespy.tools import bbox_to_slices
from proUtils import utils

import matplotlib.pyplot as plt


roi_path = 'D:\\VILI\\Lorenzo\\test_cube\\slices\\'

bin_th = 55
dth = 8
roi = utils.load_roi(roi_path, check_blank=True)
roi = nd.median_filter(roi, size=2)

th_vol = roi < bin_th
th_vol = nd.binary_fill_holes(th_vol, np.ones((3,3,3)))
th_vol = nd.binary_closing(th_vol, np.ones((2,2,2)))

dt3d = nd.distance_transform_edt(th_vol)
mask = dt3d > dth
markers = label(mask)

labels = watershed(-dt3d, markers, mask=th_vol)
props = ps.metrics.regionprops_3D(labels)

pore_volumes = [p.volume for p in props]

# Normalize the pore volumes to the range [0, 1] for colormap scaling
normalized_pore_volumes =  [(value - min(pore_volumes)) / (max(pore_volumes) - min(pore_volumes)) for value in pore_volumes]

# Apply a colormap (e.g., 'viridis' for a perceptually uniform colormap)
colormap = plt.get_cmap('jet')
color_list = colormap(normalized_pore_volumes)

# Create a color-mapped pore mask
pore_volume_colored = np.zeros(shape=th_vol.shape + (4,), dtype=np.float32)

for prop, color in tqdm(zip(props, color_list), total=len(color_list)):
    for z, x, y in prop.coords:
        pore_volume_colored[z, x, y, 0] = color[0]  # Red component
        pore_volume_colored[z, x, y, 1] = color[1]  # Green component
        pore_volume_colored[z, x, y, 2] = color[2]  # Blue component
        pore_volume_colored[z, x, y, 3] = color[3]  # Alpha channel

pore_volume_colored = (pore_volume_colored * 255).astype(np.uint8)


utils.save_vol_as_slices(pore_volume_colored, 'D:\\VILI\\Lorenzo\\test_cube\\pore_volume_colored')


"""
Putting different labels for each identified pores
the color is not scalled based on the size of the volume.
"""
# Sort pores by volume in ascending order
props.sort(key=lambda x: x.volume, reverse=False)

pore_mask = np.zeros(shape=th_vol.shape, dtype=np.uint16)
object_values = list(range(1, 100*len(props)+1, 100))
for obj_val, prop in tqdm(zip(object_values, props), desc="Creating Pore Mask"):
    for z,x,y in prop.coords:
        pore_mask[z,x,y] = obj_val


utils.save_vol_as_slices(pore_mask, 'pore_mask_scaled')

# adding color based on the labels 
object_values = list(range(1, 100*len(props)+1, 100))
# Normalize the volumes to the range [0, 1] for colormap scaling
normalized_volumes =  [(value - min(object_values)) / (max(object_values) - min(object_values)) for value in object_values]
# Apply a colormap (e.g., 'viridis' for a perceptually uniform colormap)
colormap = plt.get_cmap('jet')
colors_list = colormap(normalized_volumes)


# Create a color-mapped pore mask
pore_mask_colored = np.zeros(shape=th_vol.shape + (3,), dtype=np.float32)
for obj_val, prop, color in zip(object_values, props, colors_list):
    for z, x, y in prop.coords:
        pore_mask_colored[z, x, y, 0] = color[0]  # Red component
        pore_mask_colored[z, x, y, 1] = color[1]  # Green component
        pore_mask_colored[z, x, y, 2] = color[2]  # Blue component
        
pore_mask_colored = (pore_mask_colored * 255).astype(np.uint8)

"""
color coded inscribed spheres
"""
# sphere diametes
sphere_diameter = [p.equivalent_diameter_area for p in props]

# Normalize the diameter to the range [0, 1] for colormap scaling
normalized_diameters =  [(value - min(sphere_diameter)) / (max(sphere_diameter) - min(sphere_diameter)) for value in sphere_diameter]
# Apply a colormap (e.g., 'viridis' for a perceptually uniform colormap)
colormap = plt.get_cmap('jet')
colors_list = colormap(normalized_diameters)

# Create a color-mapped pore mask
sphere_colored = np.zeros(shape=th_vol.shape + (4,), dtype=np.float32)

for prop, color in tqdm(zip(props, colors_list), total=len(props)):
    im = np.zeros(shape=roi.shape)
    mask = prop.image
    temp = mask * prop['inscribed_sphere']
    s = bbox_to_slices(prop.bbox)
    im[s] = temp
    mask_cords = np.transpose(np.where(im==1))
    for z, x, y in mask_cords:
        sphere_colored[z, x, y, 0] = color[0]  # Red component
        sphere_colored[z, x, y, 1] = color[1]  # Green component
        sphere_colored[z, x, y, 2] = color[2]  # Blue component
        sphere_colored[z, x, y, 3] = color[3]  # Alpha channel

sphere_colored = (sphere_colored * 255).astype(np.uint8)
utils.save_vol_as_slices(sphere_colored, 'H:\\test_cube\\new\\sphere_colored2')
