# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:03:35 2023

@author: mrahm
"""
import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import io, img_as_ubyte
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.draw import disk
from tqdm import tqdm

import porespy as ps

from proUtils import utils


import pandas as pd

def extract_properties(props_list):
    """
    Extract properties from a list of instances of 'r' and store them in a pandas dataframe.

    Parameters
    ----------
    list_of_r_instances : list
        List of instances of 'r'.
    Returns
    -------
    pandas.DataFrame
        Dataframe with columns for each property and rows for each instance of 'r'.
    """
    # Create a list of properties
    feature_list = ['label', 'volume', 'bbox_volume', 'sphericity', 'surface_area', 'convex_volume',
                     'centroid', 'equivalent_diameter_area', 'euler_number', 'extent', 
                     'axis_major_length', 'axis_minor_length', 'solidity']

    # Create a dictionary to store the properties of the current instance of r
    properties = {}

    for feature in feature_list:
        properties[feature] = []


    # Iterate over each instance of r
    for prop in props_list:
        # Iterate over the property list
        for feature in feature_list:
            try:
                # Extract the property from the current instance of r
                value = getattr(prop, feature)

                if feature == 'centroid':
                    # If the property is 'centroid', convert it to the nearest integer
                    value = np.round(value).astype(int).tolist()

                properties[feature].append(value)

            except AttributeError:
                # If the property does not exist, set it to None
                properties[feature] = None
            except NotImplementedError:
                # If there is a NotImplementedError, set it to None and continue to the next iteration
                properties[feature] = None
                continue

    # Create a pandas dataframe from the data list
    df = pd.DataFrame(properties)

    
    return df



    
def make_pore_masks(roi_path, bin_th=55, dth=10, vol_range=None, seperate_msk=True):
    """
    Generate pore masks from a given region of interest.

    Parameters:
    - roi_path (str): Path to the region of interest.
    - bin_th (int): Binary threshold value (default is 55).
    - dth (int): Distance threshold value (default is 10).
    - vol_range (tuple): Range of volumes for the pores.
    - separate_msk (bool): Whether to save separate masks for each pore or a combined mask.

    Returns:
    None
    """
    pore_folder = roi_path + '_dt'+str(dth)+'_pores'
    vol = utils.load_roi(roi_path, check_blank=True)
    vol = nd.median_filter(vol, size=2)
    
    th_vol = vol < bin_th
    th_vol = nd.binary_fill_holes(th_vol, np.ones((3,3,3)))
    th_vol = nd.binary_closing(th_vol, np.ones((2,2,2)))
    th_vol_folder = os.path.join(pore_folder, 'th_vol')
    utils.save_vol_as_slices(img_as_ubyte(th_vol), th_vol_folder)
    
    dt3d = nd.distance_transform_edt(th_vol)
    mask = dt3d > dth
    markers = label(mask)
    
    labels = watershed(-dt3d, markers, mask=th_vol)
    props = ps.metrics.regionprops_3D(labels)
    
    if vol_range is not None:
        props = [x for x in props if x.volume >=vol_range[0] and x.volume <=vol_range[1]]
    
    if not seperate_msk:
        if len(props) < 256:
            pore_mask = np.zeros(shape=th_vol.shape, dtype='uint8')
            object_values = np.random.randint(1, 255, len(props))   
        else:
            pore_mask = np.zeros(shape=th_vol.shape, dtype='uint16')
            object_values = np.random.randint(1, 65535, len(props))
        
        for obj_val, prop in tqdm(zip(object_values, props), desc="Creating Pore Mask"):
            for z,x,y in prop.coords:
                pore_mask[z,x,y] = obj_val
        if vol_range is not None:       
            fName = os.path.join(pore_folder, 'pores_bwtn_'+ str(int(vol_range[0]))+ 'and' +str(int(vol_range[1])) )
        else:
            fName = os.path.join(pore_folder, 'allpores')
        utils.save_vol_as_slices(pore_mask, fName)
    else:
        for prop in tqdm(props, desc="Creating Pore Mask"):
            # Create an empty binary mask
            pore = np.zeros(shape=vol.shape, dtype=np.uint8)
            # Set the pixels corresponding to the coordinates to 255
            for z, x, y in prop.coords:
                pore[z, x, y] = 255
    
            fName = os.path.join(pore_folder, str(int(prop.volume)))
            utils.save_vol_as_slices(pore, fName)



def make_3d_colormap(arr, xy_shape=(300,300), pos=None, radius=10, 
                     colormap='jet', save_dir=None):

    color_range = list(np.arange(int(np.min(arr)), int(np.max(arr))))
    # Normalize the diameter to the range [0, 1] for colormap scaling
    normalized_color_range =  [(value - min(color_range)) / (max(color_range) - min(color_range)) for value in color_range]
    
    colormap = plt.get_cmap('jet')
    
    color_list= colormap(normalized_color_range)
    
    color_bar = np.zeros((xy_shape[0], xy_shape[1], len(color_list), 4), dtype=np.float32)
    
    for index in range(color_bar.shape[2]):        
        
        if pos is None:
            col = int(xy_shape[0]//2)
            row = int(xy_shape[1]//2)

        rr, cc = disk((row,col), radius)

        color_bar[rr, cc, index] = color_list[index]*255
    
    if save_dir is not None:    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for num in tqdm(range(0, color_bar.shape[2])):
            I = np.asarray(color_bar[:,:,num], dtype='uint8')
            fName = os.path.join(save_dir, f'slice_{num:04d}.tif') 
            io.imsave(fName, I)
        
    else:
        return color_bar
    

        