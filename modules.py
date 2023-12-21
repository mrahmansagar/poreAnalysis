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
from skimage.draw import disk, rectangle
from tqdm import tqdm
from glob import glob
import porespy as ps

from proUtils import utils

import json
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

#-------------------------------------------------------------------------#
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

#-------------------------------------------------------------------------#

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

    

 #============================================================================#
class PoreFeatures:
    """
    PoreFeatures class for extracting and visualizing features from pore data.

    Parameters:
    - root_dir (str): The root directory containing pore data.

    Methods:
    - clear_features(): Clear the stored features.
    - load_porespy_features(feature=None, mode='mean', sub_folder=None, file_pattern='*'): Load features from JSON files.
    - draw_features_map(imarray, cmap='jet', cube_size=300, overlap=100, rec_bkg=True, bkg_transp=0.9): Draw features map.
    - save_features_map(save_dir=None, fileName='features_map.tif'): Save the features map as an image file.
    Attributes:
    - root_dir (str): The root directory containing pore data.
    - found_files (list): List of found JSON files.
    - features (list): List of extracted features.
    - feature_as_circles (numpy.ndarray): Array representing the features map.
    """
    def __init__(self, root_dir):
        """
        Initialize the PoreFeatures class with a root directory.

        Parameters:
        - root_dir (str): The root directory containing pore data.
        """
        self.root_dir = root_dir
        self.found_files = []
        self.features = []
        self.feature_as_circles = None
    
    #-------------------------------------------------------------------------#
    def load_files(self, sub_folder=None, file_pattern='*'):
        """
        Load features from JSON files.
        
        Parameters:
        - sub_folder (str): Sub-folder within the root directory.
        - file_pattern (str): Pattern for identifying relevant files.
        
        Returns:
        - list: List of found files.
        """
        
        # Construct the directory path based on the provided parameters
        if sub_folder is None:
            json_dir = os.path.join(self.root_dir, 'porespy')
        else:
            json_dir = os.path.join(self.root_dir, sub_folder)

        # Use os.path.join for file_pattern as well
        file_pattern = os.path.join(json_dir, file_pattern)
        
        # Use glob directly to find files
        self.found_files = glob(file_pattern)
        
        print(f'found {len(self.found_files)} files in {json_dir}')
    
    #-------------------------------------------------------------------------#
    
    def clear_files(self):
        """ Clear the loaded files. """
        self.found_files = []
    
    #-------------------------------------------------------------------------#
    
    def clear_features(self):
        """Clear the stored features."""
        self.features = []
    
    #-------------------------------------------------------------------------#
    
    def set_features(self, new_features):
        """Set a new list of extracted features."""
        self.features = new_features
    
    #-------------------------------------------------------------------------#
    
    def load_porespy_features(self, feature=None, mode='mean'):
        """
        Load features from JSON files.

        Parameters:
        - feature (str): The name of the feature to extract.
        - mode (str): The extraction mode ('mean', 'median', 'all').
        

        Returns:
        - list: List of extracted features.
        """
        valid_features = ['volume', 'bbox_volume', 'sphericity', 'surface_area',
                          'convex_volume', 'equivalent_diameter_area', 'euler_number',
                          'extent', 'axis_major_length', 'axis_minor_length', 'solidity']
        
        if feature is None:
            print(f'feature not specified. Use one of {valid_features}')
        elif feature not in valid_features:
            raise ValueError(f"Invalid feature: {feature}. Use one of {valid_features}")
        
        valid_modes = ['mean', 'median', 'all']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Use one of {valid_modes}")

        for afile in self.found_files:
            with open(afile) as file:
                data = json.load(file)
                
                try:
                    extract_feature = data[feature]
                    if len(extract_feature) > 0:
                        if mode=='mean':
                            self.features.append(np.mean(extract_feature))
                        elif mode=='median':
                            self.features.append(np.median(extract_feature))
                        elif mode=='all':
                            self.features.extend(extract_feature)
                        else:
                            print(f'Error: {mode} not implemented')
                    
                
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")    

    #-------------------------------------------------------------------------#
    
    def get_features(self):
        """Get the list of extracted features."""
        return self.features
    
    #-------------------------------------------------------------------------#
    
    def draw_features_map(self, imarray, cmap='jet', cube_size=300, overlap=100, 
                          rec_bkg=True, bkg_transp=0.9,):
        """
        Draw features map on an input image array.

        Parameters:
        - imarray (numpy.ndarray): Input image array.
        - cmap (str): Colormap for feature visualization.
        - cube_size (int): Size of cubes.
        - overlap (int): Overlap between cubes.
        - rec_bkg (bool): Whether to include rectangles as background.
        - bkg_transp (float): Background transparency.

        Returns:
        - numpy.ndarray: Array representing the features map.
        """
        normalized_features = [(value - min(self.features)) / (max(self.features) - min(self.features)) for value in self.features]
        # Apply a colormap (e.g., 'viridis' for a perceptually uniform colormap)
        colormap = plt.get_cmap(cmap)
        colors_list = colormap(normalized_features)
        
        self.feature_as_circles = np.zeros(shape=imarray.shape + (4,), dtype=np.float32)
        
        rect_size = cube_size - overlap
        max_radius = rect_size//2
        
        for i, (afile, val) in enumerate(zip(self.found_files, normalized_features)):
            cube_cords = afile.split('\\')[-1]
            cube_cords = cube_cords.split('_bth')[0]
            xrange = cube_cords.split('x')[1]
            x = int(xrange.split('-')[0])
            yrange = cube_cords.split('x')[2]
            y = int(yrange.split('-')[0])
            
            if rec_bkg:
                #generate rec coordinates
                r, c = rectangle((x,y), (x+rect_size, y+rect_size))
                # Assign RGBA values to the corresponding region in ratio_colored
                self.feature_as_circles[r, c, 0] = colors_list[2][0]  # Red channel
                self.feature_as_circles[r, c, 1] = colors_list[2][1]  # Green channel
                self.feature_as_circles[r, c, 2] = colors_list[2][2]  # Blue channel
                self.feature_as_circles[r, c, 3] = colors_list[2][3] - bkg_transp  # Alpha channel
        
            # Generate disk coordinates
            rr, cc = disk((x+max_radius, y+max_radius), int(max_radius*val),
                          shape=imarray.shape)
        
            # Assign RGBA values to the corresponding region in ratio_colored
            self.feature_as_circles[rr, cc, 0] = colors_list[i][0]  # Red channel
            self.feature_as_circles[rr, cc, 1] = colors_list[i][1]  # Green channel
            self.feature_as_circles[rr, cc, 2] = colors_list[i][2]  # Blue channel
            self.feature_as_circles[rr, cc, 3] = colors_list[i][3]  # Alpha channel
        
        self.feature_as_circles = (self.feature_as_circles * 255).astype(np.uint8)  
        
        plt.imshow(imarray, alpha=0.5)
        plt.imshow(self.feature_as_circles)
        plt.show()
        
    #-------------------------------------------------------------------------#
    
    def get_feature_map(self):
        """Get the array representing the features map."""
        return self.feature_as_circles
    
    #-------------------------------------------------------------------------#
    
    def save_features_map(self, save_dir=None, fileName='features_map.tif'):
        """
        Save the features map as an image file.

        Parameters:
        - save_dir (str): Directory to save the image file. If None, save in 
            the root directory.
        - fileName (str): Name of the image file.
        """
        if save_dir is None:
            file_path = os.path.join(self.root_dir, fileName)
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_path = os.path.join(save_dir, fileName)
            
        io.imsave(file_path, self.feature_as_circles) 