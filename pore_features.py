# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:57:13 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""

import os
from glob import glob
import numpy as np

import matplotlib.pyplot as plt
from skimage.draw import disk, rectangle

import json

class PoreFeatures:
    """
    PoreFeatures class for extracting and visualizing features from pore data.

    Parameters:
    - root_dir (str): The root directory containing pore data.

    Methods:
    - clear_features(): Clear the stored features.
    - load_porespy_features(feature=None, mode='mean', sub_folder=None, file_pattern='*'): Load features from JSON files.
    - draw_features_map(imarray, cmap='jet', cube_size=300, overlap=100, rec_bkg=True, bkg_transp=0.9): Draw features map.

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
    
    def clear_files(self):
        """ Clear the loaded files. """
        self.found_files = []
    
    def clear_features(self):
        """Clear the stored features."""
        self.features = []
        
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
                
        return self.features
    
    
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
        
        for i, (afile, val) in enumerate(zip(self.found_files, self.features)):
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
            rr, cc = disk((x+max_radius, y+max_radius), int(max_radius*val), shape=imarray.shape)
        
            # Assign RGBA values to the corresponding region in ratio_colored
            self.feature_as_circles[rr, cc, 0] = colors_list[i][0]  # Red channel
            self.feature_as_circles[rr, cc, 1] = colors_list[i][1]  # Green channel
            self.feature_as_circles[rr, cc, 2] = colors_list[i][2]  # Blue channel
            self.feature_as_circles[rr, cc, 3] = colors_list[i][3]  # Alpha channel
        
        self.feature_as_circles = (self.feature_as_circles * 255).astype(np.uint8)  
        
        plt.imshow(imarray, alpha=0.5)
        plt.imshow(self.feature_as_circles)
        plt.show()
        
        return self.feature_as_circles 
    
    
        
    
    
    
    
    