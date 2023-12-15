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
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.found_files = []
        self.features = []
        self.feature_as_circles = None
        
    def clear_features(self):
        """Clear the stored features."""
        self.features = []
        
    def load_features(self, feature=None, mode='mean', sub_folder=None, file_pattern='*'):
        
        valid_modes = ['mean', 'median', 'all']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Use one of {valid_modes}")

        # Construct the directory path based on the provided parameters
        if sub_folder is None:
            json_dir = os.path.join(self.root_dir, 'porespy/')
        else:
            json_dir = os.path.join(self.root_dir, sub_folder, '/')

        # Use os.path.join for file_pattern as well
        file_pattern = os.path.join(json_dir, file_pattern)

        # Use glob directly to find files
        self.found_files = glob(file_pattern)

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
        
        
    
    
    
    
    