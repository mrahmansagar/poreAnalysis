# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:16:03 2022

@author: Mmr Sagar
PhD Student | AG Alves 
MPI for Multidisciplinary Sciences, Germany 

Investigation for setting a range for convertiong the data into a specific data 
type

"""


# import neccessary libraries 
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 
from tqdm import tqdm 
from tkinter import Tcl



# My utilies 
from poreUtils import * 

data_dir = 'D:\\sagar\Data\\MD_1264_A3_1_Z3.3mm\\roi'
#selected_files = os.listdir(data_dir)

selected_files = ['0-300x1000-1300x1400-1700']



# mins = []
# maxs = []
# for afile in tqdm(selected_files):
#     file_path = os.path.join(data_dir, afile)
#     raw_vol = []
#     #sorting the slices according to their names like in windows 
#     slices = Tcl().call('lsort', '-dict', os.listdir(file_path))
#     for aSlice in slices:
#         img = Image.open(os.path.join(file_path, aSlice))
#         imgarray = np.array(img)
#         raw_vol.append(imgarray)
        
#     raw_vol = np.asarray(raw_vol)
    
#     mins.append(raw_vol.min())
#     maxs.append(raw_vol.max())
    
    


for afile in tqdm(selected_files):
    file_path = os.path.join(data_dir, afile)
    raw_vol = []
    #sorting the slices according to their names like in windows 
    slices = Tcl().call('lsort', '-dict', os.listdir(file_path))
    for aSlice in slices:
        img = Image.open(os.path.join(file_path, aSlice))
        imgarray = np.array(img)
        raw_vol.append(imgarray)
        
    raw_vol = np.asarray(raw_vol)
    norm_raw_vol = norm8bit(raw_vol, minVal=0.0005, maxVal=0.003)
    #raw_vol.tofile(file_path + '_8bit.raw')
    
    
    
    
    
    
    