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
import poreUtils 

data_dir = 'D:\sagar\Data\MD_1264_A1_1_Z3.3mm\selected_roi'
selected_files = os.listdir(data_dir)



roi_info = {}
mins = []
maxs = []
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
    
    roi_info[afile] = [raw_vol.min(), raw_vol.max()]
    mins.append(raw_vol.min())
    maxs.append(raw_vol.max())
    
    #raw_vol = poreUtils.norm8bit(raw_vol)
    
    
    
def norm8bit(v, mn, mx):
    """
    NORM8BIT function takes an array and normalized it before converting it into 
    a 8 bit unsigned integer and returns it.

    Parameters
    ----------
    v : numpy.ndarray
        Array of N dimension.

    Returns
    -------
    numpy.ndarray (uint8)
        Numpy Array of same dimension as input with data type as unsigned integer 8 bit

    """
    
    #mn = v.min()
    #mx = v.max()
      
    mx -= mn
      
    v = ((v - mn)/mx) * 255
    
    return v.astype(np.uint8)



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
    raw_vol = norm8bit(raw_vol, mn=max(mins), mx=min(maxs))
    tifffile.imsave((file_path + '_8bit.raw'), raw_vol)
    
    
    
    
    
    
    