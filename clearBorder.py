# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:52:19 2022

@author: Manager
"""

import time 
import os 
import matplotlib.pyplot as plt
import tifffile 
import json
import ctypes

import pypore3d
from pypore3d import *
from pypore3d.p3dFiltPy import *
from pypore3d.p3dBlobPy import *
from pypore3d.p3dSkelPy import *
from pypore3d.p3dSITKPy import *


from poreUtils import *



data_dir = data_dir = 'D:\\sagar\\Data\\MD_1264_A1_1_Z3.3mm\\selected_roi\\'
raw_file = os.path.join(data_dir, '600-900x800-1100x1600-1900_8bit.raw')

out_dir = 'D:\\sagar\\poreAnalysisOutput\\'

# File specifications
x, y, z = 300, 300, 300
res = 0.002 # 2 micron

# reading the 8bit raw file 
volc = py_p3dReadRaw8(raw_file, x, y, z)
# visualization and conversion into numpy array 
vol = swigObjt2uint8Array(volc, x, y, z, plot=True)

# Applying a median filter and thresholding for a binary image 
filt_volc = py_p3dMedianFilter8(volc, x, y, z, width=3)

th_volc, th_valc  = py_p3dAutoThresholding8(filt_volc, x, y, z, methodNum=2) # method 2 is Otsu

invert_vol(th_volc, dimx=x, dimy=y, dimz=z)

th_vol = swigObjt2uint8Array(th_volc, x, y, z, plot=True)

thE_volc = py_p3d_Erode(th_volc, x, y, z, kWidth=1) # tried with 1, 2, 3

thE_vol = swigObjt2uint8Array(thE_volc, x, y, z, plot=True)
 
tifffile.imsave(os.path.join(out_dir, '600-900x800-1100x1600-1900_thE_vol_8bit.raw'), thE_vol)
 
clb_th_volc = py_p3dClearBorderFilter8(th_volc, x, y, z) 

clb_thE_volc = py_p3dClearBorderFilter8(thE_volc, x, y, z)


# Saved the files in local folder for visual comparison in imageJ or something else.


 