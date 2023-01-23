# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:58:09 2023

@author: Mmr Sagar
"""



import os 
import numpy as np 
import matplotlib.pyplot as plt
import json 
import ctypes 
import tifffile 

from scipy import ndimage as nd

import pypore3d 
from pypore3d import *
from pypore3d.p3dFiltPy import *
from pypore3d.p3dBlobPy import *
from pypore3d.p3dSkelPy import *
from pypore3d.p3dSITKPy import *


from glob import glob 
from tqdm import tqdm 

from poreUtils import *

data_dir = 'D:\\sagar\\Data\\MD_1264_A2_1_Z3.3mm\\result\\'
#data_dir = 'D:\\sagar\\Data\\MD_1264_A18\\'
# roi specification 
x, y, z = 300, 300, 300
res = 0.002 # 2 micron 

rois = glob(data_dir + '*.raw')


for roi in tqdm(rois[0:1]):
    result = {}
    # Reading the file 
    volc = py_p3dReadRaw8(roi, x, y, z)
    # Appying Median filter for removing noise
    volc = py_p3dMedianFilter8(volc, x, y, z, width=3)
    # convert it to numpy array for manual thresholding and a bit processing which are easy in numpy array
    vol = swigObjt2uint8Array(volc, x, y, z)
    th_vol = vol < 55 
    th_vol = nd.binary_closing(th_vol, np.ones((3,3,3)))
    # From binary to 0-255
    th_vol = norm8bit(th_vol, 0, 1)
    # Converting back to py_p3d data format to be able to use the functionality 
    th_vol.tofile('th_vol.raw')
    th_volc = py_p3dReadRaw8('th_vol.raw', x, y, z)
    #os.remove('th_vol.raw')
    
    # Basic Analysis
    basic_stats = py_p3dBasicAnalysis(th_volc, x, y, z, resolution=res)
    basic_anaysis = formatBasicStats(basic_stats, definition=True)
    result['basic_analysis'] = basic_anaysis
    
    
    # Anisotropy Analysis
    anisotropy_stats = py_p3dAnisotropyAnalysis(th_volc, x, y, z, resolution=res)
    anisotropy_analysis = formatAnisotropyStats(anisotropy_stats, definition=True)
    result['anisotropy_analysis'] = anisotropy_analysis
    
    # Blob Analysis 
    blob_stats, blob_im, star_im = py_p3dBlobAnalysis(th_volc, x, y, z, resolution=res)
    blob_analysis = formatBlobStats(blob_stats, definition=True)
    result['blob_analysis'] = blob_analysis
    
    
blob_vol = swigObjt2uint8Array(blob_im, x, y, z)
star_vol = swigObjt2uint8Array(star_im, x, y, z)

cor_vol = np.copy(star_vol)

cor_vol[cor_vol > 1] = 0


ax = plt.figure().add_subplot(projection='3d')
ax.quiver(1,8,137, 298, 296, 223)
plt.show()