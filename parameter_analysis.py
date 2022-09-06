# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:42:14 2022

@author: Mmr Sagar
PhD Student | AG Alves 
MPI for Multidisciplinary Sciences, Germany 


Analysis for parameters obtained from pore analysis.
"""


import os
import random 
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import json

from scipy.stats import norm

from poreUtils import *

root_dir = 'D:\sagar\Data'

samples = os.listdir(root_dir)

#result_path = os.path.join(root_dir, samples[0], 'result\\')

#json_files = glob(result_path + '*.json')

blob_volumes = []

#for s in samples:
sample_path = os.path.join(root_dir, 'MD_1264_B9_Z3.3mm', 'result')
#print(sample_path)
if os.path.exists(sample_path):
    fpath = glob(sample_path+'\*.json')
    for f in fpath:
        file = open(f)
        data = json.load(file)
        blob_volumes += data['blob_analysis']['result']['VOLUME']
        
        
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


norm_blob_volumes = NormalizeData(blob_volumes)
        
plt.hist(blob_volumes, bins=100)#, range=(0, 1))  


# Fit a normal distribution to
# the data:
# mean and standard deviation
mu, std = norm.fit(blob_volumes)

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
  
plt.plot(x, p, 'k', linewidth=1)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)
  
plt.show()
                         
# Opening JSON file
#f = open(json_files[0])
  
#result = json.load(f)

# Closing file
#f.close()