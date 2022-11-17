# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:34:52 2022

@author: Mmr Sagar
PhD Student | AG Alves 
MPI for Multidisciplinary Sciences, Germany 

"""

import os
import random 
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

from tqdm import tqdm 

from PIL import Image

from poreUtils import *



#
# Finding the boxplot related parameters 
# https://towardsdatascience.com/how-to-fetch-the-exact-values-from-a-boxplot-python-8b8a648fc813
#

def extract_boxplot(mplt_boxplot):
    medians = [item.get_ydata()[0] for item in mplt_boxplot['medians']]
    means = [item.get_ydata()[0] for item in mplt_boxplot['means']]
    minimums = [item.get_ydata()[0] for item in mplt_boxplot['caps']][::2]
    maximums = [item.get_ydata()[0] for item in mplt_boxplot['caps']][1::2]
    q1 = [min(item.get_ydata()) for item in mplt_boxplot['boxes']]
    q3 = [max(item.get_ydata()) for item in mplt_boxplot['boxes']]
    fliers = [item.get_ydata() for item in mplt_boxplot['fliers']]
    lower_outliers = []
    upper_outliers = []
    for i in range(len(fliers)):
        lower_outliers_by_box = []
        upper_outliers_by_box = []
        for outlier in fliers[i]:
            if outlier < q1[i]:
                lower_outliers_by_box.append(outlier)
            else:
                upper_outliers_by_box.append(outlier)
        lower_outliers.append(lower_outliers_by_box)
        upper_outliers.append(upper_outliers_by_box)
    
    return {'medians': medians, 'means': means, 'minimums': minimums, 'maximums': maximums, 'q1': q1, 'q3':q3, 'lower_outliers': lower_outliers, 'upper_outliers': upper_outliers}
 

#
# Only taking the rois into the consideration 
#
root_dir = 'D:\sagar\Data'

samples = os.listdir(root_dir)
#samples = ['MD_1264_B1_1_Z3.3mm_corr_phrt']

roi_paths = []

for s in samples:
    sample_path = os.path.join(root_dir, s, 'roi')
    #print(sample_path)
    if os.path.exists(sample_path):
        fpath = glob(sample_path+'\*')
        for f in fpath:
            roi_paths.append(f)
        
        
# Taking randomly selected slices into a volume   
globVol = []

for r in tqdm(random.sample(roi_paths, 50)):
#for r in tqdm(roi_paths):
    for aSlice in random.sample(os.listdir(r), 20):
        im = Image.open(os.path.join(r, aSlice))
        imarray = np.array(im)
        #imarray = norm8bit(imarray)
        globVol.append(imarray)
        
globVol = np.asarray(globVol) 

#
# Ploting the histogram of the created volume 
data = np.asarray(globVol.flat)
plt.hist(data, bins=100)
plt.show()

# # Cumulative histogram 
# from scipy import stats

# result = stats.cumfreq(globVol, numbins=100)
# x = result.lowerlimit + np.linspace(0, result.binsize*result.cumcount.size, result.cumcount.size)
# plt.bar(x, result.cumcount, width=result.binsize)
# plt.show()


# #
# # Normalized cumulative histogram 
# #
# def NormalizeData(data):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))

# norCumcount = NormalizeData(result.cumcount)
# plt.bar(x, norCumcount, width=result.binsize)
# plt.show()


# Use clip before normalization 
# np.clip(a, 0.0005, 0.003)
# If cliping is done then usingnormalization with min and max value should be enough 


# 
# Ploting the box plot with the histogram 
#
sns.set(rc={'figure.figsize':(16,10)})
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
markersize = 3
linewidth = 1

fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)})
sns.boxplot(x=data, ax=ax_box)
sns.histplot(x=data, bins=100, kde=False, stat='count', ax=ax_hist)
#sns.histplot(x=data, bins=100, kde=False, stat='count', ax=ax_hist, cumulative=True)#, alpha=0.4)
plt.savefig('range_histogram.png', dpi=1000)
plt.show()

bp = plt.boxplot(data)   
plt.show()
fetch_bp = extract_boxplot(bp)
