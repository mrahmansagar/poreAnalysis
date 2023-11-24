# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:03:14 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""


import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import numpy as np 

from glob import glob 
from tqdm import tqdm 


from scipy import ndimage as nd
from skimage.measure import label
from skimage.segmentation import watershed

import time

import json 

# Using porespy 
import porespy as ps
np.random.seed(1)

from proUtils import utils
# from the same directory and with some useful functions for poreanalysis
import modules



# list all the folders where there rois 
data_dir = 'E:\\Data\\sam_data\\new'

# all the sample folder where there are rois 
sample_folders = os.listdir(data_dir)

# folder where rois are and where to save the results(folders are relative to the data_dir)
roi_folder = 'roi'
result_folder = 'porespy'

# threshold for binarization 
bin_th = 55
# threshold for distance transform to create seed for segmentation 
dt_th = 10

file_sufix = f'_dth{dt_th}.json'


# Getting all the rois in the sample folders 
all_rois = []
valid_folders = []
for d in sample_folders:
    roi_path = os.path.join(data_dir, d, roi_folder)
    if (os.path.exists(roi_path)):
        valid_folders.append(os.path.join(data_dir, d,))
        all_rois += glob((roi_path +'\*'))
        

# find the rois that are already processed and remove them from the list of rois 
porespy_processed = []

for p in sample_folders:
    porespy_path = os.path.join(data_dir, p, result_folder) 
    porespy_processed += glob((porespy_path + '\*' + file_sufix))

    # remove the ones that are already processed 
for apfile in porespy_processed:
    all_rois.remove(apfile.replace(result_folder, roi_folder).split('_bth')[0])
    

# creating folder to put the result in every sample directory   
for s in valid_folders:
    if not os.path.exists(os.path.join(s, result_folder)):
        os.makedirs(os.path.join(s, result_folder))
        
# list of processed rois (only for checking the progress)
processed_rois = []
faulty_rois = []

for roi_path in tqdm(all_rois):
    start_time = time.time()
    try:
        print('processing....', roi_path)
        vol = utils.load_roi(roi_path, check_blank=True)
        vol = nd.median_filter(vol, size=2)
        th_vol = vol < bin_th
        th_vol = nd.binary_fill_holes(th_vol, np.ones((3,3,3)))
        th_vol = nd.binary_closing(th_vol, np.ones((2,2,2)))
        # distance transform to find the seeds for the segmentation
        dt3d = nd.distance_transform_edt(th_vol)
        #creating seed points for with definded dt_th
        mask = dt3d > dt_th
        markers = label(mask)
        labels = watershed(-dt3d, markers, mask=th_vol)
        #using the porespy package to create props for each label of watershade segmentation  
        props = ps.metrics.regionprops_3D(labels)
        # using the props to find the features of each identified object
        df = modules.extract_properties(props)
        features = df.to_dict(orient='list')
        features['bin_th'] = [bin_th]
        features['dt_th'] = [dt_th]
        features['dtmax'] = [dt3d.max()]
        
        jsonString = json.dumps(features)
        jsonFile = open(roi_path.replace(roi_folder, result_folder) + '_bth' + str(bin_th) + file_sufix, 'w')
        jsonFile.write(jsonString)
        jsonFile.close()
        processed_rois.append(roi_path)
        
    except:
        faulty_rois.append(roi_path)
        print('skipping....', roi_path)
        pass
    
    end_time = time.time()
    execution_time = end_time - start_time

    print("Execution time: {:.2f} seconds".format(execution_time))

print(faulty_rois)