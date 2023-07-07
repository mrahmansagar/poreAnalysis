import time
start_time = time.time()
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from tkinter import Tcl # for sorting the slices as in windows dir
from glob import glob 
from tqdm.notebook import tqdm 
import pandas as pd

from poreUtils import *


from PIL import Image

from scipy import ndimage as nd
from skimage.measure import label
#from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import json 


import multiprocessing

# Using porespy 
import porespy as ps
import scipy.ndimage as spim
ps.visualization.set_mpl_style()
np.random.seed(1)


# list all the folders where there rois 

# data dir 
data_dir = 'E:\\Data\\sam_data\\new'


# all the sample folder where there are rois 
sample_folders = os.listdir(data_dir)

all_rois = []
valid_folders = []
for d in sample_folders:
    roi_path = os.path.join(data_dir, d, 'roi')
    if (os.path.exists(roi_path)):
        valid_folders.append(os.path.join(data_dir, d,))
        all_rois += glob((roi_path +'\*'))


# find the rois that are already processed and remove them from the list of rois 
porespy_processed_dth10 = []

for p in sample_folders:
    porespy_path = os.path.join(data_dir, p, 'porespy') 
    porespy_processed_dth10 += glob((porespy_path + '\*_dth10.json'))

# remove the ones that are already processed 
for apfile in porespy_processed_dth10:
    all_rois.remove(apfile.replace('porespy', 'roi').split('_bth')[0])

print('processing', len(all_rois))

# Creating a separate folder to put porespy result
for s in valid_folders:
    if not os.path.exists(os.path.join(s, 'porespy')):
        os.makedirs(os.path.join(s, 'porespy'))



def parallel_process(path_roi):
    
    bin_th = 55
    dt_th = 10
    
    try:
        print('processing....', path_roi)
        
        tiffs = os.listdir(path_roi)
        slices = Tcl().call('lsort', '-dict', tiffs)
        vol = np.empty(shape=(300, 300, 300), dtype=np.uint8)
        # Temporary list to hold blank slices
        blank_slices = []

        for i, fname in enumerate(slices):
            im = Image.open(os.path.join(path_roi, fname))
            imarray = np.array(im)
            
            if np.all(imarray == 0):
                blank_slices.append(imarray)
            else:
                vol[i - len(blank_slices), :, :] = imarray


        # Append blank slices at the end
        if len(blank_slices) > 0:
            vol[-len(blank_slices):] = blank_slices

        th_vol = vol < bin_th
        th_vol = nd.binary_fill_holes(th_vol, np.ones((3,3,3)))
        th_vol = nd.binary_closing(th_vol, np.ones((3,3,3)))
        dt3d = nd.distance_transform_edt(th_vol)



        mask = dt3d > dt_th
        markers = label(mask)
        labels = watershed(-dt3d, markers, mask=th_vol)
        props = ps.metrics.regionprops_3D(labels)

        # Create a list of properties
        property_list = ['label', 'volume', 'bbox_volume', 'sphericity', 'surface_area', 'convex_volume',
                            'centroid', 'equivalent_diameter_area', 'euler_number', 'extent', 
                            'axis_major_length', 'axis_minor_length', 'solidity']

        # Create a dictionary to store the properties of the current instance of r
        properties = {}

        for prop in property_list:
            properties[prop] = []

        # Iterate over each instance of r
        for r_instance in props:
            # Iterate over the property list
            for prop in property_list:
                try:
                    # Extract the property from the current instance of r
                    value = getattr(r_instance, prop)

                    if prop == 'centroid':
                        # If the property is 'centroid', convert it to the nearest integer
                        value = np.round(value).astype(int).tolist()

                    properties[prop].append(value)

                except AttributeError:
                    # If the property does not exist, set it to None
                    properties[prop] = None
                except NotImplementedError:
                    # If there is a NotImplementedError, set it to None and continue to the next iteration
                    properties[prop] = None
                    continue
        
        # Create a pandas dataframe from the data list
        df = pd.DataFrame(properties)


        features = df.to_dict(orient='list')
        features['bin_th'] = [bin_th]
        features['dt_th'] = [dt_th]
        features['dtmax'] = [dt3d.max()]
        
        jsonString = json.dumps(features)
        jsonFile = open(path_roi.replace('roi', 'porespy') + '_bth' + str(bin_th) + '_dth' + str(dt_th) + '.json', 'w')
        jsonFile.write(jsonString)
        jsonFile.close()        
        
    except:
        print('skipping....', path_roi)
        pass




if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=2)
    pool.map(parallel_process, all_rois[0:2])
    pool.close()

end_time = time.time()
execution_time = end_time - start_time

print("Execution time: {:.2f} seconds".format(execution_time))
