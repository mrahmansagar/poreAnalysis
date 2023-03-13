import numpy as np 
import matplotlib.pyplot as plt 
import os 
from tkinter import Tcl # for sorting the slices as in windows dir
from glob import glob 
from tqdm import tqdm 

from poreUtils import *


from PIL import Image

from scipy import ndimage as nd
from skimage.measure import label
#from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import json 

# Using porespy 
import porespy as ps
import scipy.ndimage as spim
ps.visualization.set_mpl_style()
np.random.seed(1)

import pandas as pd

def extract_properties(list_of_r_instances):
    """
    Extract properties from a list of instances of 'r' and store them in a pandas dataframe.

    Parameters
    ----------
    list_of_r_instances : list
        List of instances of 'r'.
    Returns
    -------
    pandas.DataFrame
        Dataframe with columns for each property and rows for each instance of 'r'.
    """
    # Create a list of properties
    property_list = ['label', 'volume', 'bbox_volume', 'sphericity', 'surface_area', 'convex_volume',
                     'centroid', 'equivalent_diameter_area', 'euler_number', 'extent', 
                     'axis_major_length', 'axis_minor_length', 'solidity']

    # Create a dictionary to store the properties of the current instance of r
    properties = {}

    for prop in property_list:
        properties[prop] = []


    # Iterate over each instance of r
    for r_instance in list_of_r_instances:
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

    
    return df

# list all the folders where there rois 

# data dir 
data_dir = 'E:\\Data\\sam_data'


# all the sample folder where there are rois 
sample_folders = os.listdir(data_dir)

all_rois = []
valid_folders = []
for d in sample_folders:
    roi_path = os.path.join(data_dir, d, 'roi')
    if (os.path.exists(roi_path)):
        valid_folders.append(os.path.join(data_dir, d,))
        all_rois += glob((roi_path +'\*'))
        


# Creating a separate folder to put porespy restult

for s in valid_folders:
    if not os.path.exists(os.path.join(s, 'porespy')):
        os.makedirs(os.path.join(s, 'porespy'))


bin_th = 55
dt_th = 12

processed_rois = []
faulty_rois = []

for roi_path in tqdm(all_rois):
    try:
        print('processing....', roi_path)
        
        tiffs = os.listdir(roi_path)
        slices = Tcl().call('lsort', '-dict', tiffs)
        vol = np.empty(shape=(300, 300, 300), dtype=np.uint8)
        for i, fname in enumerate(slices):
            im = Image.open(os.path.join(roi_path, fname))
            imarray = np.array(im)
            imarray = np.clip(imarray, 0.0005, 0.003)
            imarray = norm8bit(imarray)
            vol[i, :, :] = imarray

        th_vol = vol < bin_th
        th_vol = nd.binary_closing(th_vol, np.ones((3,3,3)))
        th_vol = th_vol.astype(np.uint8)
        th_vol = nd.binary_fill_holes(th_vol, np.ones((3,3,3)))
        th_vol = nd.binary_dilation(th_vol, np.ones((3,3,3)))
        th_vol = nd.binary_erosion(th_vol, np.ones((3,3,3)))

        dt3d = nd.distance_transform_edt(th_vol)
        
        mask = dt3d > dt_th
        markers = label(mask)
        labels = watershed(-dt3d, markers, mask=th_vol)

        props = ps.metrics.regionprops_3D(labels)

        df = extract_properties(props)
        features = df.to_dict(orient='list')
        features['bin_th'] = [bin_th]
        features['dt_th'] = [dt_th]
        features['dtmax'] = [dt3d.max()]
        

        jsonString = json.dumps(features)
        jsonFile = open(roi_path.replace('roi', 'porespy') + '_bth' + str(bin_th) + '_dth' + str(dt_th) + '_v2.json', 'w')
        jsonFile.write(jsonString)
        jsonFile.close()
        processed_rois.append(roi_path)
        
    except:
        faulty_rois.append(roi_path)
        print('skipping....', roi_path)
        pass

print(faulty_rois)