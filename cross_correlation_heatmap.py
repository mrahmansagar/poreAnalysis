import os 
os.sys.path.insert(0, 'E:\\dev\\packages')

from glob import glob
import numpy as np

import json
import pandas as pd


from proUtils import utils
from proUtils import plot


# Directory where the scans are stored with results 
root_dir = 'E:\\Data\\sam_data\\new\\'

# geting the group information
df = pd.read_csv('study_group.csv')

# Getting the scans where analysis has been run 
scans = []
for p in os.listdir(root_dir):
    file_path = os.path.join(root_dir, p, 'porespy')
    if os.path.exists(file_path):
        scans.append(p)

# creating diffrent groups from the available scans
scan_groups = {}
for g in np.unique(df['Group']):
    scan_groups[g] = []

for scan in scans:
    scan_index = df.loc[df['Scan'] == str(scan)].index[0]
    group = df['Group'][scan_index]
    scan_groups[group].append(scan)

print(scan_groups)


bleo_scans = scan_groups['Ble']
bleo_vili_scans = scan_groups['Ble-VILI']
con_scans = scan_groups['Con']
cntrl_vili_scans = scan_groups['Con-VILI']


file_group = con_scans
file_regx = 'dth8.json'

# Cross-Correlation of features for all scans in each group 
listOfFeatures = ['volume', 'sphericity', 'surface_area', 'convex_volume', 
                  'equivalent_diameter_area', 'extent', 'solidity', 'big_pore_ratio']

# initialized data dictionary with an empty list with each feature as a key
data_dict = {feature: [] for feature in listOfFeatures}


json_files = []
for aScan in file_group:
    fpath = os.path.join(root_dir, aScan, 'porespy')
    json_files += glob(fpath + '\*' + file_regx)


for afile in json_files:
    f = open(afile)
    data = json.load(f)
    f.close()
       
    for feature in listOfFeatures:
        if feature == 'big_pore_ratio':
            extract_feature = data['volume']
            if len(extract_feature) > 0:
                data_dict[feature].append(utils.ratio_above_threshold(extract_feature, 100000))
        else:
            extract_feature = data[feature]
            if len(extract_feature) > 0:
                data_dict[feature].append(np.mean(extract_feature))
    

# changing the big feature name to make the plot look better
data_dict['equ_dia_area'] = data_dict.pop('equivalent_diameter_area')
plot.cross_corr_heatmap(data_dict)


# Cross-Correlation of features for all scans irrespective of group 
file_group = cntrl_vili_scans + bleo_vili_scans + bleo_scans
file_regx = 'dth10.json'

json_files = []
for aScan in file_group:
    fpath = os.path.join(root_dir, aScan, 'porespy')
    json_files += glob(fpath + '\*' + file_regx)


for afile in json_files:
    f = open(afile)
    data = json.load(f)
    f.close()
       
    for feature in listOfFeatures:
        if feature == 'big_pore_ratio':
            extract_feature = data['volume']
            if len(extract_feature) > 0:
                data_dict[feature].append(utils.ratio_above_threshold(extract_feature, 100000))
        else:
            extract_feature = data[feature]
            if len(extract_feature) > 0:
                data_dict[feature].append(np.mean(extract_feature))
    

# changing the big feature name to make the plot look better
data_dict['equ_dia_area'] = data_dict.pop('equivalent_diameter_area')
plot.cross_corr_heatmap(data_dict)

