# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:00:10 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""
import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import numpy as np 

import pandas as pd 
import json
from glob import glob 

from proUtils import plot
from proUtils import utils

# Directory where the scans are stored with results 
root_dir = 'E:\\Data\\sam_data\\new\\'

# Getting the scans where porespy analysis has been run 
scans = []
for p in os.listdir(root_dir):
    file_path = os.path.join(root_dir, p, 'porespy')
    if os.path.exists(file_path):
        scans.append(p)
        
# geting the group information
df = pd.read_csv('study_group.csv')
print(df)

# creating diffrent groups from the available scans
scan_groups = {}
for g in np.unique(df['Group']):
    scan_groups[g] = []

for scan in scans:
    scan_index = df.loc[df['Scan'] == str(scan)].index[0]
    group = df['Group'][scan_index]
    scan_groups[group].append(scan)

print(scan_groups)


parameter_per_group = []
for key in scan_groups.keys():
    json_files = []
    if key == 'Con':
        for aFile in scan_groups[key]:
            fpath = os.path.join(root_dir, aFile, 'porespy')
            json_files += glob(fpath + '\*' + 'dth8' + '.json')
    else:
        for aFile in scan_groups[key]:
            fpath = os.path.join(root_dir, aFile, 'porespy')
            json_files += glob(fpath + '\*' + 'dth10' + '.json')

    parameter = []
    for json_file in json_files:
        file = open(json_file)
        data = json.load(file)
        file.close()
        
        # do what we want to plot for exampe, blob volume
        extract_feature = data['volume']
        
        if len(extract_feature) > 0:
            # parameter += extract_feature
            parameter.append(np.mean(extract_feature))
            # parameter.append(utils.ratio_above_threshold(extract_feature, 100000))

    parameter_per_group.append([key, parameter]) 

data = [x[1] for x in parameter_per_group]
grps = [x[0] for x in parameter_per_group]

plot.box_plot(data, grps, pval=True, showfliers=False)
