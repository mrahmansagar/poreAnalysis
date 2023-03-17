import os 
import sys
sys.path.append('E:\\dev\\')

from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json
import pandas as pd

from tqdm import tqdm


from processTools import utils


# Directory where the scans are stored with results 
root_dir = 'E:\\Data\\sam_data\\'

# geting the group information
df = pd.read_csv('study_group.csv')

# Getting the scans where pypore3d analysis has been run 
scans = []
for p in os.listdir(root_dir):
    file_path = os.path.join(root_dir, p, 'porespy')
    if os.path.exists(file_path):
        scans.append(p)

# creating diffrent groupsfrom the available scans
scan_groups = {}
for g in np.unique(df['Group']):
    scan_groups[g] = []

for scan in scans:
    scan_index = df.loc[df['Scan'] == str(scan)].index[0]
    group = df['Group'][scan_index]
    scan_groups[group].append(scan)

print(scan_groups)


cntrl_vili_scans = scan_groups['Con-VILI']
bleo_vili_scans = scan_groups['Ble-VILI']
bleo_scans = scan_groups['Ble']

# Cross-Correlation of features for all scans in each group together 
listOfFeatures = ['volume', 'sphericity', 'surface_area', 'convex_volume', 'equivalent_diameter_area', 'extent', 'solidity']

for i, agrp in enumerate([cntrl_vili_scans, bleo_vili_scans, bleo_scans]):

    data_dict = {}
    for af in listOfFeatures:
        data_value = []
        for aScan in agrp:
            fpath = os.path.join(root_dir, aScan, 'porespy')
            json_files = glob(fpath + '\*' + 'dth12_v2' + '.json')

            # list of featutes in all roi in a scan. example [ median(roi1_volume), median(roi2_volume), .........]
            parameter = [] 
            for json_file in json_files:
                file = open(json_file)
                data = json.load(file)
                file.close()

                extract_feature = data[af]

                if len(extract_feature) > 0:
                    parameter.append(np.median(extract_feature))
            
            # value appended per scan
            data_value.append(np.mean(parameter))

        data_dict[af] = data_value


    data_value = []
    for aScan in agrp:
        fpath = os.path.join(root_dir, aScan, 'porespy')
        json_files = glob(fpath + '\*' + 'dth12' + '.json')

        # list of featutes in all roi in a scan. example [ median(roi1_volume), median(roi2_volume), .........]
        parameter = [] 
        for json_file in json_files:
            file = open(json_file)
            data = json.load(file)
            file.close()
            extract_feature = data['volume']

            if len(extract_feature) > 0:
                parameter.append(utils.ratio_above_threshold(extract_feature, 100000))
        
        # value appended per scan
        data_value.append(np.mean(parameter))

    data_dict['Big blob ratio'] = data_value

    # create a pandas dataframe from the dictionary
    df = pd.DataFrame(data_dict)

    # calculate the correlation matrix
    corr_matrix = df.corr()

    fig = plt.figure(i)
    # plot the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
   
    plt.tight_layout()
    # plt.savefig('heatmap_'+str(i)+'.svg', bbox_inches="tight")
plt.show()



# Cross-Correlation of features for all scans irrespective of group 

listOfFeatures = ['volume', 'sphericity', 'surface_area', 'convex_volume', 'equivalent_diameter_area', 'extent', 'solidity']


data_dict = {}
for af in listOfFeatures:
    data_value = []
    for aScan in (cntrl_vili_scans + bleo_vili_scans + bleo_scans):
        fpath = os.path.join(root_dir, aScan, 'porespy')
        json_files = glob(fpath + '\*' + 'dth12_v2' + '.json')

        # list of featutes in all roi in a scan. example [ median(roi1_volume), median(roi2_volume), .........]
        parameter = [] 
        for json_file in json_files:
            file = open(json_file)
            data = json.load(file)
            file.close()

            extract_feature = data[af]

            if len(extract_feature) > 0:
                parameter.append(np.median(extract_feature))
        
        # value appended per scan
        data_value.append(np.mean(parameter))

    data_dict[af] = data_value


data_value = []
for aScan in (cntrl_vili_scans + bleo_vili_scans + bleo_scans):
    fpath = os.path.join(root_dir, aScan, 'porespy')
    json_files = glob(fpath + '\*' + 'dth12' + '.json')

    # list of featutes in all roi in a scan. example [ median(roi1_volume), median(roi2_volume), .........]
    parameter = [] 
    for json_file in json_files:
        file = open(json_file)
        data = json.load(file)
        file.close()
        extract_feature = data['volume']

        if len(extract_feature) > 0:
            parameter.append(utils.ratio_above_threshold(extract_feature, 100000))
    
    # value appended per scan
    data_value.append(np.mean(parameter))

data_dict['Big blob ratio'] = data_value

# create a pandas dataframe from the dictionary
df = pd.DataFrame(data_dict)

# calculate the correlation matrix
corr_matrix = df.corr()

# plot the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.savefig('heatmap_all_dth12_v2.svg', bbox_inches="tight")
plt.show()