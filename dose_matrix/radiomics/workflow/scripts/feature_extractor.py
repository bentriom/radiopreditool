
import SimpleITK as sitk
import numpy as np
import pandas as pd
import radiomics
import csv
import os
from radiomics import featureextractor
from functools import reduce
from csv2nii import get_ctr_numcent

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
firstorder_features = ["Mean", "TotalEnergy"]
extractor.enableFeaturesByName(firstorder = firstorder_features)
features_name = ["original_firstorder_" + x for x in firstorder_features]

def write_header(voi_type, labels_voi, radiomics_dir):
    all_features_name = []
    for label in labels_voi:
        all_features_name += [str(label) + "_" + x for x in features_name]
    header = ["ctr", "numcent", "newdosi_filename"] +  all_features_name
    with open(radiomics_dir + "header.csv", "w") as header_file:
        header_writer = csv.writer(header_file, delimiter = ',')
        header_writer.writerow(header)

def compute_radiomics(image_path, mask_path, voi_type, labels_voi, newdosi_filename, radiomics_dir, subdir):
    ctr, numcent = get_ctr_numcent(newdosi_filename)
    nbr_features_per_label = len(features_name)
    all_features_values = np.zeros(0)
    for label in labels_voi:
        try:
            dict_features_values = extractor.execute(image_path, mask_path, label = label)
        except ValueError as err:
            print(f"Raised ValueError \n({err=})\nIgnoring the label mask")
            all_features_values = np.append(all_features_values, [np.nan] * nbr_features_per_label)
            continue
        label_features_values = [dict_features_values[x] for x in features_name]
        all_features_values = np.append(all_features_values, label_features_values)
    os.makedirs(radiomics_dir + subdir, exist_ok = True)
    with open(radiomics_dir + subdir + "/" + newdosi_filename + "_radiomics_" + voi_type + ".csv", "w") as radiomics_file:
        radiomics_writer = csv.writer(radiomics_file, delimiter = ',')
        radiomics_writer.writerow([ctr, numcent, newdosi_filename] + list(all_features_values))

