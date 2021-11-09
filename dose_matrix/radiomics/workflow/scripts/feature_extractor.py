
import SimpleITK as sitk
import numpy as np
import pandas as pd
import radiomics
import csv
import os
from radiomics import featureextractor
from functools import reduce
from csv2nii import get_ctr_numcent

all_features_name = []
features_name_per_label = []

def create_image_mask_example():
    array_image = np.zeros((32,32,32)) 
    array_mask = np.zeros((32,32,32))
    xx, yy, zz = np.meshgrid(np.arange(10,20), np.arange(10,20), np.arange(10,20))
    x, y, z = xx.flatten(), yy.flatten(), zz.flatten()
    array_image[x, y, z] = np.random.rand(10**3)
    array_mask[x, y, z] = 1
    image = sitk.GetImageFromArray(array_image)
    mask = sitk.GetImageFromArray(array_mask)
    return image, mask

def write_header(voi_type, labels_voi, radiomics_dir, params_file):
    global all_features_name, features_name_per_label
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    ex_image, ex_mask = create_image_mask_example()
    dict_features_values = extractor.execute(ex_image, ex_mask, label = 1)
    features_name_per_label = [x for x in dict_features_values if not x.startswith("diagnostics_")]
    print(features_name_per_label)
    for label in labels_voi:
        all_features_name += [str(label) + "_" + x for x in features_name_per_label]
    print(all_features_name)
    header = ["ctr", "numcent", "newdosi_filename"] +  all_features_name
    with open(radiomics_dir + "header.csv", "w") as header_file:
        header_writer = csv.writer(header_file, delimiter = ',')
        header_writer.writerow(header)

def compute_radiomics(image_path, mask_path, voi_type, labels_voi, newdosi_filename, radiomics_dir, subdir, params_file):
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    ctr, numcent = get_ctr_numcent(newdosi_filename)
    nbr_features_per_label = len(features_name_per_label)
    all_features_values = np.zeros(0)
    for label in labels_voi:
        try:
            dict_features_values = extractor.execute(image_path, mask_path, label = label)
        except ValueError as err:
            print(f"Raised ValueError \n({err=})\nIgnoring the label mask")
            all_features_values = np.append(all_features_values, [np.nan] * nbr_features_per_label)
            continue
        label_features_values = [dict_features_values[x] for x in features_name_per_label]
        all_features_values = np.append(all_features_values, label_features_values)
    os.makedirs(radiomics_dir + subdir, exist_ok = True)
    with open(radiomics_dir + subdir + "/" + newdosi_filename + "_radiomics_" + voi_type + ".csv", "w") as radiomics_file:
        radiomics_writer = csv.writer(radiomics_file, delimiter = ',')
        radiomics_writer.writerow([ctr, numcent, newdosi_filename] + list(all_features_values))

