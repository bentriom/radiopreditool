
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import math as m
import sys
from datetime import datetime

def get_ctr_numcent(dosi_filename):    
    split_filename = dosi_filename.split("_")
    ctr_patient = int(split_filename[1])
    str_numcent_patient = split_filename[2].split(".")[0]
    str_numcent_patient = str_numcent_patient[0:-1] if str_numcent_patient[-1].isalpha() else str_numcent_patient
    numcent_patient = int(str_numcent_patient)
    return ctr_patient, numcent_patient

def get_date(dosi_filename):    
    split_filename = dosi_filename.split("_")
    str_date_treatment = split_filename[3].split(".")[0]
    date_treatment = datetime.strptime(str_date_treatment, "%Y%m%d")
    return date_treatment

def to_nii(path_csv, path_nii, list_csv_files, voi_type):
    assert voi_type in ['T', 'NUAGE', 'MATH']
    # Reads the newdosi files and sums the matrices if interval time <= 3 months
    idx_sort_by_date_csv_files = np.argsort([get_date(newdosi_file) for newdosi_file in list_csv_files])
    first_newdosi_file = list_csv_files[idx_sort_by_date_csv_files[0]]
    df_dosi = pd.read_csv(path_csv + first_newdosi_file)
    df_dosi.columns = df_dosi.columns.str.upper()
    ctr_patient, numcent_patient = get_ctr_numcent(first_newdosi_file)
    date_last_treatment = get_date(first_newdosi_file)
    for i in idx_sort_by_date_csv_files[1:]:
        date_treatment = get_date(list_csv_files[i])
        delta_time = (date_treatment - date_last_treatment)
        # The two RT treatments were made within 3 months
        if delta_time.total_seconds() <= 3*30*24*3600:
            df_other_dosi = pd.read_csv(path_csv + list_csv_files[i])
            df_other_dosi.columns = df_other_dosi.columns.str.upper()
            assert df_dosi['X'].equals(df_other_dosi['X']) and \
                   df_dosi['Y'].equals(df_other_dosi['Y']) and \
                   df_dosi['Z'].equals(df_other_dosi['Z'])
            df_dosi['ID2013A'] += df_other_dosi['ID2013A']
        date_last_treatment = date_treatment
    patient_filename = f"newdosi_{ctr_patient}_{numcent_patient}"

    # Coordinates, labels and doses as 3D arrays
    x = np.array(df_dosi['X'] - min(df_dosi['X']), dtype='int') // 2
    y = np.array(df_dosi['Y'] - min(df_dosi['Y']), dtype='int') // 2
    z = np.array(df_dosi['Z'] - min(df_dosi['Z']), dtype='int') // 2
    image_size = (max(x)+1,max(y)+1,max(z)+1)
    labels_3d = np.zeros(image_size)
    labels_3d[x,y,z] = df_dosi[voi_type]
    dosi_3d = np.zeros(image_size)
    dosi_3d[x,y,z] = df_dosi['ID2013A']

    # Save as images
    os.makedirs(path_nii, exist_ok=True)
    file_dosi_nii = path_nii + patient_filename + '_ID2013A.nii.gz'
    image_dosi = sitk.GetImageFromArray(dosi_3d)
    image_dosi.SetSpacing((2.0,2.0,2.0))
    image_dosi.SetOrigin((0.0,0.0,0.0))
    sitk.WriteImage(image_dosi, file_dosi_nii)
    file_mask_nii = path_nii + patient_filename + f"_{voi_type}.nii.gz"
    image_mask = sitk.GetImageFromArray(labels_3d)
    image_mask.SetSpacing((2.0,2.0,2.0))
    image_mask.SetOrigin((0.0,0.0,0.0))
    sitk.WriteImage(image_mask, file_mask_nii)

