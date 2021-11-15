
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
    # Date is missing for some files
    if str_date_treatment == "00000000":
        date_treatment = datetime.strptime("19700101", "%Y%m%d")
    else:    
        date_treatment = datetime.strptime(str_date_treatment, "%Y%m%d")
    return date_treatment

def check_nan_values(df):
    return (df['X'].isnull().values.any() or df['Y'].isnull().values.any() or df['Z'].isnull().values.any() or df['ID2013A'].isnull().values.any())

def to_nii(path_csv, path_nii, list_csv_files, voi_type):
    voi_type = voi_type.upper()
    assert voi_type in ['T', 'NUAGE', 'MATH']
    relevant_cols = ['X', 'Y', 'Z', voi_type, 'ID2013A']
    int_cols = ['X', 'Y', 'Z', voi_type]
    # Reads the newdosi files and sums the matrices if interval time <= 3 months
    idx_sort_by_date_csv_files = np.argsort([get_date(newdosi_file) for newdosi_file in list_csv_files])
    first_newdosi_file = list_csv_files[idx_sort_by_date_csv_files[0]]
    df_dosi = pd.read_csv(path_csv + first_newdosi_file)
    df_dosi.columns = df_dosi.columns.str.upper()
    df_dosi = df_dosi[relevant_cols]
    test_xyz_nan = check_nan_values(df_dosi)
    if test_xyz_nan:
        nbr_rows_before = df_dosi.shape[0]
        df_dosi = df_dosi.dropna(subset = ['X', 'Y', 'Z', 'ID2013A'])
        nbr_rows_after = df_dosi.shape[0]
        print(f"{first_newdosi_file}: has NaN values in X, Y or Z. Dropping {nbr_rows_before - nbr_rows_after} rows.")
    ctr_patient, numcent_patient = get_ctr_numcent(first_newdosi_file)
    date_last_treatment = get_date(first_newdosi_file)
    for i in idx_sort_by_date_csv_files[1:]:
        current_newdosi_file = list_csv_files[i]
        date_treatment = get_date(current_newdosi_file)
        delta_time = (date_treatment - date_last_treatment)
        # The two RT treatments were made beyond 3 months
        if delta_time.total_seconds() > 3*30*24*3600:
            break
        else:
            df_other_dosi = pd.read_csv(path_csv + current_newdosi_file)
            df_other_dosi.columns = df_other_dosi.columns.str.upper()
            df_other_dosi = df_other_dosi[relevant_cols]
            test_xyz_nan = check_nan_values(df_other_dosi)
            if test_xyz_nan:
                nbr_rows_before = df_other_dosi.shape[0]
                df_other_dosi = df_other_dosi.dropna(subset = ['X', 'Y', 'Z', 'ID2013A'])
                nbr_rows_after = df_other_dosi.shape[0]
                print(f"{current_newdosi_file}: has NaN values in X, Y or Z. Dropping {nbr_rows_before - nbr_rows_after} rows.")
            if df_dosi.shape[0] != df_other_dosi.shape[0]:
                print(f"{first_newdosi_file} and {current_newdosi_file}: rows numbers are different. Stopping the sum.")
                break
            if df_other_dosi.shape[0] <= 1:
                print(f"{current_newdosi_file}: has <= 1 rows. Stopping the sum.")
                break
            well_ordered_rows = df_dosi['X'].equals(df_other_dosi['X']) and \
                                df_dosi['Y'].equals(df_other_dosi['Y']) and \
                                df_dosi['Z'].equals(df_other_dosi['Z']) and \
                                df_dosi[voi_type].equals(df_other_dosi[voi_type])
            if well_ordered_rows:
                df_dosi['ID2013A'] += df_other_dosi['ID2013A']
            else:
                print(f"{first_newdosi_file} and {current_newdosi_file}: rows are not well ordered.")
                for col in int_cols:
                    df_dosi[col] = df_dosi[col].astype(int)
                    df_other_dosi[col] = df_other_dosi[col].astype(int)
                df_dosi = df_dosi.sort_values(by = int_cols)
                df_other_dosi = df_other_dosi.sort_values(by = int_cols)
                df_other_dosi.index = df_dosi.index
                if not (df_dosi['X'].equals(df_other_dosi['X']) and \
                        df_dosi['Y'].equals(df_other_dosi['Y']) and \
                        df_dosi['Z'].equals(df_other_dosi['Z']) and \
                        df_dosi[voi_type].equals(df_other_dosi[voi_type])):
                    print(f"{first_newdosi_file} and {current_newdosi_file}: same rows number but different. Stopping the sum.")
                    break
                df_dosi['ID2013A'] += df_other_dosi['ID2013A']
        #date_last_treatment = date_treatment
    patient_filename = f"newdosi_{ctr_patient}_{numcent_patient}"

    # Images settings
    os.makedirs(path_nii, exist_ok=True)
    file_dosi_nii = path_nii + patient_filename + '_ID2013A.nii.gz'
    file_mask_nii = path_nii + patient_filename + f"_{voi_type}.nii.gz"
    
    # If the first newdosi file was empty
    if df_dosi.shape[0] <= 1:
        print(f"{first_newdosi_file}: has <= 1 rows. Creating empty nii files.")
        open(file_dosi_nii, 'w').close()
        open(file_mask_nii, 'w').close()
    else:
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
        image_dosi = sitk.GetImageFromArray(dosi_3d)
        image_dosi.SetSpacing((2.0,2.0,2.0))
        image_dosi.SetOrigin((0.0,0.0,0.0))
        sitk.WriteImage(image_dosi, file_dosi_nii)
        image_mask = sitk.GetImageFromArray(labels_3d)
        image_mask.SetSpacing((2.0,2.0,2.0))
        image_mask.SetOrigin((0.0,0.0,0.0))
        sitk.WriteImage(image_mask, file_mask_nii)

