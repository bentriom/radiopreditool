
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os, sys, logging
from datetime import datetime
from radiopreditool_utils import get_ctr_numcent, get_date, check_nan_values, check_summable_df, col_super_t, setup_logger

# Array preprocessing utils
def process_array_image(doses_array):
    new_array = drop_invalid_range(doses_array)
    
    return new_array

def drop_invalid_range(doses_array):
    D98 = np.nanpercentile(doses_array, 2)  
    D2 = np.nanpercentile(doses_array, 98)
    print(f"d98: {D98} d2: {D2} max: {doses_array.max()}")
    doses_array[doses_array > D2] = D2
    doses_array[doses_array < D98] = D98

    return doses_array

# Save the nifti files
def save_nii(df_dosi, patient_filename, path_nii, save_masks = True, biggest_image_size = None, get_only_size = False):
    logger = logging.getLogger("csv2nii")
    os.makedirs(path_nii, exist_ok=True)
    # Images settings
    file_dosi_nii = path_nii + patient_filename + '_ID2013A.nii.gz'
    file_mask_t_nii = path_nii + patient_filename + '_mask_t.nii.gz'
    file_mask_super_t_nii = path_nii + patient_filename + '_mask_super_t.nii.gz'
    dosi_values = df_dosi.values
    if df_dosi.shape[0] <= 1:
        logger.warning(f"{patient_filename}: df_dosi has <= 1 rows. Creating empty nii files.")
        open(file_dosi_nii, 'w').close()
        open(file_mask_t_nii, 'w').close()
        open(file_mask_super_t_nii, 'w').close()
        return (0, 0, 0)
    elif len(np.unique(dosi_values[dosi_values > 0])) <= 1:
        logger.warning(f"{patient_filename}: df_dosi {df_dosi.shape} has too few values: {np.unique(dosi_values)}. Creating empty nii files.")
        open(file_dosi_nii, 'w').close()
        open(file_mask_t_nii, 'w').close()
        open(file_mask_super_t_nii, 'w').close()
        return (0, 0, 0)
    else:
        # Coordinates, labels and doses as 3D arrays
        x = np.array(df_dosi['X'] - min(df_dosi['X']), dtype='int') // 2
        y = np.array(df_dosi['Y'] - min(df_dosi['Y']), dtype='int') // 2
        z = np.array(df_dosi['Z'] - min(df_dosi['Z']), dtype='int') // 2
        image_size = (max(x)+1,max(y)+1,max(z)+1)
        if get_only_size:
            return image_size
        if biggest_image_size is not None:
            image_size = biggest_image_size
        labels_t_3d = np.zeros(image_size)
        labels_t_3d[x,y,z] = df_dosi['T']
        labels_super_t_3d = np.zeros(image_size)
        labels_super_t_3d[x,y,z] = df_dosi['SUPER_T']
        dosi_3d = np.zeros(image_size)
        dosi_3d[x,y,z] = df_dosi['ID2013A']
        # Preprocesing of array images
        dosi_3d = process_array_image(dosi_3d)
        # Save as images
        image_dosi = sitk.GetImageFromArray(dosi_3d)
        image_dosi.SetSpacing((2.0,2.0,2.0))
        image_dosi.SetOrigin((0.0,0.0,0.0))
        sitk.WriteImage(image_dosi, file_dosi_nii)
        if save_masks:
            image_mask_t = sitk.GetImageFromArray(labels_t_3d)
            image_mask_t.SetSpacing((2.0,2.0,2.0))
            image_mask_t.SetOrigin((0.0,0.0,0.0))
            sitk.WriteImage(image_mask_t, file_mask_t_nii)
            image_mask_super_t = sitk.GetImageFromArray(labels_super_t_3d)
            image_mask_super_t.SetSpacing((2.0,2.0,2.0))
            image_mask_super_t.SetOrigin((0.0,0.0,0.0))
            sitk.WriteImage(image_mask_super_t, file_mask_super_t_nii)

# Requires: list of newdosi files for one patient
# Guarantees: 
def to_nii(path_csv, path_nii, list_csv_files):
    logger = logging.getLogger("csv2nii")
    relevant_cols = ['X', 'Y', 'Z', 'T', 'SUPER_T', 'ID2013A']
    int_cols = ['X', 'Y', 'Z', 'T', 'SUPER_T']
    # Reads the newdosi files and sums the matrices if interval time <= 3 months
    idx_sort_by_date_csv_files = np.argsort([get_date(newdosi_file) for newdosi_file in list_csv_files])
    first_newdosi_file = list_csv_files[idx_sort_by_date_csv_files[0]]
    df_dosi = pd.read_csv(path_csv + first_newdosi_file)
    df_dosi.columns = df_dosi.columns.str.upper()
    test_xyz_nan = check_nan_values(df_dosi)
    if test_xyz_nan:
        nbr_rows_before = df_dosi.shape[0]
        df_dosi = df_dosi.dropna(subset = ['X', 'Y', 'Z', 'ID2013A'])
        nbr_rows_after = df_dosi.shape[0]
        logger.warning(f"{first_newdosi_file}: has NaN values in X, Y or Z. Dropping {nbr_rows_before - nbr_rows_after} rows.")
    col_super_t(df_dosi)
    df_dosi = df_dosi[relevant_cols]
    ctr_patient, numcent_patient = get_ctr_numcent(first_newdosi_file)
    date_last_treatment = get_date(first_newdosi_file)
    for i in idx_sort_by_date_csv_files[1:]:
        current_newdosi_file = list_csv_files[i]
        date_treatment = get_date(current_newdosi_file)
        delta_time = (date_treatment - date_last_treatment)
        # The two RT treatments were made beyond 6 months
        if delta_time.total_seconds() > 6*30*24*3600:
            break
        else:
            df_other_dosi = pd.read_csv(path_csv + current_newdosi_file)
            df_other_dosi.columns = df_other_dosi.columns.str.upper()
            test_xyz_nan = check_nan_values(df_other_dosi)
            if test_xyz_nan:
                nbr_rows_before = df_other_dosi.shape[0]
                df_other_dosi = df_other_dosi.dropna(subset = ['X', 'Y', 'Z', 'ID2013A'])
                nbr_rows_after = df_other_dosi.shape[0]
                logger.warning(f"{current_newdosi_file}: has NaN values in X, Y or Z. Dropping {nbr_rows_before - nbr_rows_after} rows.")
            if df_dosi.shape[0] != df_other_dosi.shape[0]:
                logger.warning(f"{first_newdosi_file} and {current_newdosi_file}: rows numbers are different. Stopping the sum.")
                break
            if df_other_dosi.shape[0] <= 1:
                logger.warning(f"{current_newdosi_file}: has <= 1 rows. Stopping the sum.")
                break
            col_super_t(df_other_dosi)
            df_other_dosi = df_other_dosi[relevant_cols]
            well_ordered_rows = check_summable_df(df_dosi, df_other_dosi)
            if well_ordered_rows:
                df_dosi['ID2013A'] += df_other_dosi['ID2013A']
            else:
                logger.warning(f"{first_newdosi_file} and {current_newdosi_file}: rows are not well ordered.")
                for col in int_cols:
                    df_dosi[col] = df_dosi[col].astype(int)
                    df_other_dosi[col] = df_other_dosi[col].astype(int)
                df_dosi = df_dosi.sort_values(by = int_cols)
                df_other_dosi = df_other_dosi.sort_values(by = int_cols)
                df_other_dosi.index = df_dosi.index
                if not check_summable_df(df_dosi, df_other_dosi):
                    logger.warning(f"{first_newdosi_file} and {current_newdosi_file}: same rows number but different. Stopping the sum.")
                    break
                df_dosi['ID2013A'] += df_other_dosi['ID2013A']
        #date_last_treatment = date_treatment
    patient_filename = f"newdosi_{ctr_patient}_{numcent_patient}"

    os.makedirs(path_nii, exist_ok=True)
    save_nii(df_dosi, patient_filename, path_nii) 

