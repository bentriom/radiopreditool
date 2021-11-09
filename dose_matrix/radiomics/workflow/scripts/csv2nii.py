
import numpy as np
import pandas as pd
#import matplotlib.pylab as plt
import SimpleITK as sitk
import os
import math as m
import sys
# nii.gz or nrrd format ?

#DATASET_DIR = sys.argv[1]
#NII_DIR = sys.argv[2]

def get_ctr_numcent(dosi_filename):    
    split_filename = dosi_filename.split("_")
    ctr_patient = int(split_filename[1])
    str_numcent_patient = split_filename[2].split(".")[0]
    str_numcent_patient = str_numcent_patient[0:-1] if str_numcent_patient[-1].isalpha() else str_numcent_patient
    numcent_patient = int(str_numcent_patient)
    return ctr_patient, numcent_patient

def to_nii(path_csv, path_nii, list_csv_files, voi_type):
    assert voi_type in ['T', 'NUAGE', 'MATH']
    first_newdosi_file = list_csv_files[0]
    df_dosi = pd.read_csv(path_csv + first_newdosi_file)
    df_dosi.columns = df_dosi.columns.str.upper()
    for i in range(1,len(list_csv_files)):
        df_other_dosi = pdf.read_csv(path_csv + list_csv_files[i])
        df_other_dosi.columns = df_other_dosi.columns.str.upper()
        df_dosi += df_other_dosi
    ctr_patient, numcent_patient = get_ctr_numcent(first_newdosi_file)
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

#list_subdirs = ["Curie", "GR"]
#for subdir in list_subdirs:
#    path_csv = DATASET_DIR + subdir + "/"
#    path_nii = NII_DIR + subdir + "/"
#    for file_csv in os.listdir(path_csv):
#        print(file_csv)
#        to_nii(path_csv, path_nii, file_csv)

