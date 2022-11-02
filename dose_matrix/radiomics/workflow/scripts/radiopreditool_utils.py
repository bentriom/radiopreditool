
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os, re
from multiprocessing import cpu_count
from datetime import datetime

# Snakefile tools
def addslash(subdir):
    return (subdir if subdir == "" else subdir + "/")

def get_patient_file(newdosi_file):
    split_newdosi = newdosi_file.split("_")
    return split_newdosi[0] + "_" + split_newdosi[1] + "_" + split_newdosi[2][0:-1]

# Create empty image
def create_image_mask_example(zero_img = False, num_mask = 1):
    array_image = np.zeros((32,32,32))
    array_mask = np.zeros((32,32,32))
    xx, yy, zz = np.meshgrid(np.arange(10,20), np.arange(10,20), np.arange(10,20))
    x, y, z = xx.flatten(), yy.flatten(), zz.flatten()
    if not zero_img:
        array_image[x, y, z] = np.random.rand(10**3)
    array_mask[x, y, z] = num_mask
    image = sitk.GetImageFromArray(array_image)
    image.SetSpacing((2.0,2.0,2.0))
    image.SetOrigin((0.0,0.0,0.0))
    mask = sitk.GetImageFromArray(array_mask)
    mask.SetSpacing((2.0,2.0,2.0))
    mask.SetOrigin((0.0,0.0,0.0))
    return image, mask

# Data fccss specific 
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
        date_treatment = datetime.strptime("19000101", "%Y%m%d")
    else:    
        date_treatment = datetime.strptime(str_date_treatment, "%Y%m%d")
    return date_treatment

def check_nan_values(df_dosi):
    return (df_dosi['X'].isnull().values.any() or df_dosi['Y'].isnull().values.any() or \
            df_dosi['Z'].isnull().values.any() or df_dosi['ID2013A'].isnull().values.any())

def check_summable_df(df_dosi_A, df_dosi_B, voi_type = 'T'):
    return (df_dosi_A['X'].equals(df_dosi_B['X']) and df_dosi_A['Y'].equals(df_dosi_B['Y']) and \
            df_dosi_A['Z'].equals(df_dosi_B['Z']) and df_dosi_A[voi_type].equals(df_dosi_B[voi_type]))

# Labels T
def get_super_t(label_t):
    labels_heart = range(320, 325)
    labels_brain = range(370, 381)
    labels_thyroid = range(702, 705)
    labels_breast_right = [413, 415, 417, 419]
    labels_breast_left = [414, 416, 418, 420]
    labels_breasts = labels_breast_right + labels_breast_left
    if pd.isnull(label_t):
        return np.nan
    elif label_t in labels_heart:
        return 1320
    elif label_t in labels_brain:
        return 1370
    elif label_t in labels_thyroid:
        return 1702
    elif label_t in labels_breasts:
        return 1410
    elif label_t in labels_breast_right:
        return 1411
    elif label_t in labels_breast_left:
        return 1412
    else:
        return 1000

def col_super_t(df_dosi):
    df_dosi['SUPER_T'] = df_dosi['T'].astype(int).apply(get_super_t)

def get_clinical_features(df_dataset, event_col, duration_col):
    regex = "^(([0-9]{3,4}_)|(dv_)|" + f"({event_col})|({duration_col})|(ctr)|(numcent)|(has_radiomics))"
    return [col for col in df_dataset.columns if not re.match(regex, col)]

def get_all_radiomics_features(df_dataset):
    return [col for col in df_dataset.columns if re.match("^[0-9]{3,4}_", col)]

def get_t_radiomics_features(df_dataset):
    return [col for col in df_dataset.columns if re.match("^[0-9]{3}_", col)]

def get_super_t_radiomics_features(df_dataset):
    return [col for col in df_dataset.columns if re.match("^1[0-9]{3}_", col)]

def get_labels_t(df_dataset):
    features = get_t_radiomics_features(df_dataset)
    return np.unique([f.split("_")[0] for f in features]).tolist()

def get_labels_super_t(df_dataset):
    features = get_super_t_radiomics_features(df_dataset)
    return np.unique([f.split("_")[0] for f in features]).tolist()

def get_all_labels(df_dataset):
    return get_labels_super_t(df_dataset) + get_labels_t(df_dataset)

def get_all_dosesvol_features(df_dataset):
    return [col for col in df_dataset.columns if re.match("dv_\w+_[0-9]{3,4}", col)]

def pretty_dosesvol(label):
    matches = re.match("([0-9]{3,4})_original_dosesvolumes_(\w+)", label)
    return "dv_" + matches[2] + "_" + matches[1] if bool(matches) else label

def pretty_label(label):
    matches = re.match("([0-9]{3,4})_[a-z]+_[a-z]+_(\w+)", label)
    return matches[1] + " " + matches[2] if bool(matches) else label

def pretty_labels(labels):
    return [pretty_label(label) for label in labels]

# Sksurv utils
def get_events(structured_y):
    return [event[0] for event in structured_y]

def get_times(structured_y):
    return [event[1] for event in structured_y]

def event_balance(structured_y):
    y = get_events(structured_y)
    counts = np.bincount(y)
    return counts[1]/sum(counts)

# Log
def setup_logger(name, log_file, level = logging.INFO, mode_file = "w", creation_msg = True):
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt = '%m/%d/%Y %H:%M:%S')
    handler = logging.FileHandler(log_file, mode = mode_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    if creation_msg:
        logger.info(f"Logger {name} created at {datetime.now()}")
    return logger

def get_ncpus():
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    elif "LOCAL_SNAKEMAKE_NCPUS" in os.environ:
        return int(os.environ["LOCAL_SNAKEMAKE_NCPUS"])
    else:
        return cpu_count()

