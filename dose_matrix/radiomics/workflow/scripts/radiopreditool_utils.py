
import logging
import numpy as np
import pandas as pd
import scipy.optimize as opt
import SimpleITK as sitk
import os, re
from multiprocessing import cpu_count
from datetime import datetime
import numpy as np
import numpy.random as rand

# Generates survival data with controlled censorship
def generate_survival_times(n_samples, n_covariates, betas, frac_censor = 0.8, censor_type = "type_I",
                            mat_cov = None, distribution = "exp"):
    assert n_covariates == len(betas)
    assert distribution in ["exp"]
    assert censor_type in ["type_I", "random"]
    # Simulate covariates
    if mat_cov is None:
        X = rand.randn(n_covariates, n_samples)
    else:
        assert mat_cov.shape == (n_covariates, n_covariates)
        X = rand.multivariate_normal(cov = mat_cov, size = n_samples)
    assert X.shape == (n_covariates, n_samples)
    if distribution == "exp":
        inv_cumbasehazard = lambda x:  (x/0.025)
    betas = np.asarray(betas)
    # Simulate survival times: T = inv_cumbasehazard(-ln(U) * exp(-beta' * X)) with U uniform
    if distribution == "exp":
        mean_base_hazard = 10
        risks_samples = np.exp(np.dot(betas, X))
        means_surv_time = 1 / ((1/mean_base_hazard) * risks_samples) # 1/lambda_Ts
        times_event = rand.exponential(means_surv_time)
        if censor_type == "random":
            assert frac_censor >= 0.5, "Censor time modeled by an exponential and P(censoring) < 0.5"
            means_exp_censor = 1 / (-(1/means_surv_time) * np.log(2*(1-frac_censor))) # 1/lambda_Cs
            difference_times_event_censor = rand.laplace(means_surv_time, means_exp_censor) # T - C
            status = 1 * (difference_times_event_censor <= 0)
            times_censor = np.maximum(times_event - difference_times_event_censor, 0)
            times_observed = np.minimum(times_event, times_censor)
        elif censor_type == "type_I":
            study_end = opt.minimize_scalar(lambda t: (np.sum(times_event<=t)/len(times_event)-(1-frac_censor))**2,
                                            bounds = (0, max(times_event)), method = "bounded").x
            times_observed = times_event
            status = 1 * (times_event <= study_end)
            times_observed[times_observed > study_end] = study_end

    return times_observed, status, X

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
    labels_heart = range(320, 325) # 1320
    labels_brain = range(370, 381) # 1370
    labels_thyroid = range(702, 705) # 1702
    labels_breast_right = [413, 415, 417, 419] # 2413
    labels_breast_left = [414, 416, 418, 420] # 3413
    if pd.isnull(label_t):
        return np.nan
    elif label_t in labels_heart:
        return 1320
    elif label_t in labels_brain:
        return 1370
    elif label_t in labels_thyroid:
        return 1702
    elif label_t in labels_breast_right:
        return 2413
    elif label_t in labels_breast_left:
        return 3413
    else:
        return 1000

def get_super_t_active_marrow_subset(label_t):
    labels_heart = range(320, 325) # 1320
    labels_brain = range(370, 381) # 1370
    labels_thyroid = range(702, 705) # 1702
    labels_breast_right = [413, 415, 417, 419] # 2413
    labels_breast_left = [414, 416, 418, 420] # 3413
    labels_active_marrow_subset = [605, 607, 608] # 2601
    if pd.isnull(label_t):
        return np.nan
    elif label_t in labels_heart:
        return 1320
    elif label_t in labels_brain:
        return 1370
    elif label_t in labels_thyroid:
        return 1702
    elif label_t in labels_breast_right:
        return 2413
    elif label_t in labels_breast_left:
        return 3413
    elif label_t in labels_active_marrow_subset:
        return 2601
    else:
        return 1000

def col_super_t(df_dosi, name_super_t_func):
    # get_super_t_func = getattr(radiopreditool_utils, name_super_t_func)
    get_super_t_func = globals()[name_super_t_func]
    df_dosi['SUPER_T'] = df_dosi['T'].astype(int).apply(get_super_t_func)

def get_clinical_features(df_dataset, event_col, duration_col):
    regex = "^(([0-9]{3,4}_)|(dv_)|" + f"({event_col})|({duration_col})|(ctr)|(numcent)|(has_radiomics))"
    return [col for col in df_dataset.columns if not re.match(regex, col)]

def get_all_radiomics_features(df_dataset):
    return [col for col in df_dataset.columns if re.match("^[0-9]{3,4}_", col)]

def get_t_radiomics_features(df_dataset):
    return [col for col in df_dataset.columns if re.match("^[0-9]{3}_", col)]

def get_super_t_radiomics_features(df_dataset):
    return [col for col in df_dataset.columns if re.match("^[0-9]{4}_", col)]

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

def is_slurm_run():
    return ("SLURM_CPUS_PER_TASK" in os.environ) or ("SLURM_NTASKS" in os.environ)

def get_ncpus():
    if is_slurm_run():
        return 40
    elif "LOCAL_SNAKEMAKE_NCPUS" in os.environ:
        return int(os.environ["LOCAL_SNAKEMAKE_NCPUS"])
    else:
        return cpu_count()

