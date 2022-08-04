
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
from radiopreditool_utils import addslash, get_patient_file, get_ncpus
from scipy.stats import entropy
from functools import partial
from multiprocessing import Pool, Array

def get_doses_array(doses_nii_file, mask_nii_file, mask_label):
    doses = sitk.GetArrayFromImage(sitk.ReadImage(doses_nii_file))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_nii_file))
    assert mask_label in mask

    return doses[mask == mask_label]

def get_entropy(newdosi_patient, nii_dir, list_binwidth, list_rules):
    # Counter
    global counter_entropy
    with counter_entropy.get_lock():
        counter_entropy[0] += 1
    if np.floor(counter_entropy[0] / counter_entropy[1] * 10) != \
    np.floor((counter_entropy[0]-1) / counter_entropy[1] * 10):
        print(f"{int(np.floor(counter_entropy[0] / counter_entropy[1] * 10))}0%")
    # If the images are empty
    ctr, numcent = os.path.basename(newdosi_patient).split('_')[1:3]
    if os.path.getsize(nii_dir + newdosi_patient + "_ID2013A.nii.gz") == 0:
        return [ctr, numcent] + [np.nan] * len(list_binwidth)
    doses = get_doses_array(nii_dir + newdosi_patient + "_ID2013A.nii.gz", \
                            nii_dir + newdosi_patient + "_mask_super_t.nii.gz", 1320) 
    list_entropy = []
    for binwidth in list_binwidth:
        nbr_bins = np.max(doses) // binwidth
        bins = np.arange(0, np.max(doses) + binwidth, binwidth)
        hist_counts = np.histogram(doses, bins = bins)[0]
        list_entropy.append(entropy(hist_counts, base = 2))
    for rule in list_rules:
        nbr_bins = len(np.histogram_bin_edges(doses, bins = rule)) - 1
        list_entropy.append(nbr_bins)
    for rule in list_rules:
        nbr_bins = len(np.histogram_bin_edges(doses, bins = rule)) - 1
        binwidth = np.max(doses) / nbr_bins
        list_entropy.append(binwidth)

    return [ctr, numcent] + list_entropy

def compute_entropy(doses_dataset_subdirs, nii_dir, metadata_dir):
    # Create useful variables for rules based on configuration
    list_newdosi_files = [addslash(subdir) + f.split(".")[0]  for subdir in doses_dataset_subdirs \
                          for f in os.listdir(nii_dir + subdir) if ".nii.gz" in f]
    list_newdosi_patients = list(set(['_'.join(newdosi.split('_')[0:3]) for newdosi in list_newdosi_files]))
    list_binwidth = [0.1, 0.5, 1.0, 1.5, 2.5, 5.0]
    list_rules = ["auto", "fd", "sturges", "doane"]
    global counter_entropy
    counter_entropy = Array('i', [0, len(list_newdosi_patients)])
    with Pool(get_ncpus()-2) as p:
        results = p.map(partial(get_entropy, nii_dir = nii_dir, \
                                list_binwidth = list_binwidth, list_rules = list_rules), \
                        list_newdosi_patients)
    cols_entropy = [f"entropy_binwidth_{binw}" for binw in list_binwidth]
    cols_rules = [f"length_{rule}" for rule in list_rules] + \
                 [f"binwidth_{rule}" for rule in list_rules]
    pd.DataFrame(results, columns = ["ctr", "numcent"] + cols_entropy + cols_rules).to_csv(metadata_dir + "entropy_newdosi.csv.gz", index = None, encoding = 'utf-8')

