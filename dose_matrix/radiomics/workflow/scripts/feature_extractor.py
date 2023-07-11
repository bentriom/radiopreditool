
import SimpleITK as sitk
import numpy as np
import pandas as pd
import nibabel as nib
import dosesvolumes
import radiomics
# Kind of experimental trick to add my custom doses-volumes stastic
# with pyradiomics engine
features_classes = radiomics.getFeatureClasses()
def newGetFeatureClasses(dict_features_classes):
    if 'dosesvolumes' not in dict_features_classes:
        dict_features_classes["dosesvolumes"] = dosesvolumes.RadiomicsDosesVolumes
    return dict_features_classes
radiomics.getFeatureClasses = lambda : newGetFeatureClasses(features_classes)
radiomics.getFeatureClasses()
from radiomics import featureextractor
import re, os, csv, logging
from functools import reduce
from radiopreditool_utils import get_ctr_numcent, pretty_dosesvol, create_image_mask_example

def write_header(labels_super_t_voi, labels_t_voi, radiomics_dir, params_file):
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    ex_image, ex_mask = create_image_mask_example()
    dict_features_values = extractor.execute(ex_image, ex_mask, label = 1)
    features_name_per_label = [x for x in dict_features_values if not x.startswith("diagnostics_")]
    all_features_names = []
    for label in labels_super_t_voi + labels_t_voi:
        all_features_names += [str(label) + "_" + x for x in features_name_per_label]
    header = ["ctr", "numcent"] +  all_features_names
    header = [pretty_dosesvol(col) for col in header]
    with open(radiomics_dir + "header.csv", "w") as header_file:
        header_writer = csv.writer(header_file, delimiter = ',')
        header_writer.writerow(header)
    with open(radiomics_dir + "nbr_features_per_label", 'w') as f:
        f.write(str(len(features_name_per_label)))
    return len(features_name_per_label)

# Creates a new nifti segmentation (mask) of the dose image for the whole body (label 10000)
def create_whole_body_mask(mask_super_t_path):
    assert os.path.isfile(mask_super_t_path)
    image_mask_super_t = nib.load(mask_super_t_path)
    array_mask_whole_body = image_mask_super_t.get_fdata()
    array_mask_whole_body[array_mask_whole_body != 0] = 10000
    image_mask_whole_body = sitk.GetImageFromArray(np.transpose(array_mask_whole_body))
    image_mask_whole_body.SetSpacing((2.0,2.0,2.0))
    image_mask_whole_body.SetOrigin((0.0,0.0,0.0))
    mask_whole_body_path = mask_super_t_path.replace("mask_super_t.nii.gz", "mask_whole_body.nii.gz")
    assert image_mask_whole_body.GetSize() == image_mask_super_t.shape
    sitk.WriteImage(image_mask_whole_body, mask_whole_body_path)

    return mask_whole_body_path

# Compute dosiomics
# newdosi_filename: string newdosi_(CTR)_(NUMCENT)
def compute_radiomics(image_path, mask_super_t_path, mask_t_path, labels_super_t_voi, labels_t_voi,
                      newdosi_filename, radiomics_dir, subdir, params_file, nbr_features_per_label):
    logger = logging.getLogger("feature_extractor")
    ctr, numcent = get_ctr_numcent(newdosi_filename)
    all_features_values = np.zeros(0)
    # If the images are empty
    if os.path.getsize(image_path) == 0 and os.path.getsize(mask_t_path) == 0 and os.path.getsize(mask_super_t_path) == 0:
        logger.warn(f"{newdosi_filename}: empty nii images. Creating NaN values for the radiomics.")
        all_features_values = [np.nan] * nbr_features_per_label * (len(labels_super_t_voi) + len(labels_t_voi))
    # Else we compute the radiomics
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
        for label in labels_super_t_voi:
            logger.info(f"{newdosi_filename}: Computing label {label}")
            try:
                if label == 10000:
                    mask_whole_body_path = create_whole_body_mask(mask_super_t_path)
                    dict_features_values = extractor.execute(image_path, mask_whole_body_path, label = label)
                    os.remove(mask_whole_body_path)
                else:
                    dict_features_values = extractor.execute(image_path, mask_super_t_path, label = label)
            except ValueError as err:
                logger.warn(f"Raised ValueError. Ignoring the label mask {label}.")
                all_features_values = np.append(all_features_values, [np.nan] * nbr_features_per_label)
                continue
            label_features_values = [dict_features_values[x] for x in dict_features_values if not x.startswith("diagnostics_")]
            all_features_values = np.append(all_features_values, label_features_values)
        for label in labels_t_voi:
            logger.info(f"{newdosi_filename}: Computing label {label}")
            try:
                dict_features_values = extractor.execute(image_path, mask_t_path, label = label)
            except ValueError as err:
                logger.warn(f"Raised ValueError. Ignoring the label mask {label}.")
                all_features_values = np.append(all_features_values, [np.nan] * nbr_features_per_label)
                continue
            label_features_values = [dict_features_values[x] for x in dict_features_values if not x.startswith("diagnostics_")]
            all_features_values = np.append(all_features_values, label_features_values)
    os.makedirs(radiomics_dir + subdir, exist_ok = True)
    with open(radiomics_dir + subdir + "/" + newdosi_filename + "_radiomics.csv", "w") as radiomics_file:
        radiomics_writer = csv.writer(radiomics_file, delimiter = ',')
        radiomics_writer.writerow([ctr, numcent] + list(all_features_values))
    logger.info(f"{newdosi_filename}: End of extraction")

