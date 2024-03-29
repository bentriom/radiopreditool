
import pandas as pd
import numpy as np
import os, logging, tqdm
from multiprocessing import Pool, Value, cpu_count
from joblib import Parallel, delayed
from radiopreditool_utils import *
from functools import partial

class Checker(object):
    def __init__(self, doses_dataset_dir, n_patients, ncalls):
        self.doses_dataset_dir = doses_dataset_dir
        self.n_patients = n_patients
        self.ncalls = ncalls
    def __call__(self, df):
        self.ncalls.value += 1
        return check_files_patient(df, self.doses_dataset_dir)

def print_progress(value, total):
    threshold_patients = self.n_patients * 0.01
    print(self.ncalls.value)
    if ((self.ncalls.value-1 // threshold_patients) < (self.ncalls.value // threshold_patients)):
        print(f"{self.ncalls.value // threshold_patients}%")

def check_files_patient(df_files_patient, doses_dataset_dir):
    logger = logging.getLogger("check_dataset")
    relevant_cols = ['X', 'Y', 'Z', 'T', 'NUAGE', 'ID2013A']
    int_cols = ['X', 'Y', 'Z', 'T']
    labels_t = [301, 304, 305, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
                369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380,
                410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 425, 426,
                500, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619,
                702, 703, 704]
    labels_nuage = [359, 360]
    df_result = df_files_patient.copy().reset_index()
    cols_labels_t = [f"count_{label}" for label in labels_t]
    cols_labels_nuage = [f"count_{label}" for label in labels_nuage]
    df_result[["nbr_nan_rows", "remaining_rows", "missing_date", "outdated_treatment", "different_shapes",
               "size_x", "size_y", "size_z"]] = 0
    df_result[["well_ordered_rows", "summable"]] = 1
    df_result[cols_labels_t] = 0
    idx_sort_by_date_csv_files = np.argsort([get_date(newdosi_file) \
                                             for newdosi_file in df_result["filename_dose_matrix"]])
    df_result = df_result.iloc[list(idx_sort_by_date_csv_files)]
    path_csv = doses_dataset_dir + df_result.at[0, "localisation"] + "/"
    first_newdosi_file = df_result.at[0, "filename_dose_matrix"]
    df_dosi = pd.read_csv(path_csv + first_newdosi_file)
    df_dosi.columns = df_dosi.columns.str.upper()
    df_dosi = df_dosi[relevant_cols]
    # Check NaN values
    test_xyz_nan = check_nan_values(df_dosi)
    df_result.at[0, "remaining_rows"] = df_dosi.shape[0]
    if test_xyz_nan:
        nbr_rows_before = df_dosi.shape[0]
        df_dosi = df_dosi.dropna(subset = ['X', 'Y', 'Z', 'ID2013A'])
        nbr_rows_after = df_dosi.shape[0]
        nbr_nan_rows = nbr_rows_before - nbr_rows_after
        logger.info(f"{first_newdosi_file}: {nbr_nan_rows} => {nbr_rows_after} rows")
        df_result.loc[0, ["nbr_nan_rows", "remaining_rows"]] = [nbr_nan_rows, nbr_rows_after]
    # Check if missing date
    date_last_treatment = get_date(first_newdosi_file)
    if date_last_treatment == datetime.strptime("19000101", "%Y%m%d"):
        df_result.at[0, "missing_date"] = 1
    # Check if the set of voxels of each T/nuage label is non-empty
    count_labels_t = [df_dosi['T'].isin([label]).sum() for label in labels_t]
    df_result.loc[0, cols_labels_t] = count_labels_t
    count_labels_nuage = [df_dosi['NUAGE'].isin([label]).sum() for label in labels_nuage]
    df_result.loc[0, cols_labels_nuage] = count_labels_nuage
    # Check is nipples nuage has other T than 500
    df_result.loc[0, "359_not_T_500"] = (df_dosi.loc[df_dosi["NUAGE"] == 359, "T"] != 500).sum()
    df_result.loc[0, "360_not_T_500"] = (df_dosi.loc[df_dosi["NUAGE"] == 360, "T"] != 500).sum()
    # Get the dose image size
    if df_result.loc[0, "remaining_rows"] > 1:
        x = np.array(df_dosi['X'] - min(df_dosi['X']), dtype='int') // 2
        y = np.array(df_dosi['Y'] - min(df_dosi['Y']), dtype='int') // 2
        z = np.array(df_dosi['Z'] - min(df_dosi['Z']), dtype='int') // 2
        df_result.loc[0, ["size_x", "size_y", "size_z"]] = [max(x)+1,max(y)+1,max(z)+1]

    # Iterate over the other files
    for i in range(1, df_result.shape[0]):
        current_newdosi_file = df_result.loc[i, "filename_dose_matrix"]
        df_other_dosi = pd.read_csv(path_csv + current_newdosi_file)
        df_other_dosi.columns = df_other_dosi.columns.str.upper()
        df_other_dosi = df_other_dosi[relevant_cols]
        date_treatment = get_date(current_newdosi_file)
        delta_time = (date_treatment - date_last_treatment)
        # Check the date from the first treatment
        if date_treatment == datetime.strptime("19000101", "%Y%m%d"):
            df_result.at[0, "missing_date"] = 1
        if delta_time.total_seconds() > 6*30*24*3600:
            df_result.at[i, "outdated_treatment"] = 1
        # Check NaN values
        test_xyz_nan = check_nan_values(df_other_dosi)
        df_result.at[i, "remaining_rows"] = df_other_dosi.shape[0]
        if test_xyz_nan:
            nbr_rows_before = df_other_dosi.shape[0]
            df_other_dosi = df_other_dosi.dropna(subset = ['X', 'Y', 'Z', 'ID2013A'])
            nbr_rows_after = df_other_dosi.shape[0]
            nbr_nan_rows = nbr_rows_before - nbr_rows_after
            logger.info(f"{current_newdosi_file}: {nbr_nan_rows} => {nbr_rows_after} rows")
            df_result.loc[i, ["nbr_nan_rows", "remaining_rows"]] = [nbr_nan_rows, nbr_rows_after]
        # Check the dimensions between the two df
        if df_dosi.shape[0] != df_other_dosi.shape[0]:
            df_result.at[i, "different_shapes"] = 1
            df_result.at[i, "well_ordered_rows"] = 0
            df_result.at[i, "summable"] = 0
            continue
        # Check if rows are well ordered
        well_ordered_rows = check_summable_df(df_dosi, df_other_dosi)
        if well_ordered_rows:
            df_result.at[i, "well_ordered_rows"] = 1
        # Check if the two dataframes are summmable after sorting the rows
        else:
            for col in int_cols:
                df_dosi[col] = df_dosi[col].astype(int)
                df_other_dosi[col] = df_other_dosi[col].astype(int)
            df_dosi = df_dosi.sort_values(by = int_cols)
            df_other_dosi = df_other_dosi.sort_values(by = int_cols)
            df_other_dosi.index = df_dosi.index
            df_result.at[i, "summable"] = int(check_summable_df(df_dosi, df_other_dosi))
        # Check if the set of voxels of each T/nuage label is non-empty
        count_labels_t = [df_other_dosi['T'].isin([label]).sum() for label in labels_t]
        df_result.loc[i, cols_labels_t] = count_labels_t
        count_labels_nuage = [df_other_dosi['NUAGE'].isin([label]).sum() for label in labels_nuage]
        df_result.loc[i, cols_labels_nuage] = count_labels_nuage
        # Check is nipples nuage has other T than 500
        df_result.loc[i, "359_not_T_500"] = (df_other_dosi.loc[df_other_dosi["NUAGE"] == 359, "T"] != 500).sum()
        df_result.loc[i, "360_not_T_500"] = (df_other_dosi.loc[df_other_dosi["NUAGE"] == 360, "T"] != 500).sum()
        # Get the dose image size
        if df_result.loc[i, "remaining_rows"] > 1:
            x = np.array(df_other_dosi['X'] - min(df_other_dosi['X']), dtype='int') // 2
            y = np.array(df_other_dosi['Y'] - min(df_other_dosi['Y']), dtype='int') // 2
            z = np.array(df_other_dosi['Z'] - min(df_other_dosi['Z']), dtype='int') // 2
            df_result.loc[i, ["size_x", "size_y", "size_z"]] = [max(x)+1,max(y)+1,max(z)+1]

    return df_result

def analyze_dataset(doses_dataset_dir, metadata_dir):
    df_files = pd.read_csv(metadata_dir + "list_newdosi_files.csv")
    grouped_files = df_files.groupby(by = ["ctr", "numcent"])
    logger = setup_logger("check_dataset", metadata_dir + "check_dataset.log")
    logger.info(f"Number of files: {df_files.shape[0]}")
    logger.info(f"Number of patients: {len(grouped_files)}")
    nworkers = get_ncpus()
    logger.info(f"Number of workers: {nworkers}")
    # res_checks = [Checker(doses_dataset_dir)(group) for name, group in grouped_files]
    partial_check_files_patient = partial(check_files_patient, doses_dataset_dir = doses_dataset_dir)
    pool = Pool(nworkers)
    groups = [group for name, group in grouped_files]
    res_checks = list(tqdm.tqdm(pool.imap(partial_check_files_patient, groups), total = len(groups)))
    pool.close()
    pool.join()
    df_files_checks = pd.concat(res_checks)
    del df_files_checks["index"]
    df_files_checks.to_csv(metadata_dir + "list_newdosi_checks.csv", index = False)
    df_files_summable = df_files_checks.loc[df_files_checks["summable"] == 1, :]
    file_biggest_size = metadata_dir + "biggest_image_size.csv"
    np.max(df_files_summable[["size_x", "size_y", "size_z"]], axis = 0).to_csv(file_biggest_size, header = None)

