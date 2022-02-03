
import pandas as pd
import numpy as np
import os, logging
from multiprocessing import Pool, cpu_count
from radiopreditool_utils import *

class Checker(object):
    def __init__(self, doses_dataset_dir):
        self.doses_dataset_dir = doses_dataset_dir
    def __call__(self, df):
        return check_files_patient(self.doses_dataset_dir, df)

def check_files_patient(doses_dataset_dir, df_files_patient):
    logger = logging.getLogger("check_dataset")
    relevant_cols = ['X', 'Y', 'Z', 'T', 'ID2013A']
    int_cols = ['X', 'Y', 'Z', 'T']
    df_result = df_files_patient.copy().reset_index()
    df_result[["nbr_nan_rows", "remaining_rows", "missing_date", "outdated_treatment", "different_shapes"]] = 0
    df_result[["well_ordered_rows", "summable"]] = 1
    idx_sort_by_date_csv_files = np.argsort([get_date(newdosi_file) for newdosi_file in df_result["filename_dose_matrix"]])
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
    return df_result

def analyze_dataset(doses_dataset_dir, metadata_dir):
    df_files = pd.read_csv(metadata_dir + "list_newdosi_files.csv")
    grouped_files = df_files.groupby(by = ["ctr", "numcent"])
    logger = setup_logger("check_dataset", metadata_dir + "check_dataset.log")
    logger.info(f"Number of files: {df_files.shape[0]}")
    logger.info(f"Number of patients: {len(grouped_files)}")
    logger.info(f"Number of workers: {cpu_count()}")
    nworkers = int(os.getenv("SLURM_CPUS_PER_TASK"))
    if nworkers == None:
        nworkers = cpu_count()
    with Pool(nworkers) as p:
        res_checks = p.map(Checker(doses_dataset_dir), [group for name, group in grouped_files])
    df_files_checks = pd.concat(res_checks)
    del df_files_checks["index"]
    df_files_checks.to_csv(metadata_dir + "list_newdosi_checks.csv", index = False)

