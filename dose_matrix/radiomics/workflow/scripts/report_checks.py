
import pandas as pd

def print_report(metadata_dir):
    df_files_checks = pd.read_csv(metadata_dir + "list_newdosi_checks.csv")
    df_empty = df_files_checks[df_files_checks["remaining_rows"] <= 1]
    df_unordered = df_files_checks[df_files_checks["well_ordered_rows"] == 0]
    df_missing_date = df_files_checks[df_files_checks["missing_date"] == 1]
    df_outdated = df_files_checks[df_files_checks["outdated_treatment"] == 1]
    df_not_summable = df_files_checks[df_files_checks["summable"] == 0]
    df_different_shapes = df_files_checks[df_files_checks["different_shapes"] == 1]
    str_report = f"{df_empty.shape[0]} empty files: {df_empty['filename_dose_matrix'].to_list()}\n"
    str_report += f"{df_unordered.shape[0]} unordered files: {df_unordered['filename_dose_matrix'].to_list()}\n"
    str_report += f"{df_missing_date.shape[0]} files with missing date: {df_missing_date['filename_dose_matrix'].to_list()}\n"
    str_report += f"{df_not_summable.shape[0]} not summable files: {df_not_summable['filename_dose_matrix'].to_list()}\n"
    str_report += f"{df_different_shapes.shape[0]} with a different shape from the first file: {df_different_shapes['filename_dose_matrix'].to_list()}\n"
    with open(metadata_dir + "report_checks.txt", "w") as f_report:
        f_report.write(str_report)

