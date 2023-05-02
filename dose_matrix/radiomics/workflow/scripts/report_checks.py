
import pandas as pd
import logging, re
from radiopreditool_utils import setup_logger

def print_report(metadata_dir, fccss_clinical_dataset, labels_t_organs_file):
    logger = setup_logger("report_checks", metadata_dir + "report_checks.txt")
    df_newdosi_checks = pd.read_csv(metadata_dir + "list_newdosi_checks.csv")
    df_empty = df_newdosi_checks[df_newdosi_checks["remaining_rows"] <= 1]
    df_unordered = df_newdosi_checks[df_newdosi_checks["well_ordered_rows"] == 0]
    df_missing_date = df_newdosi_checks[df_newdosi_checks["missing_date"] == 1]
    df_outdated = df_newdosi_checks[df_newdosi_checks["outdated_treatment"] == 1]
    df_not_summable = df_newdosi_checks[df_newdosi_checks["summable"] == 0]
    df_different_shapes = df_newdosi_checks[df_newdosi_checks["different_shapes"] == 1]
    logger.info(f"{df_empty.shape[0]} empty files: {df_empty['filename_dose_matrix'].to_list()}")
    logger.info(f"{df_unordered.shape[0]} unordered files: {df_unordered['filename_dose_matrix'].to_list()}")
    logger.info(f"{df_missing_date.shape[0]} files with missing date: "
                f"{df_missing_date['filename_dose_matrix'].to_list()}")
    logger.info(f"{df_not_summable.shape[0]} not summable files: {df_not_summable['filename_dose_matrix'].to_list()}")
    logger.info(f"{df_different_shapes.shape[0]} with a different shape from the first file: "
                f"{df_different_shapes['filename_dose_matrix'].to_list()}")
    # Table of labels t with their missing rate in the fccss newdosi files
    cols_labels_t = [col for col in df_newdosi_checks.columns if re.match("^count_[0-9]", col)]
    df_detect_labels_t = df_newdosi_checks[["ctr", "numcent"] + cols_labels_t].groupby(by = ["ctr", "numcent"])\
                                                                              .agg(lambda x: (x>0).any())
    df_fccss_all = pd.read_csv(fccss_clinical_dataset, low_memory = False)
    df_fccss_women = df_fccss_all.loc[df_fccss_all["Sexe"] == 2, :]
    df_fccss_men = df_fccss_all.loc[df_fccss_all["Sexe"] == 1, :]
    for (name, df_fccss) in [("all", df_fccss_all), ("women", df_fccss_women), ("men", df_fccss_men)]:
        df_fccss = df_fccss[["ctr", "numcent"]]
        df_fccss_detect_labels_t = df_fccss.merge(df_detect_labels_t.reset_index(drop = True),
                                                  on = ["ctr", "numcent"], how = "inner")
        print(df_fccss.shape)
        print(df_fccss_detect_labels_t.shape)
        df_labels_t_organs  = pd.read_csv(labels_t_organs_file).set_index("NN")
        int_labels_t = [int(col.replace("has_", "")) for col in cols_labels_t]
        missing_labels_t = df_fccss_detect_labels_t.sum().loc[cols_labels_t]
        n_patients = df_fccss_detect_labels_t.shape[0]
        cols_df_summary =  {"label_t_name": df_labels_t_organs.loc[int_labels_t, "StructLabel"].values,
                            "missing_count": (n_patients - missing_labels_t).values,
                            "missing_ratio": (1 - missing_labels_t/n_patients).values}
        df_summary_labels_t = pd.DataFrame(cols_df_summary, index = int_labels_t)
        df_summary_labels_t.to_csv(metadata_dir + f"fccss_missing_labels_t_{name}.csv",
                                   index = True, encoding = "utf-8")

