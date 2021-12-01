
import pandas as pd
import numpy as np
from datetime import datetime

def survival_date(row):
    if row["numcent"] == 199103047:
        return datetime.strptime("03/11/2019", "%d/%m/%Y")
    if row["Pathologie_cardiaque"] == 1:
        return datetime.strptime(row["date_pathol_cardiaque"], "%d/%m/%Y")
    if row["deces"] == 1:
        if pd.isna(row["date_deces"]):
            return datetime.strptime(row["date_sortie"], "%d/%m/%Y")
        else:
            return datetime.strptime(row["date_deces"], "%d/%m/%Y")
    else:
        cols_date = ["date_sortie", "datederm", "date_dvigr", "date_rep", "date_rep2", "cslt_date_cslt", "date_diag"]
        min_date = datetime.strptime("31/12/2016", "%d/%m/%Y")
        return max([datetime.strptime(row[col], "%d/%m/%Y") for col in cols_date if not pd.isna(row[col])] + [min_date])

def create_trainset(file_radiomics, file_fccss_clinical, analyzes_dir):
    df_radiomics = pd.read_csv(file_radiomics)
    df_fccss = pd.read_csv(file_fccss_clinical, low_memory = False)
    cols_date = ["date_sortie", "datederm", "date_dvigr", "date_rep", "date_rep2", "cslt_date_cslt", "date_diag"]
    cols_survival = ["numcent", "Pathologie_cardiaque", "deces", "date_pathol_cardiaque", "date_deces"] + cols_date 
    df_survival = df_fccss[cols_survival]
    df_fccss["survival_date"] = df_survival.apply(survival_date, axis = 1)
    df_fccss["datetime_date_diag"] = df_fccss["date_diag"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y"))
    df_fccss["survival_time_years"] = df_fccss[["survival_date", "datetime_date_diag"]].apply(lambda x: (x["survival_date"] - x["datetime_date_diag"]).total_seconds() / (365.25 * 24 * 3600), axis = 1)
    cols_radiomics = df_radiomics.columns.to_list()
    cols_radiomics.remove("ctr"), cols_radiomics.remove("numcent")
    cols_clinical = []
    df_trainset = df_fccss[["ctr", "numcent"] + ["Pathologie_cardiaque", "survival_time_years"] + cols_clinical]
    df_trainset = df_trainset.merge(df_radiomics, how = "left", on = ["ctr", "numcent"])
    df_trainset.to_csv(analyzes_dir + "trainset.csv")

