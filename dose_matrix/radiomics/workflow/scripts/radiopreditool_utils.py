
import pandas as pd
from datetime import datetime

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

def check_summable_df(df_dosi_A, df_dosi_B, voi_type):
    return (df_dosi_A['X'].equals(df_dosi_B['X']) and df_dosi_A['Y'].equals(df_dosi_B['Y']) and \
            df_dosi_A['Z'].equals(df_dosi_B['Z']) and df_dosi_A[voi_type].equals(df_dosi_B[voi_type]))

