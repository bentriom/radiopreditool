
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from itertools import compress

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy.stats import zscore
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as hac
import statsmodels.stats.multitest as mlt

from multiprocessing import Pool, cpu_count
import os, sys, importlib, string, re
sys.path.append("radiomics/workflow/scripts")

import csv2nii, feature_extractor, check_dataset, trainset
from radiopreditool_utils import *
from radiomics import featureextractor
from lifelines import CoxPHFitter
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA

DATETIME_FORMAT_FCCSS = "%Y-%m-%d"

## Create datasets

# Fill missing values of the dataset
def fill_missing(df_fccss):
    # Patients who have pathol cardiaque >= grade 3, a date_pc but no date_pc_3
    mask = pd.isnull(df_fccss["date_pathol_cardiaque_3"]) & (df_fccss["Pathologie_cardiaque_3"] == 1)
    df_fccss.loc[mask, "date_pathol_cardiaque_3"] = df_fccss.loc[mask, "date_pathol_cardiaque"]
    mask = pd.isnull(df_fccss["date_pathol_cardiaque_3_new"]) & (df_fccss["Pathologie_cardiaque_3_new"] == 1)
    df_fccss.loc[mask, "date_pathol_cardiaque_3_new"] = df_fccss.loc[mask, "date_pathol_cardiaque"]
    # Dummy variables for the first cancer (iccc)
    enc = OneHotEncoder()
    enc.fit(df_fccss["iccc"].values.reshape(-1,1))
    dummy_iccc_cols = pd.Series(enc.get_feature_names_out()).str.replace("x0", "iccc").values
    df_fccss[dummy_iccc_cols] = enc.transform(df_fccss["iccc"].values.reshape(-1,1)).toarray()
    # Age at diagnosis of the first tumor
    def age_at_diagnosis(patient):
        date_diag = datetime.strptime(patient.loc["date_diag"], DATETIME_FORMAT_FCCSS)
        birthdate = datetime.strptime(patient.loc["date_nais"], DATETIME_FORMAT_FCCSS)
        age = date_diag.year - birthdate.year - ((date_diag.month, date_diag.day) < (birthdate.month, birthdate.day))
        return age
    def year_diagnosis(patient):
        date_diag = datetime.strptime(patient.loc["date_diag"], DATETIME_FORMAT_FCCSS)
        return date_diag.year
    def categ_age_at_diagnosis(age):
        if 0 <= age <= 5:
            return 0
        if 5 < age <= 10:
            return 1
        if 10 < age <= 15:
            return 2
        if age > 15:
            return 3
    def chimio_groupe_autre(patient):
        if patient["chimiotherapie"] == 1 and patient[["ALKYL", "ANTHRA", "VINCA", "ANTIM", "ANTIB"]].isin([0]).all():
            return 1
        else:
            return 0
    df_fccss.loc[:, "age_at_diagnosis"] = df_fccss[["date_diag", "date_nais"]].apply(age_at_diagnosis, axis=1)
    df_fccss.loc[:, "year_diagnosis"] = df_fccss[["date_diag"]].apply(year_diagnosis, axis=1)
    df_fccss.loc[:, "categ_age_at_diagnosis"] = df_fccss["age_at_diagnosis"].apply(categ_age_at_diagnosis)
    df_fccss.loc[:, "CHIMIO_GROUPE_AUTRE"] = df_fccss[["chimiotherapie","ALKYL", "ANTHRA", "VINCA", "ANTIM", "ANTIB"]]\
                                             .apply(chimio_groupe_autre, axis=1)

# Compute survival times
def survival_date(event_col, date_event_col, row):
    if row["numcent"] == 199103047:
        return datetime.strptime("03/11/2019", "%d/%m/%Y")
    elif row[event_col] == 1:
        return datetime.strptime(row[date_event_col], DATETIME_FORMAT_FCCSS)
    elif row["deces"] == 1:
        if not pd.isnull(row["date_deces"]):
            return datetime.strptime(row["date_deces"], DATETIME_FORMAT_FCCSS)
        else:
            return datetime.strptime(row["date_sortie"], DATETIME_FORMAT_FCCSS)
    else:
        cols_date = ["date_sortie", "datederm", "date_dvigr", "date_rep", "date_rep2", "cslt_date_cslt", "date_diag"]
        min_date = datetime.strptime("31/12/2016", "%d/%m/%Y")
        return max([datetime.strptime(row[col], DATETIME_FORMAT_FCCSS) \
                    for col in cols_date if not pd.isna(row[col])] + [min_date])

# Complete features for patients with no radiotherapy
def fill_features_no_rt(df_dataset, col_treated_by_rt, params_file):
    mask_no_rt = df_dataset[col_treated_by_rt] == 0
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    ex_image, ex_mask = create_image_mask_example(zero_img = True)
    dict_features_values = extractor.execute(ex_image, ex_mask, label = 1)
    # Get all the masks/labels used in the extraction
    list_labels = get_all_labels(df_dataset)
    features_values = [dict_features_values[x] for x in dict_features_values if not x.startswith("diagnostics_")]
    for label in list_labels:
        features = [pretty_dosesvol(f"{label}_{x}") for x in dict_features_values if not x.startswith("diagnostics_")]
        mask_features_in_df = [f in df_dataset.columns for f in features]
        df_dataset.loc[mask_no_rt, list(compress(features,mask_features_in_df))] = \
            list(compress(features_values,mask_features_in_df))

# Filter patients according to some criteria
def filter_patients(df_dataset, name_filter_dataset, event_col, duration_col):
    if name_filter_dataset in ["", None]:
        return df_dataset
    elif name_filter_dataset == "positive_entropy":
        assert "1320_original_firstorder_Entropy" in df_dataset.columns
        mask = (df_dataset["has_radiomics"] == 0) | \
               ((df_dataset["has_radiomics"] == 1) & (df_dataset["1320_original_firstorder_Entropy"] > 0))
        return df_dataset.loc[mask, :]
    elif name_filter_dataset == "sampling":
        return df_dataset.groupby(event_col, group_keys = False).apply(lambda x: x.sample(frac = 0.2))
    elif name_filter_dataset == "women":
        assert "Sexe" in df_dataset.columns
        mask = (df_dataset["Sexe"] == 2)
        return df_dataset.loc[mask, :]
    elif re.match("^women_sim", name_filter_dataset):
        if name_filter_dataset == "women_sim":
            n_samples = 400
            frac_censor = 0.1
            distribution = "exp"
        else:
            match_settings = re.match("^women_sim_(\w+)_censor_(\S+)_nsamples_(\d+)", name_filter_dataset)
            n_samples = int(match_settings[3])
            frac_censor = float(match_settings[2])
            distribution = match_settings[1]
        logger = logging.getLogger("dataset")
        logger.info("!! This is a simulated dataset !!")
        logger.info(f"Number of patients: {n_samples} with {frac_censor*100:.1f}% censorship.")
        logger.info(f"Survival time distribution: {distribution}.")
        cols_dosiomics = [col for col in df_dataset.columns if re.match("^[0-9]{3,5}_original_", col)]
        cols_dv = [col for col in df_dataset.columns if re.match("^dv_\w", col)]
        cols = cols_dv + cols_dosiomics
        clinical_features =  get_clinical_features(df_dataset, event_col, duration_col)
        betas = np.zeros(len(cols))
        mask_pos = np.isin(cols, [f"{label}_original_firstorder_Mean" for label in [2413, 3413, 2601, 10000]])
        mask_neg = np.isin(cols, [f"{label}_original_firstorder_Entropy" for label in [2413, 3413, 2601, 10000]])
        betas[mask_pos] = 0.75
        betas[mask_neg] = -0.5
        surv_times, status, X = generate_survival_times(n_samples, len(cols), betas,
                                                        target_frac_censor = frac_censor, distribution = distribution)
        df_dataset_sim = pd.DataFrame(columns = ["ctr", "numcent", "has_radiomics", event_col, duration_col] + \
                                      clinical_features + cols)
        df_dataset_sim[cols] = np.transpose(X)
        df_dataset_sim["has_radiomics"] = 1
        df_dataset_sim[event_col] = status
        df_dataset_sim[duration_col] = surv_times
        df_dataset_sim["ctr"] = 22
        df_dataset_sim["numcent"] = rand.choice(range(n_samples), n_samples, replace = False)
        # D50 is a median but here we copy the value of the corresponding mean
        # to ensure that doses vol models can predict the event too
        cols_dv_D50 = [f"dv_D50_{label}" for label in get_all_labels(df_dataset_sim)]
        cols_means = [f"{label}_original_firstorder_Mean" for label in get_all_labels(df_dataset_sim)]
        df_dataset_sim.drop(columns = [col for col in df_dataset_sim.columns if re.match("^dv_\w", col)], inplace = True)
        df_dataset_sim[cols_dv_D50] = df_dataset_sim[cols_means]
        return df_dataset_sim
    else:
        raise NotImplementedError(f"name_filter_dataset: {name_filter_dataset} not supported.")

# Create dataset based on availaible dosiomics, clinical variables and survival data
def create_dataset(file_radiomics, file_fccss_clinical, analyzes_dir, clinical_variables,
                   event_col, date_event_col, params_file, name_filter_dataset = None):
    logger = setup_logger("dataset", analyzes_dir + "dataset.log")
    logger.info(f"Event col: {event_col}. Date of event col: {date_event_col}")
    os.makedirs(analyzes_dir + "datasets", exist_ok = True)
    df_radiomics = pd.read_csv(file_radiomics)
    df_radiomics["has_radiomics"] = 1
    df_fccss = pd.read_csv(file_fccss_clinical, low_memory = False)
    fill_missing(df_fccss)
    logger.info(f"df_radiomics (fccss patients + others): {df_radiomics.shape}")
    mask_na = df_radiomics.isna().any(axis = 1)
    logger.info(f"In df_radiomics, number of patients with NA: {sum(mask_na)}")
    logger.info(f"df_fccss: {df_fccss.shape}")
    # Create survival time col
    logger.info("Creating survival columns")
    cols_date = ["date_sortie", "datederm", "date_dvigr", "date_rep", "date_rep2", "cslt_date_cslt", "date_diag"]
    cols_survival = ["numcent", event_col, "deces", date_event_col, "date_deces"] + cols_date
    df_survival = df_fccss[cols_survival]
    df_fccss["survival_date"] = df_survival.apply(lambda x: survival_date(event_col, date_event_col, x), axis = 1)
    df_fccss["datetime_date_diag"] = df_fccss["date_diag"].apply(lambda x: datetime.strptime(x, DATETIME_FORMAT_FCCSS))
    surv_duration_col = "survival_time_years"
    df_fccss[surv_duration_col] = df_fccss[["survival_date", "datetime_date_diag"]].apply(
        lambda x: (x["survival_date"] - x["datetime_date_diag"]).total_seconds() / (365.25 * 24 * 3600), axis = 1)
    cols_radiomics = df_radiomics.columns.to_list()
    cols_radiomics.remove("ctr"), cols_radiomics.remove("numcent")
    # Create dataset by merging fccss and radiomics
    df_dataset = df_fccss[["ctr", "numcent"] + clinical_variables + [event_col, surv_duration_col]]
    # Keep columns for filters and statistics
    col_treated_by_rt = "radiotherapie_1K"
    cols_keep_and_drop = [col_treated_by_rt, "Sexe"]
    [df_dataset.insert(len(df_dataset.columns), col, df_fccss[col]) for col in cols_keep_and_drop \
                                                                    if col not in clinical_variables]
    df_dataset = df_dataset.merge(df_radiomics, how = "left", on = ["ctr", "numcent"])
    df_dataset.loc[:, "has_radiomics"] = df_dataset.loc[:, "has_radiomics"].replace(np.nan, 0)
    # Some filters need radiomics
    df_dataset = filter_patients(df_dataset, name_filter_dataset, event_col, surv_duration_col)
    # Eliminating patients with negative survival
    mask_negative_times = df_dataset[surv_duration_col] <= 0
    logger.info(f"Eliminating {sum(mask_negative_times)} patients with non-positive survival times")
    df_dataset = df_dataset.loc[~mask_negative_times, :]
    # Fill columns about radiotherapie: dosiomics filled of zeros for patients with no RT
    fill_features_no_rt(df_dataset, col_treated_by_rt, params_file)
    logger.info(f"Full fccss dataset (with '{name_filter_dataset}' filter) : {df_dataset.shape}")
    logger.info(f"Full fccss dataset without NA: {df_dataset.dropna().shape}")
    logger.info(f"Full fccss dataset with radiomics: {df_dataset.loc[df_dataset['has_radiomics'] == 1, :].shape}")
    # Save
    [df_dataset.drop(columns = col, inplace = True) for col in cols_keep_and_drop if col not in clinical_variables]
    df_dataset.to_csv(analyzes_dir + "datasets/dataset.csv.gz", index = False)

# /!\ Deprecated /!\
# One split train / test
def split_dataset(file_radiomics, file_fccss_clinical, analyzes_dir, clinical_variables, event_col, date_event_col,
                  end_name_sets = "", test_size = 0.3, seed = None):
    logger = setup_logger("trainset", analyzes_dir + "trainset.log")
    logger.info(f"Event col: {event_col}. Date of event col: {date_event_col}")
    surv_duration_col = "survival_time_years"
    df_dataset = pd.read_csv(analyzes_dir + "datasets/dataset.csv.gz")
    cols_y = [event_col, surv_duration_col]
    col_treated_by_rt = "radiotherapie_1K"
    if col_treated_by_rt not in clinical_variables:
        df_fccss = pd.read_csv(file_fccss_clinical, low_memory = False)
        df_fccss = df_fccss[["ctr", "numcent", col_treated_by_rt]]
        df_dataset = df_dataset.merge(df_fccss, how = "left", on = ["ctr", "numcent"])
    df_X = df_dataset[[c for c in df_dataset.columns if c not in cols_y]]
    df_y = df_dataset[["ctr", "numcent"] + cols_y]
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y,
                                                                    random_state = seed, shuffle = True,
                                                                    test_size = test_size, stratify = df_y[event_col])
    df_trainset = df_X_train.merge(df_y_train, how = "inner", on = ["ctr", "numcent"])
    df_testset = df_X_test.merge(df_y_test, how = "inner", on = ["ctr", "numcent"])
    df_trainset_omit = df_trainset.dropna()
    df_testset_omit = df_testset.dropna()
    df_trainset_radiomics = df_trainset.loc[df_trainset["has_radiomics"] == 1,:]
    df_testset_radiomics = df_testset.loc[df_testset["has_radiomics"] == 1,:]
    logger.info(f"Trainset: {df_trainset.shape}")
    logger.info(f"Trainset with radiomics features: {df_trainset_radiomics.shape}")
    logger.info(f"Testset: {df_testset.shape} ({test_size * 100}%)")
    logger.info(f"Testset with radiomics features: {df_testset_radiomics.shape}")
    nsamples_train = df_trainset.shape[0]
    nsamples_test = df_testset.shape[0]
    nsamples_train_omit = df_trainset_omit.shape[0]
    nsamples_test_omit = df_testset_omit.shape[0]
    logger.info(f"Balance train/test event: {df_trainset[event_col].sum()/nsamples_train:.4f}"
                                          f"{df_testset[event_col].sum()/nsamples_test:.4f}")
    logger.info(f"Balance train/test event, omitting NAs: {df_trainset_omit[event_col].sum()/nsamples_train_omit:.4f}"
                                                       f" {df_testset_omit[event_col].sum()/nsamples_test_omit:.4f}")
    logger.info(f"Balance train/test treated by RT: {df_trainset[col_treated_by_rt].sum()/nsamples_train:.4f}"
                                                 f" {df_testset[col_treated_by_rt].sum()/nsamples_test:.4f}")
    logger.info(f"Balance train/test with radiomics features: {df_trainset['has_radiomics'].sum()/nsamples_train:.4f}"
                                                           f" {df_testset['has_radiomics'].sum()/nsamples_test:.4f}")
    # Save
    if col_treated_by_rt not in clinical_variables:
        df_trainset.drop(columns = col_treated_by_rt, inplace = True)
        df_testset.drop(columns = col_treated_by_rt, inplace = True)
    df_trainset.to_csv(analyzes_dir + f"datasets/trainset{end_name_sets}.csv.gz", index = False)
    df_testset.to_csv(analyzes_dir + f"datasets/testset{end_name_sets}.csv.gz", index = False)

# Stratified K fold cross-validation splits
def kfold_multiple_splits_dataset(nb_estim, file_radiomics, file_fccss_clinical,
                                  analyzes_dir, clinical_variables, event_col, date_event_col, seed = None):
    logger = setup_logger("trainset", analyzes_dir + "trainset.log")
    logger.info(f"{nb_estim}-Fold stratified cross-validation")
    logger.info(f"Event col: {event_col}. Date of event col: {date_event_col}")
    surv_duration_col = "survival_time_years"
    df_dataset = pd.read_csv(analyzes_dir + "datasets/dataset.csv.gz")
    cols_y = [event_col, surv_duration_col]
    col_treated_by_rt = "radiotherapie_1K"
    if col_treated_by_rt not in clinical_variables:
        df_fccss = pd.read_csv(file_fccss_clinical, low_memory = False)
        df_fccss = df_fccss[["ctr", "numcent", col_treated_by_rt]]
        df_dataset = df_dataset.merge(df_fccss, how = "left", on = ["ctr", "numcent"])
    df_X = df_dataset[[c for c in df_dataset.columns if c not in cols_y]]
    df_y = df_dataset[["ctr", "numcent"] + cols_y]
    skf = StratifiedKFold(n_splits = nb_estim)
    for i, (train_index, test_index) in enumerate(skf.split(df_X, df_y[event_col])):
        df_trainset = df_dataset.loc[train_index, :]
        df_testset = df_dataset.loc[test_index, :]
        del_col_omit = [col_treated_by_rt] if col_treated_by_rt not in clinical_variables else []
        df_trainset_omit = df_trainset.drop(columns = del_col_omit).dropna()
        df_testset_omit = df_testset.drop(columns = del_col_omit).dropna()
        df_trainset_radiomics = df_trainset.loc[df_trainset["has_radiomics"] == 1,:]
        df_testset_radiomics = df_testset.loc[df_testset["has_radiomics"] == 1,:]
        logger.info(f"Trainset n°{i}: {df_trainset.shape}")
        logger.info(f"Trainset with radiomics features: {df_trainset_radiomics.shape}")
        logger.info(f"Testset: {df_testset.shape} ({df_testset.shape[0]/df_dataset.shape[0]*100:.2f}%)")
        logger.info(f"Testset with radiomics features: {df_testset_radiomics.shape}")
        nsamples_train = df_trainset.shape[0]
        nsamples_test = df_testset.shape[0]
        nsamples_train_omit = df_trainset_omit.shape[0]
        nsamples_test_omit = df_testset_omit.shape[0]
        logger.info(f"Balance train/test event: {df_trainset[event_col].sum()/nsamples_train:.4f}"
                                             f" {df_testset[event_col].sum()/nsamples_test:.4f}")
        logger.info(f"Balance train/test event, omitting NAs: {df_trainset_omit[event_col].sum()/nsamples_train_omit:.4f}"
                                                           f" {df_testset_omit[event_col].sum()/nsamples_test_omit:.4f}")
        logger.info(f"Balance train/test treated by RT: {df_trainset[col_treated_by_rt].sum()/nsamples_train:.4f}"
                                                     f" {df_testset[col_treated_by_rt].sum()/nsamples_test:.4f}")
        logger.info(f"Balance train/test with radiomics features: {df_trainset['has_radiomics'].sum()/nsamples_train:.4f}"
                                                               f" {df_testset['has_radiomics'].sum()/nsamples_test:.4f}")
        # Save
        if col_treated_by_rt not in clinical_variables:
            df_trainset.drop(columns = col_treated_by_rt, inplace = True)
            df_testset.drop(columns = col_treated_by_rt, inplace = True)
        df_trainset.to_csv(analyzes_dir + f"datasets/trainset_{i}.csv.gz", index = False)
        df_testset.to_csv(analyzes_dir + f"datasets/testset_{i}.csv.gz", index = False)

## Feature elimination: eliminate sparse and redundant columns

# Filter 1: eliminate radiomics with too much missing values
def filter_nan_values_radiomics(df_covariates_nan_values, features_radiomics, threshold):
    nbr_newdosi_patients = df_covariates_nan_values.shape[0]
    prop_values_cols = df_covariates_nan_values.apply(
                            lambda col_series: 1 - sum(pd.isnull(col_series))/nbr_newdosi_patients)
    filter_1_cols_radiomics = list(prop_values_cols[prop_values_cols > threshold].index.values)
    return filter_1_cols_radiomics

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def hclust_corr(df_covariates_hclust, threshold, do_plot = False, analyzes_dir = "./", name = ""):
    hclust = AgglomerativeClustering(n_clusters = None, linkage = "complete",
                                     affinity = "precomputed", distance_threshold = 1 - threshold)
    distance_matrix = df_covariates_hclust.corr(method = "kendall").apply(lambda x: 1 - abs(x)).to_numpy()
    y_clusters = hclust.fit_predict(distance_matrix)
    if do_plot:
        fig = plt.figure(figsize=(8,10))
        plt.title(f"Hierarchical Clustering Dendrogram "
                  f"(threshold: {threshold} - features: {hclust.n_clusters_}/{distance_matrix.shape[0]})")
        # plot the top three levels of the dendrogram
        plot_dendrogram(hclust, orientation = "right",
                        labels = pretty_labels(df_covariates_hclust.columns), leaf_font_size = 8)
        plt.axvline(1 - threshold)
    return df_covariates_hclust.columns.values, y_clusters

def get_nbr_features_hclust(df_covariates_hclust, threshold, do_plot = False):
    _, y_clusters = hclust_corr(df_covariates_hclust, threshold, do_plot)
    return len(np.unique(y_clusters))

# Eliminate features for a mask label (32X, 1320..)
# It performs hierarchical clustering based on Kendall's tau in order to group correlated features
# Then, univariate Cox PH is performed for each feature, and we keep the statistically significant features
def filter_corr_hclust_label_uni(df_trainset, df_covariates_hclust, corr_threshold,
                                 event_col, surv_duration_col, analyzes_dir, do_plot = False, name = ""):
    all_features_radiomics, y_clusters = hclust_corr(df_covariates_hclust, corr_threshold,
                                                     do_plot = do_plot, analyzes_dir = analyzes_dir, name = name)
    filter_2_cols_radiomics = []
    id_clusters = np.unique(y_clusters)
    df_survival = df_trainset.copy().dropna()
    for c in id_clusters:
        list_features = [f for (idx, f) in enumerate(all_features_radiomics) if y_clusters[idx] == c]
        list_pvalues = []
        list_coefs = []
        list_hr = []
        for feature in list_features:
            cph_feature = CoxPHFitter(penalizer = 0.0001)
            df_survival.loc[:,feature] = StandardScaler().fit_transform(df_survival[[feature]])
            cph_feature.fit(df_survival, duration_col = surv_duration_col,
                            event_col = event_col, formula = feature, fit_options = {"step_size": 0.4})
            pvalue = cph_feature.summary.loc[feature, "p"]
            coef = cph_feature.summary.loc[feature, "coef"]
            list_pvalues.append(pvalue)
            list_coefs.append(coef)
        bh_correction = mlt.fdrcorrection(list_pvalues, alpha = 0.05)
        mask_reject = bh_correction[0]
        # If no significant p-values, takes the feature with the smaller p-values
        if sum(mask_reject) == 0:
            selected_feature = list_features[np.argmin(list_pvalues)]
        # Else, takes the biggest coefficient among the significant p-values
        else:
            cox_significant_features = np.asarray(list_features)[mask_reject]
            cox_significant_coefs = abs(np.asarray(list_coefs)[mask_reject])
            selected_feature = cox_significant_features[np.argmax(cox_significant_coefs)]
        filter_2_cols_radiomics.append(selected_feature)
    if do_plot:
        plt.text(1.1, 0, '\n'.join(pretty_labels(filter_2_cols_radiomics)))
        save_dir = f"{analyzes_dir}corr_plots/hclust/"
        os.makedirs(save_dir, exist_ok = True)
        plt.savefig(f"{save_dir}{corr_threshold}_{name}.png", dpi = 480, bbox_inches='tight',
                    facecolor = "white", transparent = False)
        plt.close()
    return filter_2_cols_radiomics

# Eliminate features for a mask label (32X, 1320..)
# It performs hierarchical clustering based on Kendall's tau in order to group correlated features
# Then, multivariate Cox PH is performed, and we keep the feature with the largest hazard ratio
def filter_corr_hclust_label_multi(df_trainset, df_covariates_hclust, corr_threshold,
                                   event_col, surv_duration_col, analyzes_dir, do_plot = False, name = ""):
    all_features_radiomics, y_clusters = hclust_corr(df_covariates_hclust, corr_threshold,
                                                     do_plot = do_plot, analyzes_dir = analyzes_dir, name = name)
    filter_2_cols_radiomics = []
    id_clusters = np.unique(y_clusters)
    df_survival = df_trainset.copy().dropna()
    for c in id_clusters:
        list_features = [f for (idx, f) in enumerate(all_features_radiomics) if y_clusters[idx] == c]
        cph_features = CoxPHFitter(penalizer = 0.0001)
        df_survival.loc[:, list_features] = StandardScaler().fit_transform(df_survival.loc[:, list_features])
        df_survival_cluster = df_survival[list_features + [event_col, surv_duration_col]]
        cph_features.fit(df_survival_cluster, duration_col = surv_duration_col, event_col = event_col,
                         fit_options = {"step_size": 0.4})
        hazard_ratios = np.exp(cph_features.summary.loc[list_features, "coef"])
        selected_feature = list_features[np.argmax(hazard_ratios)]
        filter_2_cols_radiomics.append(selected_feature)
    if do_plot:
        plt.text(1.1, 0, '\n'.join(pretty_labels(filter_2_cols_radiomics)))
        save_dir = f"{analyzes_dir}corr_plots/hclust/"
        os.makedirs(save_dir, exist_ok = True)
        plt.savefig(f"{save_dir}{corr_threshold}_{name}.png", dpi = 480, bbox_inches='tight',
                        facecolor = "white", transparent = False)
        plt.close()
    return filter_2_cols_radiomics


# Filter 2: performs hierarchical clustering based on correlation among variables and 
# and selects the best representant with Cox PH analysis
def filter_corr_hclust_all(df_trainset, df_covariates_hclust, corr_threshold, event_col, surv_duration_col,
                           analyzes_dir, feature_select_method, do_plot = False):
    assert feature_select_method in ["univariate_cox", "multivariate_cox"]
    if feature_select_method == "univariate_cox":
        filter_func = filter_corr_hclust_label_uni
    elif feature_select_method == "multivariate_cox":
        filter_func = filter_corr_hclust_label_multi
    all_filter_2_cols = []
    labels = get_all_labels(df_covariates_hclust)
    logger = logging.getLogger("feature_elimination_hclust_corr")
    logger.info(f"Hclust on labels: {labels}")
    logger.info(f"Method for the selection of one feature within a cluster: {feature_select_method}")
    for label in labels:
        cols_from_label = [feature for feature in df_covariates_hclust.columns if re.match(f"{label}_.*", feature)]
        df_covariates_hclust_label = df_covariates_hclust[cols_from_label]
        all_filter_2_cols += filter_func(df_trainset, df_covariates_hclust_label, corr_threshold,
                                         event_col, surv_duration_col, analyzes_dir, do_plot = do_plot, name = label)
    return all_filter_2_cols

# Feature elimination pipeline with hclust on kendall's tau corr
def feature_elimination_hclust_corr(event_col, analyzes_dir, id_set = "", feature_select_method = "multivariate_cox"):
    file_trainset = f"{analyzes_dir}datasets/"
    file_trainset = f"{file_trainset}dataset.csv.gz" if id_set == "" else f"{file_trainset}trainset_{id_set}.csv.gz"
    df_trainset = pd.read_csv(file_trainset)
    features_radiomics = [feature for feature in df_trainset.columns if re.match("^[0-9]{3,5}_", feature)]
    labels_radiomics = np.unique([label.split('_')[0] for label in df_trainset.columns if re.match("^[0-9]{3,5}_", label)])
    dict_features_per_label = {label: [col for col in df_trainset.columns if re.match(f"{label}_", col)] \
                               for label in labels_radiomics}
    df_covariates_with_radiomics = df_trainset.loc[df_trainset["has_radiomics"] == 1, features_radiomics]
    logger = setup_logger("feature_elimination_hclust_corr", analyzes_dir + "feature_elimination_hclust_corr.log")
    logger.info(f"Trainset dataframe: {df_trainset.shape}")
    logger.info(f"Initial number of radiomics covariates: {df_covariates_with_radiomics.shape[1]}")
    ## First filter: eliminate features with enough missing values
    nan_values_threshold = 0.9
    filter_1_cols_radiomics = filter_nan_values_radiomics(df_covariates_with_radiomics, features_radiomics, nan_values_threshold)
    logger.info(f"After the first filter (non nan values > {nan_values_threshold}): {len(filter_1_cols_radiomics)}")
    # Correlation heatmaps if run on the whole dataset (before correlation elimination)
    if id_set == "":
        os.makedirs(analyzes_dir + "corr_plots", exist_ok = True)
        for label in labels_radiomics:
            features_label = dict_features_per_label[label]
            for group in ["firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"]:
                features = [f for f in features_label if group in f]
                df_corr = df_covariates_with_radiomics[features].corr(method = "kendall")
                df_corr.index = pretty_labels(df_corr.index)
                df_corr.columns = pretty_labels(df_corr.columns)
                fig = plt.figure(figsize=(16, 6))
                heatmap = sns.heatmap(df_corr, vmin = -1, vmax = 1, annot = True, center = 0, cmap = "vlag")
                save_dir = f"{analyzes_dir}corr_plots/all/{label}/"
                os.makedirs(save_dir, exist_ok = True)
                plt.savefig(f"{save_dir}mat_corr_{group}.png", dpi = 480,
                            bbox_inches='tight', facecolor = "white", transparent = False)
                plt.close()
    ## Second filter: eliminate very correlated features with hierarchical clustering + univariate Cox
    df_covariates_hclust = df_covariates_with_radiomics[filter_1_cols_radiomics]
    # Eliminate rows with na
    df_covariates_hclust = df_covariates_hclust.dropna()
    # Eliminate columns with unique values
    unique_cols = [c for c in df_covariates_hclust.columns if len(df_covariates_hclust[c].unique()) == 1]
    df_covariates_hclust.drop(columns = unique_cols, inplace = True)
    corr_threshold = 0.8
    logger.info(f"Correlation threshold: {corr_threshold}")
    surv_duration_col = "survival_time_years"
    logger.info(f"Hclust on dataframe: {df_covariates_hclust.shape}")
    do_hclust_plot = True if id_set == "" else False
    filter_2_cols_radiomics = filter_corr_hclust_all(df_trainset, df_covariates_hclust, corr_threshold,
                                                     event_col, surv_duration_col, analyzes_dir, feature_select_method,
                                                     do_plot = do_hclust_plot)
    logger.info(f"After the second filter (hclust): {len(filter_2_cols_radiomics)}")
    # Correlation heatmaps if run on the whole dataset (after correlation elimination)
    if id_set == "":
        os.makedirs(analyzes_dir + "corr_plots", exist_ok = True)
        for label in labels_radiomics:
            features_label = [feature for feature in filter_2_cols_radiomics if re.match(f"^{label}_", feature)]
            for group in ["firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"]:
                features = [f for f in features_label if group in f]
                df_corr = df_covariates_with_radiomics[features].corr(method = "kendall")
                df_corr.index = pretty_labels(df_corr.index)
                df_corr.columns = pretty_labels(df_corr.columns)
                fig = plt.figure(figsize=(16, 6))
                heatmap = sns.heatmap(df_corr, vmin = -1, vmax = 1, annot = True, center = 0, cmap = "vlag")
                save_dir = f"{analyzes_dir}corr_plots/features_hclust_corr/{label}/"
                os.makedirs(save_dir, exist_ok = True)
                plt.savefig(f"{save_dir}mat_corr_{group}.png", dpi = 480,
                            bbox_inches='tight', facecolor = "white", transparent = False)
                plt.close()
    kept_cols = [feature for feature in df_trainset.columns \
                 if not re.match("^[0-9]{3,5}_", feature) or feature in filter_2_cols_radiomics]
    os.makedirs(analyzes_dir + "screening", exist_ok = True)
    end_filename = "" if id_set == "" else f"_{id_set}"
    pd.DataFrame({"features": kept_cols}).to_csv(analyzes_dir + f"screening/features_hclust_corr{end_filename}.csv",
                                                 index = False, header = None)
    # df_trainset[kept_cols].to_csv(analyzes_dir + "preprocessed_trainset.csv.gz", index = False)


## PCA
def col_class_event(event_col, duration_col, inf_time, sup_time, row):
    if row[event_col] == 0:
        return 0
    if row[event_col] == 1 and (inf_time < row[duration_col] <= sup_time):
        return 1
    else:
        return 0

def pca_viz(file_dataset, event_col, analyzes_dir, duration_col = "survival_time_years"):
    logger = setup_logger("pca", analyzes_dir + "pca_viz.log")
    df_dataset = pd.read_csv(file_dataset)
    # Create columns for events within some time window
    # class_event_col: 0 if no event, 1 if event after year_max, 2 if event within year_max - year_max - 5... 
    df_dataset["class_" + event_col] = 0
    year_max = 20
    timestep = 5
    # Filter data with only known event / non-event
    mask = (df_dataset.loc[:, event_col] == 1) | \
            ((df_dataset.loc[:, event_col] == 0) & (df_dataset.loc[:, duration_col] > year_max))
    df_dataset = df_dataset.loc[mask, :]
    logger.info(f"Number of patients uncensored until {year_max} years: {df_dataset.shape[0]}")
    ranges_events = range(timestep, year_max + 1, timestep)
    df_dataset.loc[:, f"No_{event_col}"] = (df_dataset.loc[:, event_col] == 0).astype(int)
    for sup_time_event in ranges_events:
        inf_time_event = sup_time_event - timestep
        col_class = f"{event_col}_within_{inf_time_event}_{sup_time_event}"
        df_dataset.loc[:, col_class] = df_dataset.apply(
                lambda row: col_class_event(event_col, duration_col, inf_time_event, sup_time_event, row), axis = 1)
    list_pvalues = []
    # all_features = get_all_radiomics_features(df_dataset)
    all_features = [col for col in df_dataset.columns if re.match( "((1320)|(32[0-9]))_original_", col)]
    for feature in all_features:
        cph_feature = CoxPHFitter(penalizer = 0.0001)
        df_univariate = df_dataset[[event_col, duration_col, feature]].dropna()
        df_univariate.loc[:,feature] = StandardScaler().fit_transform(df_univariate[[feature]])
        cph_feature.fit(df_univariate, duration_col = duration_col, event_col = event_col, formula = feature,
                        fit_options = {"step_size": 0.4})
        pvalue = cph_feature.summary.loc[feature, "p"]
        list_pvalues.append(pvalue)
    logger.info("Non-rejected Cox test features are dropped (FDR correction - 0.01)")
    bh_correction = mlt.fdrcorrection(list_pvalues, alpha = 0.01)
    mask_reject = bh_correction[0]
    cox_rejected_features = [all_features[i] for i in range(len(all_features)) if mask_reject[i]]
    # Bar plots of event classes
    # list_labels = ["No event"] + [f"Event after {year_max}"] + [f"Event {year_max - timestep*k} - {year_max - timestep*(k-1)} years" for k in range(1, nbr_classes-1)]
    list_cols = [f"No_{event_col}"] + [f"{event_col}_within_{t-timestep}_{t}" for t in ranges_events]
    list_cols_labels = ["No event"] + [f"Event within {t-timestep}-{t} years" for t in ranges_events]
    nbr_classes = len(list_cols)
    # PCA on all labels
    fig = px.bar(df_dataset[list_cols].sum(),
                 title = f"Patients with censored time >= {year_max} years")
    fig.write_image(analyzes_dir + "pca/barplot_class_event.png", width = 1200, height = 900)
    # Different sets of features for PCA according to the image mask
    labels = get_all_labels(df_dataset)
    names_sets = ["all"] + labels
    set_of_features_radiomics = [cox_rejected_features] + [[feature for feature in cox_rejected_features if re.match(f"{label}_.*", feature)] for label in labels]
    os.makedirs(analyzes_dir + "pca", exist_ok = True)
    # Plot settings
    list_colors = plt.cm.get_cmap('Set1', nbr_classes).colors
    list_markers = ['o' if k == 0 else 'x' for k in range(nbr_classes)]
    list_alphas = [0.4 if k == 0 else 0.9 for k in range(nbr_classes)]
    # Iterate over labels: run PCA
    for (i, features_radiomics) in enumerate(set_of_features_radiomics):
        name_set = names_sets[i]
        logger.info(f"PCA {name_set}")
        logger.info(f"Covariates: {features_radiomics}")
        df_subset = df_dataset.dropna(subset = features_radiomics + [event_col, duration_col])
        X = StandardScaler().fit_transform(df_subset[features_radiomics])
        logger.info(f"X shape: {X.shape}")
        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(X)
        plt.figure()
        for id_col, col_class in enumerate(list_cols):
            idx_cluster = df_subset.loc[:, col_class] == 1
            plt.scatter(X_pca[idx_cluster, 0], X_pca[idx_cluster, 1], label = list_cols_labels[id_col],
                        alpha = list_alphas[id_col], marker = list_markers[id_col], color = list_colors[id_col])
        plt.xlabel('PCA component 1')
        plt.ylabel('PCA component 2')
        plt.title(f"PCA {name_set} ({round(sum(pca.explained_variance_ratio_), 3)} explained variance ratio)\n{df_dataset.shape[0]} patients.")
        plt.legend(loc = "upper left", bbox_to_anchor = (1.0, 1.0), fontsize = "medium");
        # plt.text(1.1 * max(X_pca[:, 0]), min(X_pca[:, 1]) - 0.2, '\n'.join(pretty_labels(features_radiomics)), size = "small")
        plt.savefig(analyzes_dir + f"pca/pca_radiomics_{name_set}.png", dpi = 480, bbox_inches='tight',
                        facecolor = "white", transparent = False)
        plt.close()

