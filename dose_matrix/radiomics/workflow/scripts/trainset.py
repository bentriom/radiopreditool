
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import zscore
from scipy.cluster.hierarchy import dendrogram

import matplotlib.pyplot as plt
import seaborn as sns

from multiprocessing import Pool, cpu_count
import os, sys, importlib, string, re
sys.path.append("radiomics/workflow/scripts")

import csv2nii, feature_extractor, check_dataset, trainset
from lifelines import CoxPHFitter
import scipy.cluster.hierarchy as hac
from sklearn.cluster import AgglomerativeClustering

# Create trainset
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

def create_trainset(file_radiomics, file_fccss_clinical, analyzes_dir, clinical_variables):
    df_radiomics = pd.read_csv(file_radiomics)
    df_radiomics["has_radiomics"] = 1
    df_fccss = pd.read_csv(file_fccss_clinical, low_memory = False)
    cols_date = ["date_sortie", "datederm", "date_dvigr", "date_rep", "date_rep2", "cslt_date_cslt", "date_diag"]
    cols_survival = ["numcent", "Pathologie_cardiaque", "deces", "date_pathol_cardiaque", "date_deces"] + cols_date 
    df_survival = df_fccss[cols_survival]
    df_fccss["survival_date"] = df_survival.apply(survival_date, axis = 1)
    df_fccss["datetime_date_diag"] = df_fccss["date_diag"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y"))
    df_fccss["survival_time_years"] = df_fccss[["survival_date", "datetime_date_diag"]].apply(lambda x: (x["survival_date"] - x["datetime_date_diag"]).total_seconds() / (365.25 * 24 * 3600), axis = 1)
    cols_radiomics = df_radiomics.columns.to_list()
    cols_radiomics.remove("ctr"), cols_radiomics.remove("numcent")
    df_trainset = df_fccss[["ctr", "numcent"] + ["Pathologie_cardiaque", "survival_time_years"] + clinical_variables]
    df_trainset = df_trainset.merge(df_radiomics, how = "left", on = ["ctr", "numcent"])
    df_trainset.loc[pd.isnull(df_trainset["has_radiomics"]), "has_radiomics"] = 0
    df_trainset.to_csv(analyzes_dir + "trainset.csv", index = False)

# Preprocessing
def filter_nan_values_radiomics(df_covariates_nan_values, features_radiomics, threshold):
    nbr_newdosi_patients = df_covariates_nan_values.shape[0]
    prop_values_cols = df_covariates_nan_values.apply(lambda col_series: 1 - sum(pd.isnull(col_series))/nbr_newdosi_patients)
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

def hclust_corr(df_covariates_hclust, threshold, do_plot = False, analyzes_dir = "./"):
    hclust = AgglomerativeClustering(n_clusters = None, linkage = "complete", 
                                     affinity = "precomputed", distance_threshold = 1 - threshold)
    distance_matrix = df_covariates_hclust.corr(method = "kendall").apply(lambda x: 1 - abs(x)).to_numpy()
    y_clusters = hclust.fit_predict(distance_matrix)
    if do_plot:
        fig = plt.figure(figsize=(8,10))
        plt.title(f"Hierarchical Clustering Dendrogram (threshold: {threshold} - number of features: {hclust.n_clusters_})")
        # plot the top three levels of the dendrogram
        plot_dendrogram(hclust, orientation = "right", labels = df_covariates_hclust.columns, leaf_font_size = 3)
        plt.axvline(1 - threshold)
        plt.savefig(f"{analyzes_dir}corr_plots/hclust_{threshold}.png", dpi = 480, bbox_inches='tight', 
                    facecolor = "white", transparent = False)
    return df_covariates_hclust.columns.values, y_clusters

def get_nbr_features_hclust(df_covariates_hclust, threshold, do_plot = False):
    _, y_clusters = hclust_corr(df_covariates_hclust, threshold, do_plot)
    return len(np.unique(y_clusters))

def filter_corr_hclust(df_trainset, df_covariates_hclust, corr_threshold, analyzes_dir):
    all_features_radiomics, y_clusters = hclust_corr(df_covariates_hclust, corr_threshold, do_plot = True, analyzes_dir = analyzes_dir) 
    filter_2_cols_radiomics = []
    id_clusters = np.unique(y_clusters)
    df_survival = df_trainset.copy()
    for c in id_clusters:
        list_features = [f for (idx, f) in enumerate(all_features_radiomics)  if y_clusters[idx] == c]
        list_pvalues = []
        for feature in list_features:
            df_survival = df_survival[~pd.isnull(df_survival[feature])]
            cph_feature = CoxPHFitter(penalizer = 0.0001)
            cph_feature.fit(df_survival, step_size = 0.5, duration_col = "survival_time_years", 
                            event_col = "Pathologie_cardiaque", formula = feature)
            pvalue = cph_feature.summary.loc[feature, "p"]
            list_pvalues.append(pvalue)
        selected_feature = list_features[np.argmin(list_pvalues)]
        filter_2_cols_radiomics.append(selected_feature)
    return filter_2_cols_radiomics

def preprocessing(file_trainset, analyzes_dir):
    df_trainset = pd.read_csv(file_trainset)
    features_radiomics = [feature for feature in df_trainset.columns if re.match("[0-9]+_.*", feature)]
    labels_radiomics = np.unique([label.split('_')[0] for label in df_trainset.columns if re.match("[0-9]+_.*", label)])
    dict_features_per_label = {label: [col for col in df_trainset.columns if label in col] for label in labels_radiomics}
    df_covariates_with_radiomics = df_trainset.loc[df_trainset["has_radiomics"] == 1, features_radiomics]

    # First filter: eliminate features with enough missing values
    nan_values_threshold = 0.9
    filter_1_cols_radiomics = filter_nan_values_radiomics(df_covariates_with_radiomics, features_radiomics, nan_values_threshold)
    # Correlation heatmaps
    os.makedirs(analyzes_dir + "corr_plots", exist_ok = True)
    for label in labels_radiomics:
        df_corr = df_covariates_with_radiomics[dict_features_per_label[label]].corr(method = "kendall")
        plt.figure(figsize=(16, 6))
        heatmap = sns.heatmap(df_corr, vmin=-1, vmax=1, annot=True)
        plt.savefig(f"{analyzes_dir}corr_plots/mat_corr_{label}.png", dpi = 480, bbox_inches='tight', 
                    facecolor = "white", transparent = False)
    # Second filter: eliminate very correlated features with hierarchical clustering + univariate Cox
    df_covariates_hclust = df_covariates_with_radiomics[filter_1_cols_radiomics]
    corr_threshold = 0.85
    filter_2_cols_radiomics = filter_corr_hclust(df_trainset, df_covariates_hclust, corr_threshold, analyzes_dir)
    preprocessed_cols = [feature for feature in df_trainset.columns if not re.match("[0-9]+_.*", feature) or feature in filter_2_cols_radiomics]
    df_trainset[preprocessed_cols].to_csv(analyzes_dir + "preprocessed_trainset.csv", index = False)

