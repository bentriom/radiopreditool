
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from scipy.stats import zscore
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as hac

import matplotlib.pyplot as plt
import seaborn as sns

from multiprocessing import Pool, cpu_count
import os, sys, importlib, string, re
sys.path.append("radiomics/workflow/scripts")

import csv2nii, feature_extractor, check_dataset, trainset
from radiopreditool_utils import *
from lifelines import CoxPHFitter
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Create trainset
def survival_date(event_col, date_event_col, row):
    if row["numcent"] == 199103047:
        return datetime.strptime("03/11/2019", "%d/%m/%Y")
    if row[event_col] == 1:
        return datetime.strptime(row[date_event_col], "%d/%m/%Y")
    if row["deces"] == 1:
        if pd.isna(row["date_deces"]):
            return datetime.strptime(row["date_sortie"], "%d/%m/%Y")
        else:
            return datetime.strptime(row["date_deces"], "%d/%m/%Y")
    else:
        cols_date = ["date_sortie", "datederm", "date_dvigr", "date_rep", "date_rep2", "cslt_date_cslt", "date_diag"]
        min_date = datetime.strptime("31/12/2016", "%d/%m/%Y")
        return max([datetime.strptime(row[col], "%d/%m/%Y") for col in cols_date if not pd.isna(row[col])] + [min_date])

def create_trainset(file_radiomics, file_fccss_clinical, analyzes_dir, clinical_variables, event_col, date_event_col,
                    test_size = 0.3, seed = None):
    logger = setup_logger("trainset", analyzes_dir + "trainset.log")
    df_radiomics = pd.read_csv(file_radiomics)
    df_radiomics["has_radiomics"] = 1
    df_fccss = pd.read_csv(file_fccss_clinical, low_memory = False)
    cols_date = ["date_sortie", "datederm", "date_dvigr", "date_rep", "date_rep2", "cslt_date_cslt", "date_diag"]
    cols_survival = ["numcent", event_col, "deces", date_event_col, "date_deces"] + cols_date 
    df_survival = df_fccss[cols_survival]
    df_fccss["survival_date"] = df_survival.apply(lambda x: survival_date(event_col, date_event_col, x), axis = 1)
    df_fccss["datetime_date_diag"] = df_fccss["date_diag"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y"))
    surv_duration_col = "survival_time_years"
    df_fccss[surv_duration_col] = df_fccss[["survival_date", "datetime_date_diag"]].apply(lambda x: (x["survival_date"] - x["datetime_date_diag"]).total_seconds() / (365.25 * 24 * 3600), axis = 1)
    cols_radiomics = df_radiomics.columns.to_list()
    cols_radiomics.remove("ctr"), cols_radiomics.remove("numcent")
    # Create dataset
    cols_y = [event_col, surv_duration_col]
    col_treated_by_rt = "radiotherapie_1K"
    df_dataset = df_fccss[["ctr", "numcent"] + clinical_variables + cols_y]
    if col_treated_by_rt not in clinical_variables:
        df_dataset[col_treated_by_rt] = df_fccss[col_treated_by_rt]
    df_dataset = df_dataset.merge(df_radiomics, how = "left", on = ["ctr", "numcent"])
    # Fill columns about radiotherapie
    df_dataset.loc[pd.isnull(df_dataset["has_radiomics"]), "has_radiomics"] = 0
    features_radiomics = [feature for feature in df_dataset.columns if re.match("[0-9]+_.*", feature)]
    df_dataset.loc[df_dataset[col_treated_by_rt] == 0, features_radiomics] = 0
    logger.info(f"Full dataset: {df_dataset.shape}")
    # Split train / test
    df_X = df_dataset[[c for c in df_dataset.columns if c not in cols_y]]
    df_y = df_dataset[["ctr", "numcent"] + cols_y]
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y,
                                                                    random_state = seed, shuffle = True,
                                                                    test_size = test_size, stratify = df_y[event_col])
    df_trainset = df_X_train.merge(df_y_train, how = "inner", on = ["ctr", "numcent"]) 
    df_testset = df_X_test.merge(df_y_test, how = "inner", on = ["ctr", "numcent"]) 
    df_trainset_radiomics = df_trainset.loc[df_trainset["has_radiomics"] == 1,:]
    df_testset_radiomics = df_testset.loc[df_testset["has_radiomics"] == 1,:]
    logger.info(f"Trainset: {df_trainset.shape}")
    logger.info(f"Trainset with radiomics features: {df_trainset_radiomics.shape}")
    logger.info(f"Testset: {df_testset.shape} ({test_size * 100}%)")
    logger.info(f"Testset with radiomics features: {df_testset_radiomics.shape}")
    nsamples_train = df_trainset.shape[0]
    nsamples_test = df_testset.shape[0]
    logger.info(f"Balance train/test event: {df_trainset[event_col].sum()/nsamples_train} {df_testset[event_col].sum()/nsamples_test}")
    logger.info(f"Balance train/test treated by RT: {df_trainset[col_treated_by_rt].sum()/nsamples_train} {df_testset[col_treated_by_rt].sum()/nsamples_test}")
    logger.info(f"Balance train/test that has radiomics features: {df_trainset['has_radiomics'].sum()/nsamples_train} {df_testset['has_radiomics'].sum()/nsamples_test}")
    # Save
    if col_treated_by_rt not in clinical_variables:
        df_dataset.drop(columns = col_treated_by_rt, inplace = True)
        df_trainset.drop(columns = col_treated_by_rt, inplace = True)
        df_testset.drop(columns = col_treated_by_rt, inplace = True)
    df_trainset.to_csv(analyzes_dir + "trainset.csv.gz", index = False)
    df_testset.to_csv(analyzes_dir + "testset.csv.gz", index = False)

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

def filter_corr_hclust(df_trainset, df_covariates_hclust, corr_threshold, event_col, surv_duration_col, analyzes_dir):
    all_features_radiomics, y_clusters = hclust_corr(df_covariates_hclust, corr_threshold, do_plot = True, analyzes_dir = analyzes_dir) 
    filter_2_cols_radiomics = []
    id_clusters = np.unique(y_clusters)
    df_survival = df_trainset.copy()
    for c in id_clusters:
        list_features = [f for (idx, f) in enumerate(all_features_radiomics)  if y_clusters[idx] == c]
        list_pvalues = []
        df_survival.dropna(subset = list_features, inplace = True)
        for feature in list_features:
            cph_feature = CoxPHFitter(penalizer = 0.0001)
            df_survival.loc[:,feature] = StandardScaler().fit_transform(df_survival[[feature]])
            cph_feature.fit(df_survival, step_size = 0.5, duration_col = surv_duration_col,
                            event_col = event_col, formula = feature)
            pvalue = cph_feature.summary.loc[feature, "p"]
            list_pvalues.append(pvalue)
        selected_feature = list_features[np.argmin(list_pvalues)]
        filter_2_cols_radiomics.append(selected_feature)
    return filter_2_cols_radiomics

# Eliminate sparse and redundant columns
def preprocessing(file_trainset, event_col, analyzes_dir):
    df_trainset = pd.read_csv(file_trainset)
    features_radiomics = [feature for feature in df_trainset.columns if re.match("[0-9]+_.*", feature)]
    labels_radiomics = np.unique([label.split('_')[0] for label in df_trainset.columns if re.match("[0-9]+_.*", label)])
    dict_features_per_label = {label: [col for col in df_trainset.columns if re.match(f"{label}_", col)] for label in labels_radiomics}
    df_covariates_with_radiomics = df_trainset.loc[df_trainset["has_radiomics"] == 1, features_radiomics]
    logger = setup_logger("preprocessing", analyzes_dir + "preprocessing.log")
    logger.info(f"Trainset dataframe: {df_trainset.shape}")
    logger.info(f"Initial number of radiomics covariates: {df_covariates_with_radiomics.shape[1]}")
    # First filter: eliminate features with enough missing values
    nan_values_threshold = 0.9
    filter_1_cols_radiomics = filter_nan_values_radiomics(df_covariates_with_radiomics, features_radiomics, nan_values_threshold)
    logger.info(f"After the first filter (non nan values > {nan_values_threshold}): {len(filter_1_cols_radiomics)}")
    # Correlation heatmaps
    os.makedirs(analyzes_dir + "corr_plots", exist_ok = True)
    for label in labels_radiomics:
        df_corr = df_covariates_with_radiomics[dict_features_per_label[label]].corr(method = "kendall")
        fig = plt.figure(figsize=(16, 6))
        heatmap = sns.heatmap(df_corr, vmin = -1, vmax = 1, annot = True, center = 0, cmap = "vlag")
        plt.savefig(f"{analyzes_dir}corr_plots/mat_corr_{label}.png", dpi = 480, bbox_inches='tight', 
                    facecolor = "white", transparent = False)
    # Second filter: eliminate very correlated features with hierarchical clustering + univariate Cox
    df_covariates_hclust = df_covariates_with_radiomics[filter_1_cols_radiomics]
    # Eliminate rows with na
    df_covariates_hclust = df_covariates_hclust.dropna()
    # Eliminate columns with unique values
    unique_cols = [c for c in df_covariates_hclust.columns if len(df_covariates_hclust[c].unique()) == 1]
    df_covariates_hclust.drop(columns = unique_cols, inplace = True)
    corr_threshold = 0.85
    surv_duration_col = "survival_time_years"
    logger.info(f"Hclust on dataframe: {df_covariates_hclust.shape}")
    filter_2_cols_radiomics = filter_corr_hclust(df_trainset, df_covariates_hclust, corr_threshold, event_col, surv_duration_col, analyzes_dir)
    logger.info(f"After the second filter (hclust): {len(filter_2_cols_radiomics)}")
    preprocessed_cols = [feature for feature in df_trainset.columns if not re.match("[0-9]+_.*", feature) or feature in filter_2_cols_radiomics]
    df_trainset[preprocessed_cols].to_csv(analyzes_dir + "preprocessed_trainset.csv.gz", index = False)

def col_class_event(event_col, delay, row):
    if row[event_col] == 0:
        return 0
    if row[event_col] == 1 and row["survival_time_years"] <= delay:
        return 1
    else:
        return 0

def pca_viz(file_trainset, event_col, analyzes_dir):
    logger = setup_logger("pca", analyzes_dir + "pca_viz.log")
    df_trainset = pd.read_csv(file_trainset)
    df_trainset["class_" + event_col] = 0
    year_max = 20
    for delay_event in range(5, year_max + 1, 5):
        col_class = f"{event_col}_at_{delay_event}"
        df_trainset[col_class] = df_trainset.apply(lambda row: col_class_event(event_col, delay_event, row), axis = 1)
        df_trainset.loc[:, "class_" + event_col] += df_trainset[col_class]
    labels = get_all_labels(df_trainset)
    names_sets = ["all"] + labels 
    set_of_features_radiomics = [get_all_radiomics_features(df_trainset)] + [[feature for feature in df_trainset.columns if re.match(f"{label}_.*", feature)] for label in labels]
    for (i, features_radiomics) in enumerate(set_of_features_radiomics):
        logger.info(f"PCA {names_sets[i]}")
        logger.info(f"Covariates: {features_radiomics}")
        df_subset = df_trainset.dropna(subset = features_radiomics + [event_col, "survival_time_years"])
        X = StandardScaler().fit_transform(df_subset[features_radiomics])
        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(X)
        y = df_subset["class_" + event_col] 
        nbr_classes = len(df_subset["class_" + event_col].unique())
        list_labels = ["No event" if i == 0 else f"Event {year_max - (5*i)} - {year_max - 5*(i-1)} years" for i in range(nbr_classes)]
        list_colors = plt.cm.get_cmap('Set1', nbr_classes).colors
        list_markers = ['o' if i == 0 else 'x' for i in range(nbr_classes)]
        list_alphas = [0.5 if i == 0 else 1 for i in range(nbr_classes)]
        plt.figure()
        for i in range(nbr_classes):
            idx_cluster = (y == i)
            plt.scatter(X_pca[idx_cluster, 0], X_pca[idx_cluster, 1], label = list_labels[i],
                        alpha = list_alphas[i], marker = list_markers[i], color = list_colors[i])
        plt.xlabel('PCA component 1')
        plt.ylabel('PCA component 2')
        plt.title(f"PCA ({round(sum(pca.explained_variance_ratio_), 3)} explained variance ratio)")
        plt.legend(loc = "upper left", bbox_to_anchor = (1.0, 1.0), fontsize = "medium");
        os.makedirs(analyzes_dir + "viz", exist_ok=True)
        plt.savefig(analyzes_dir + f"viz/pca_radiomics_{names_sets[i]}.png", dpi = 480, bbox_inches='tight', 
                        facecolor = "white", transparent = False)

