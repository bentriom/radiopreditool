import pandas as pd
import plotly.express as px
import os

def generate_line_set(df, set_type):
    return f"${df.loc['C-index',set_type]:.3f}$ & ${df.loc['IPCW C-index',set_type]:.3f}$ & ${df.loc['BS at 60',set_type]:.3f}$ & ${df.loc['IBS',set_type]:.3f}$"

def generate_line_multiple(df):
#     return f"${df.loc['C-index','Mean']:.3f} \pm {df.loc['C-index','Std']:.3f}$ & \
#             ${df.loc['IPCW C-index','Mean']:.3f} \pm {df.loc['IPCW C-index','Std']:.3f}$ & \
#             ${df.loc['BS at 60','Mean']:.3f} \pm {df.loc['BS at 60','Std']:.3f}$ & \
#             ${df.loc['IBS','Mean']:.3f} \pm {df.loc['IBS','Std']:.3f}$"
    return f"${df.loc['C-index','Mean']:.3f} \pm {df.loc['C-index','Std']:.3f}$ & \
             ${df.loc['IBS','Mean']:.3f} \pm {df.loc['IBS','Std']:.3f}$"

def get_color(model):
    splits = model.split()
    if splits[0] == "RSF":
        return "Random Survival Forest"
    if splits[0] == "Cox":
        if splits[1] == "Lasso":
            return "Cox PH with Lasso penalty"
        elif splits[1] == "Bootstrap":
            return "Cox PH with Bootstrap Lasso"
        else:
            return "Cox PH"

#for model in ["pathol_cardiaque_grade3_drugs_iccc_other"]:
local_results_dir = "/media/moud/LaCie/local_results/analyzes/"
slurm_results_dir = "../../slurm_results/analyzes/"
list_binwidth = [0.1, 0.2, 0.5, 1.0]
# list_models = [(local_results_dir, f"pathol_cardiaque_grade3_drugs_iccc_other_bw_{binw}") for binw in list_binwidth]
# list_models += [(local_results_dir, f"pathol_cardiaque_grade3_drugs_iccc_other_bw_{binw}_filter_entropy") \
#                for binw in list_binwidth]
list_models = [(slurm_results_dir, f"pathol_cardiaque_grade3_drugs_iccc_other_bw_0.5")]
for results_dir, model in list_models:
    coxph_results_dir = f"{results_dir}{model}/coxph_R/"
    rsf_results_dir = f"{results_dir}{model}/rsf/"
    os.makedirs(f"tables/{model}", exist_ok = True)
    os.makedirs(f"plots/{model}", exist_ok = True)

    ## Loading results
    # Reference models
    df_cox_mean = pd.read_csv(coxph_results_dir + "1320_mean/metrics.csv", index_col = 0)
    df_dosesvol = pd.read_csv(coxph_results_dir + "1320_dosesvol/metrics.csv", index_col = 0)
    df_dosesvol_lasso = pd.read_csv(coxph_results_dir + "1320_dosesvol_lasso/metrics.csv", index_col = 0)
    df_rsf_dosesvol = pd.read_csv(rsf_results_dir + "1320_dosesvol/metrics.csv", index_col = 0)
    # Firstorder radiomics Cox Lasso
    df_cox_1320_firstorder = pd.read_csv(coxph_results_dir + "1320_radiomics_firstorder_lasso_all/metrics.csv", index_col = 0)
    df_cox_1320_filter_firstorder = pd.read_csv(coxph_results_dir + "1320_radiomics_firstorder_lasso_features_hclust_corr/metrics.csv", index_col = 0)
    df_cox_32X_firstorder = pd.read_csv(coxph_results_dir + "32X_radiomics_firstorder_lasso_all/metrics.csv", index_col = 0)
    df_cox_32X_filter_firstorder = pd.read_csv(coxph_results_dir + "32X_radiomics_firstorder_lasso_features_hclust_corr/metrics.csv", index_col = 0)
    # Full radiomics Cox Lasso
    df_cox_1320 = pd.read_csv(coxph_results_dir + "1320_radiomics_full_lasso_all/metrics.csv", index_col = 0)
    df_cox_1320_filter = pd.read_csv(coxph_results_dir + "1320_radiomics_full_lasso_features_hclust_corr/metrics.csv", index_col = 0)
    df_cox_32X = pd.read_csv(coxph_results_dir + "32X_radiomics_full_lasso_all/metrics.csv", index_col = 0)
    df_cox_32X_filter = pd.read_csv(coxph_results_dir + "32X_radiomics_full_lasso_features_hclust_corr/metrics.csv", index_col = 0)
    # Firstorder radiomics Cox Bootstrap Lasso
    df_cox_boot_1320_firstorder = pd.read_csv(coxph_results_dir + "1320_radiomics_firstorder_bootstrap_lasso_all/metrics.csv", index_col = 0)
    df_cox_boot_1320_filter_firstorder = pd.read_csv(coxph_results_dir + "1320_radiomics_firstorder_bootstrap_lasso_features_hclust_corr/metrics.csv", index_col = 0)
    df_cox_boot_32X_firstorder = pd.read_csv(coxph_results_dir + "32X_radiomics_firstorder_bootstrap_lasso_all/metrics.csv", index_col = 0)
    df_cox_boot_32X_filter_firstorder = pd.read_csv(coxph_results_dir + "32X_radiomics_firstorder_bootstrap_lasso_features_hclust_corr/metrics.csv", index_col = 0)
    # Full radiomics Cox Bootstrap Lasso
    df_cox_boot_1320 = pd.read_csv(coxph_results_dir + "1320_radiomics_full_bootstrap_lasso_all/metrics.csv", index_col = 0)
    df_cox_boot_1320_filter = pd.read_csv(coxph_results_dir + "1320_radiomics_full_bootstrap_lasso_features_hclust_corr/metrics.csv", index_col = 0)
    df_cox_boot_32X = pd.read_csv(coxph_results_dir + "32X_radiomics_full_bootstrap_lasso_all/metrics.csv", index_col = 0)
    df_cox_boot_32X_filter = pd.read_csv(coxph_results_dir + "32X_radiomics_full_bootstrap_lasso_features_hclust_corr/metrics.csv", index_col = 0)
    # Firstorder radiomics RSF
    df_rsf_1320_firstorder = pd.read_csv(rsf_results_dir + "1320_radiomics_firstorder_all/metrics.csv", index_col = 0)
    df_rsf_1320_filter_firstorder = pd.read_csv(rsf_results_dir + "1320_radiomics_firstorder_features_hclust_corr/metrics.csv", index_col = 0)
    df_rsf_32X_firstorder = pd.read_csv(rsf_results_dir + "32X_radiomics_firstorder_all/metrics.csv", index_col = 0)
    df_rsf_32X_filter_firstorder = pd.read_csv(rsf_results_dir + "32X_radiomics_firstorder_features_hclust_corr/metrics.csv", index_col = 0)
    # Full radiomics RSF
    df_rsf_1320 = pd.read_csv(rsf_results_dir + "1320_radiomics_full_all/metrics.csv", index_col = 0)
    df_rsf_1320_filter = pd.read_csv(rsf_results_dir + "1320_radiomics_full_features_hclust_corr/metrics.csv", index_col = 0)
    df_rsf_32X = pd.read_csv(rsf_results_dir + "32X_radiomics_full_all/metrics.csv", index_col = 0)
    df_rsf_32X_filter = pd.read_csv(rsf_results_dir + "32X_radiomics_full_features_hclust_corr/metrics.csv", index_col = 0)

    nb_estim = 10
    df_multiple_cox_mean = pd.read_csv(coxph_results_dir + f"1320_mean/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_dosesvol = pd.read_csv(coxph_results_dir + f"1320_dosesvol/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_dosesvol_lasso = pd.read_csv(coxph_results_dir + f"1320_dosesvol_lasso/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_rsf_dosesvol = pd.read_csv(rsf_results_dir + f"1320_dosesvol/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    # Firstorder radiomics Cox Lasso
    df_multiple_cox_1320_firstorder = pd.read_csv(coxph_results_dir + f"1320_radiomics_firstorder_lasso_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_1320_filter_firstorder = pd.read_csv(coxph_results_dir + f"1320_radiomics_firstorder_lasso_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_32X_firstorder = pd.read_csv(coxph_results_dir + f"32X_radiomics_firstorder_lasso_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_32X_filter_firstorder = pd.read_csv(coxph_results_dir + f"32X_radiomics_firstorder_lasso_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    # Full radiomics Cox Lasso
    df_multiple_cox_1320 = pd.read_csv(coxph_results_dir + f"1320_radiomics_full_lasso_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_1320_filter = pd.read_csv(coxph_results_dir + f"1320_radiomics_full_lasso_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_32X = pd.read_csv(coxph_results_dir + f"32X_radiomics_full_lasso_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_32X_filter = pd.read_csv(coxph_results_dir + f"32X_radiomics_full_lasso_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    # Firstorder radiomics Cox Bootstrap Lasso
    df_multiple_cox_boot_1320_firstorder = pd.read_csv(coxph_results_dir + f"1320_radiomics_firstorder_bootstrap_lasso_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_boot_1320_filter_firstorder = pd.read_csv(coxph_results_dir + f"1320_radiomics_firstorder_bootstrap_lasso_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_boot_32X_firstorder = pd.read_csv(coxph_results_dir + f"32X_radiomics_firstorder_bootstrap_lasso_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_boot_32X_filter_firstorder = pd.read_csv(coxph_results_dir + f"32X_radiomics_firstorder_bootstrap_lasso_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    # Full radiomics Cox Bootstrap Lasso
    df_multiple_cox_boot_1320 = pd.read_csv(coxph_results_dir + f"1320_radiomics_full_bootstrap_lasso_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_boot_1320_filter = pd.read_csv(coxph_results_dir + f"1320_radiomics_full_bootstrap_lasso_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_boot_32X = pd.read_csv(coxph_results_dir + f"32X_radiomics_full_bootstrap_lasso_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_cox_boot_32X_filter = pd.read_csv(coxph_results_dir + f"32X_radiomics_full_bootstrap_lasso_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    # Firstorder radiomics RSF
    df_multiple_rsf_1320_firstorder = pd.read_csv(rsf_results_dir + f"1320_radiomics_firstorder_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_rsf_1320_filter_firstorder = pd.read_csv(rsf_results_dir + f"1320_radiomics_firstorder_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_rsf_32X_firstorder = pd.read_csv(rsf_results_dir + f"32X_radiomics_firstorder_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_rsf_32X_filter_firstorder = pd.read_csv(rsf_results_dir + f"32X_radiomics_firstorder_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    # Full radiomics RSF
    df_multiple_rsf_1320 = pd.read_csv(rsf_results_dir + f"1320_radiomics_full_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_rsf_1320_filter = pd.read_csv(rsf_results_dir + f"1320_radiomics_full_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_rsf_32X = pd.read_csv(rsf_results_dir + f"32X_radiomics_full_all/{nb_estim}_runs_test_metrics.csv", index_col = 0)
    df_multiple_rsf_32X_filter = pd.read_csv(rsf_results_dir + f"32X_radiomics_full_features_hclust_corr/{nb_estim}_runs_test_metrics.csv", index_col = 0)

    ## Writing latex tables
    table_train = "\\begin{tabular}{|c|c|c|c|c|}\n"
    table_train += "\\hline\n"
    table_train += "Model  & C-index & IPCW C-index & BS at 60 & IBS \\\\ \\hline\n"
    table_train += f"Cox mean heart dose & {generate_line_set(df_cox_mean, 'Train')} \\\\ \\hline\n"
    table_train += f"Cox doses volumes & {generate_line_set(df_dosesvol, 'Train')} \\\\ \\hline\n"
    table_train += f"Cox Lasso doses volumes & {generate_line_set(df_dosesvol_lasso, 'Train')}  \\\\ \\hline\n"
    table_train += f"RSF doses volumes & {generate_line_set(df_rsf_dosesvol, 'Train')}  \\\\ \\hline\n"
    # Firstorder radiomics Cox Lasso
    table_train += f"Cox Lasso firstorder whole heart (AllTrain) & {generate_line_set(df_cox_1320_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso firstorder whole heart (FETrain) & {generate_line_set(df_cox_1320_filter_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso firstorder subparts of heart (AllTrain) & {generate_line_set(df_cox_32X_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso firstorder subparts of heart (FETrain) & {generate_line_set(df_cox_32X_filter_firstorder, 'Train')}  \\\\ \\hline\n"
    # Full radiomics Cox Lasso
    table_train += f"Cox Lasso whole heart (AllTrain) & {generate_line_set(df_cox_1320, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso whole heart (FETrain) & {generate_line_set(df_cox_1320_filter, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso subparts of heart (AllTrain) & {generate_line_set(df_cox_32X, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso subparts of heart (FETrain) & {generate_line_set(df_cox_32X_filter, 'Train')}  \\\\ \\hline\n"
    # Firstorder radiomics Cox Bootstrap Lasso
    table_train += f"Cox Bootstrap Lasso firstorder whole heart (AllTrain) & {generate_line_set(df_cox_boot_1320_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Bootstrap Lasso firstorder whole heart (FETrain) & {generate_line_set(df_cox_boot_1320_filter_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Bootstrap Lasso firstorder subparts of heart (AllTrain) & {generate_line_set(df_cox_boot_32X_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Bootstrap Lasso firstorder subparts of heart (FETrain) & {generate_line_set(df_cox_boot_32X_filter_firstorder, 'Train')}  \\\\ \\hline\n"
    # Full radiomics Cox Bootstrap Lasso
    table_train += f"Cox Bootstrap Lasso whole heart (AllTrain) & {generate_line_set(df_cox_boot_1320, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Bootstrap Lasso whole heart (FETrain) & {generate_line_set(df_cox_boot_1320_filter, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Bootstrap Lasso subparts of heart (AllTrain) & {generate_line_set(df_cox_boot_32X, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Bootstrap Lasso subparts of heart (FETrain) & {generate_line_set(df_cox_boot_32X_filter, 'Train')}  \\\\ \\hline\n"
    # Firstorder radiomics Cox
    table_train += f"RSF whole heart firstorder (AllTrain) & {generate_line_set(df_rsf_1320_firstorder, 'Train')} \\\\ \\hline\n"
    table_train += f"RSF whole heart firstorder (FETrain) &{ generate_line_set(df_rsf_1320_filter_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"RSF subparts of heart firstorder (AllTrain) & {generate_line_set(df_rsf_32X_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"RSF subparts of heart firstorder (FETrain) & {generate_line_set(df_rsf_32X_filter_firstorder, 'Train')}  \\\\ \\hline\n"
    # Full radiomics Cox
    table_train += f"RSF whole heart (AllTrain) & {generate_line_set(df_rsf_1320, 'Train')} \\\\ \\hline\n"
    table_train += f"RSF whole heart (FETrain) &{ generate_line_set(df_rsf_1320_filter, 'Train')}  \\\\ \\hline\n"
    table_train += f"RSF subparts of heart (AllTrain) & {generate_line_set(df_rsf_32X, 'Train')}  \\\\ \\hline\n"
    table_train += f"RSF subparts of heart (FETrain) & {generate_line_set(df_rsf_32X_filter, 'Train')}  \\\\ \\hline\n"
    table_train += "\\end{tabular}"

    table_test = "\\begin{tabular}{|c|c|c|c|c|}\n"
    table_test += "\\hline\n"
    table_test += "Model  & C-index & IPCW C-index & BS at 60 & IBS \\\\ \\hline\n"
    table_test += f"Cox mean heart dose & {generate_line_set(df_cox_mean, 'Test')} \\\\ \\hline\n"
    table_test += f"Cox doses volumes & {generate_line_set(df_dosesvol, 'Test')} \\\\ \\hline\n"
    table_test += f"Cox Lasso doses volumes & {generate_line_set(df_dosesvol_lasso, 'Test')}  \\\\ \\hline\n"
    table_test += f"RSF doses volumes & {generate_line_set(df_rsf_dosesvol, 'Test')}  \\\\ \\hline\n"
    # Firstorder radiomics Cox
    table_test += f"Cox Lasso firstorder whole heart (AllTrain) & {generate_line_set(df_cox_1320_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso firstorder whole heart (FETrain) & {generate_line_set(df_cox_1320_filter_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso firstorder subparts of heart (AllTrain) & {generate_line_set(df_cox_32X_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso firstorder subparts of heart (FETrain) & {generate_line_set(df_cox_32X_filter_firstorder, 'Test')}  \\\\ \\hline\n"
    # Full radiomics Cox
    table_test += f"Cox Lasso whole heart (AllTrain) & {generate_line_set(df_cox_1320, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso whole heart (FETrain) & {generate_line_set(df_cox_1320_filter, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso subparts of heart (AllTrain) & {generate_line_set(df_cox_32X, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso subparts of heart (FETrain) & {generate_line_set(df_cox_32X_filter, 'Test')}  \\\\ \\hline\n"
    # Firstorder radiomics Cox Bootstrap Lasso
    table_test += f"Cox Bootstrap Lasso firstorder whole heart (AllTrain) & {generate_line_set(df_cox_boot_1320_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Bootstrap Lasso firstorder whole heart (FETrain) & {generate_line_set(df_cox_boot_1320_filter_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Bootstrap Lasso firstorder subparts of heart (AllTrain) & {generate_line_set(df_cox_boot_32X_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Bootstrap Lasso firstorder subparts of heart (FETrain) & {generate_line_set(df_cox_boot_32X_filter_firstorder, 'Test')}  \\\\ \\hline\n"
    # Full radiomics Cox Bootstrap Lasso
    table_test += f"Cox Bootstrap Lasso whole heart (AllTrain) & {generate_line_set(df_cox_boot_1320, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Bootstrap Lasso whole heart (FETrain) & {generate_line_set(df_cox_boot_1320_filter, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Bootstrap Lasso subparts of heart (AllTrain) & {generate_line_set(df_cox_boot_32X, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Bootstrap Lasso subparts of heart (FETrain) & {generate_line_set(df_cox_boot_32X_filter, 'Test')}  \\\\ \\hline\n"
    # Firstorder radiomics RSF
    table_test += f"RSF whole heart firstorder (AllTrain) & {generate_line_set(df_rsf_1320_firstorder, 'Test')} \\\\ \\hline\n"
    table_test += f"RSF whole heart firstorder (FETrain) &{ generate_line_set(df_rsf_1320_filter_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"RSF subparts of heart firstorder (AllTrain) & {generate_line_set(df_rsf_32X_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"RSF subparts of heart firstorder (FETrain) & {generate_line_set(df_rsf_32X_filter_firstorder, 'Test')}  \\\\ \\hline\n"
    # Full radiomics RSF
    table_test += f"RSF whole heart (AllTrain) & {generate_line_set(df_rsf_1320, 'Test')} \\\\ \\hline\n"
    table_test += f"RSF whole heart (FETrain) &{ generate_line_set(df_rsf_1320_filter, 'Test')}  \\\\ \\hline\n"
    table_test += f"RSF subparts of heart (AllTrain) & {generate_line_set(df_rsf_32X, 'Test')}  \\\\ \\hline\n"
    table_test += f"RSF subparts of heart (FETrain) & {generate_line_set(df_rsf_32X_filter, 'Test')}  \\\\ \\hline\n"
    table_test += "\\end{tabular}"

    #table_multiple = "\\begin{tabular}{|c|c|c|c|c|}\n"
    table_multiple = "\\begin{tabular}{|c|c|c|}\n"
    table_multiple += "\\hline\n"
    #table_multiple += "Model  & C-index & IPCW C-index & BS at 60 & IBS \\\\ \\hline\n"
    table_multiple += "Model  & C-index & IBS \\\\ \\hline\n"
    table_multiple += f"Cox mean heart dose & {generate_line_multiple(df_multiple_cox_mean)} \\\\ \\hline\n"
    table_multiple += f"Cox doses volumes & {generate_line_multiple(df_multiple_dosesvol)} \\\\ \\hline\n"
    table_multiple += f"Cox Lasso doses volumes & {generate_line_multiple(df_multiple_dosesvol_lasso)}  \\\\ \\hline\n"
    table_multiple += f"RSF doses volumes & {generate_line_multiple(df_multiple_rsf_dosesvol)}  \\\\ \\hline\n"
    # Firstorder radiomics Cox Lasso
    table_multiple += f"Cox Lasso whole heart firstorder (AllTrain) & {generate_line_multiple(df_multiple_cox_1320_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso whole heart firstorder (FETrain) & {generate_line_multiple(df_multiple_cox_1320_filter_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso subparts of heart firstorder (AllTrain) & {generate_line_multiple(df_multiple_cox_32X_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso subparts of heart firstorder (FETrain) & {generate_line_multiple(df_multiple_cox_32X_filter_firstorder)}  \\\\ \\hline\n"
    # Full radiomics Cox Lasso
    table_multiple += f"Cox Lasso whole heart (AllTrain) & {generate_line_multiple(df_multiple_cox_1320)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso whole heart (FETrain) & {generate_line_multiple(df_multiple_cox_1320_filter)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso subparts of heart (AllTrain) & {generate_line_multiple(df_multiple_cox_32X)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso subparts of heart (FETrain) & {generate_line_multiple(df_multiple_cox_32X_filter)}  \\\\ \\hline\n"
    # Firstorder radiomics Cox Bootstrap Lasso
    table_multiple += f"Cox Bootstrap Lasso whole heart firstorder (AllTrain) & {generate_line_multiple(df_multiple_cox_boot_1320_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Bootstrap Lasso whole heart firstorder (FETrain) & {generate_line_multiple(df_multiple_cox_boot_1320_filter_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Bootstrap Lasso subparts of heart firstorder (AllTrain) & {generate_line_multiple(df_multiple_cox_boot_32X_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Bootstrap Lasso subparts of heart firstorder (FETrain) & {generate_line_multiple(df_multiple_cox_boot_32X_filter_firstorder)}  \\\\ \\hline\n"
    # Full radiomics Cox Bootstrap Lasso
    table_multiple += f"Cox Bootstrap Lasso whole heart (AllTrain) & {generate_line_multiple(df_multiple_cox_boot_1320)}  \\\\ \\hline\n"
    table_multiple += f"Cox Bootstrap Lasso whole heart (FETrain) & {generate_line_multiple(df_multiple_cox_boot_1320_filter)}  \\\\ \\hline\n"
    table_multiple += f"Cox Bootstrap Lasso subparts of heart (AllTrain) & {generate_line_multiple(df_multiple_cox_boot_32X)}  \\\\ \\hline\n"
    table_multiple += f"Cox Bootstrap Lasso subparts of heart (FETrain) & {generate_line_multiple(df_multiple_cox_boot_32X_filter)}  \\\\ \\hline\n"
    # Firstorder radiomics RSF
    table_multiple += f"RSF whole heart firstorder (AllTrain) & {generate_line_multiple(df_multiple_rsf_1320_firstorder)} \\\\ \\hline\n"
    table_multiple += f"RSF whole heart firstorder (FETrain) &{ generate_line_multiple(df_multiple_rsf_1320_filter_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"RSF subparts of heart firstorder (AllTrain) & {generate_line_multiple(df_multiple_rsf_32X_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"RSF subparts of heart firstorder (FETrain) & {generate_line_multiple(df_multiple_rsf_32X_filter_firstorder)}  \\\\ \\hline\n"
    # Full radiomics RSF
    table_multiple += f"RSF whole heart (AllTrain) & {generate_line_multiple(df_multiple_rsf_1320)} \\\\ \\hline\n"
    table_multiple += f"RSF whole heart (FETrain) &{ generate_line_multiple(df_multiple_rsf_1320_filter)}  \\\\ \\hline\n"
    table_multiple += f"RSF subparts of heart (AllTrain) & {generate_line_multiple(df_multiple_rsf_32X)}  \\\\ \\hline\n"
    table_multiple += f"RSF subparts of heart (FETrain) & {generate_line_multiple(df_multiple_rsf_32X_filter)}  \\\\ \\hline\n"
    table_multiple += "\\end{tabular}"

    ## Plots
    map_model_type = {'Random Survival Forest': 'red', 'Cox PH with Lasso penalty': 'blue',
                      'Cox PH with Bootstrap Lasso': 'purple', 'Cox PH': 'green'}
    # Multiple results
    dict_results_multiple = {
        "Cox mean heart dose" : df_multiple_cox_mean,
        "Cox doses-volumes indicators": df_multiple_dosesvol,
        "Cox Lasso doses-volumes indicators": df_multiple_dosesvol_lasso,
        "RSF doses-volumes indicators": df_multiple_rsf_dosesvol,
        "Cox Lasso whole-heart first-order dosiomics": df_multiple_cox_1320_firstorder,
        "Cox Lasso screened whole-heart first-order dosiomics": df_multiple_cox_1320_filter_firstorder,
        "Cox Lasso heart's subparts first-order dosiomics": df_multiple_cox_32X_firstorder,
        "Cox Lasso heart's subparts screened first-order dosiomics": df_multiple_cox_32X_filter_firstorder,
        "Cox Lasso whole-heart dosiomics": df_multiple_cox_1320,
        "Cox Lasso screened whole-heart dosiomics": df_multiple_cox_1320_filter,
        "Cox Lasso heart's subparts dosiomics": df_multiple_cox_32X,
        "Cox Lasso heart's subparts screened dosiomics": df_multiple_cox_32X_filter,
        "Cox Bootstrap Lasso whole-heart first-order dosiomics": df_multiple_cox_boot_1320_firstorder,
        "Cox Bootstrap Lasso screened whole-heart first-order dosiomics": df_multiple_cox_boot_1320_filter_firstorder,
        "Cox Bootstrap Lasso heart's subparts first-order dosiomics": df_multiple_cox_boot_32X_firstorder,
        "Cox Bootstrap Lasso heart's subparts screened first-order dosiomics": df_multiple_cox_boot_32X_filter_firstorder,
        "Cox Bootstrap Lasso whole-heart dosiomics": df_multiple_cox_boot_1320,
        "Cox Bootstrap Lasso screened whole-heart dosiomics": df_multiple_cox_boot_1320_filter,
        "Cox Bootstrap Lasso heart's subparts dosiomics": df_multiple_cox_boot_32X,
        "Cox Bootstrap Lasso heart's subparts screened dosiomics": df_multiple_cox_boot_32X_filter,
        "RSF whole-heart first-order dosiomics": df_multiple_rsf_1320_firstorder,
        "RSF screened whole-heart first-order dosiomics": df_multiple_rsf_1320_filter_firstorder,
        "RSF heart's subparts first-order dosiomics": df_multiple_rsf_32X_firstorder,
        "RSF heart's subparts screened first-order dosiomics": df_multiple_rsf_32X_filter_firstorder,
        "RSF whole-heart dosiomics": df_multiple_rsf_1320,
        "RSF screened whole-heart dosiomics": df_multiple_rsf_1320_filter,
        "RSF heart's subparts dosiomics": df_multiple_rsf_32X,
        "RSF heart's subparts screened dosiomics": df_multiple_rsf_32X_filter}
    idx_res = dict_results_multiple.keys()
    df_results_multiple = pd.DataFrame(data = {"model": idx_res})
    df_results_multiple.set_index("model", drop = False, inplace = True)
    df_results_multiple.loc[idx_res, "mean_harrell"] = [dict_results_multiple[model].loc["C-index", "Mean"] for model in idx_res]
    df_results_multiple.loc[idx_res, "std_harrell"] = [dict_results_multiple[model].loc["C-index", "Std"] for model in idx_res]
    df_results_multiple.loc[idx_res, "mean_ipcw"] = [dict_results_multiple[model].loc["IPCW C-index", "Mean"] for model in idx_res]
    df_results_multiple.loc[idx_res, "std_ipcw"] = [dict_results_multiple[model].loc["IPCW C-index", "Std"] for model in idx_res]
    df_results_multiple.loc[:, "model type"] = df_results_multiple["model"].apply(get_color)
    # Harrell's C-index multiple runs
    df_results_multiple.sort_values(by = ["mean_harrell"], ascending = False, inplace = True)
    fig = px.scatter(df_results_multiple, x = "model", y = "mean_harrell", color = "model type",
                     color_discrete_map = map_model_type, error_y = "std_harrell")
    fig.update_xaxes(tickangle = -60, tickmode = "linear")
    fig.update_yaxes(range = [0.6, 0.8], title = "Mean Harrell's C-index")
    fig.update_xaxes(categoryorder = "total descending", title = "Model", tickfont = {'size': 19})
    fig.update_layout(legend = {'font' : {'size' : 15}})
    fig.update_layout(title = model)
    # fig.update_layout(xaxis_categoryorder = "total descending")
    fig.write_image(f"plots/{model}/multiple_results_harrell_cindex.png", width = 1200, height = 900)
    # IPCW C-index multiple runs
    df_results_multiple.sort_values(by = ["mean_ipcw"], ascending = False, inplace = True)
    fig = px.scatter(df_results_multiple, x = "model", y = "mean_ipcw", color = "model type",
                     color_discrete_map = map_model_type, error_y = "std_ipcw")
    fig.update_xaxes(tickangle = -60, tickmode = "linear")
    fig.update_yaxes(range = [0.6, 0.8], title = "Mean IPCW C-index")
    fig.update_xaxes(categoryorder = "total descending", title = "Model", tickfont = {'size': 19})
    fig.update_layout(legend = {'font' : {'size' : 15}})
    fig.update_layout(title = model)
    # fig.update_layout(xaxis_categoryorder = "total descending")
    fig.write_image(f"plots/{model}/multiple_results_ipcw_cindex.png", width = 1200, height = 900)
    # Test set
    dict_results_test = {
        "Cox mean heart dose" : df_cox_mean,
        "Cox doses-volumes indicators": df_dosesvol,
        "Cox Lasso doses-volumes indicators": df_dosesvol_lasso,
        "RSF doses-volumes indicators": df_rsf_dosesvol,
        "Cox Lasso whole-heart first-order dosiomics": df_cox_1320_firstorder,
        "Cox Lasso screened whole-heart first-order dosiomics": df_cox_1320_filter_firstorder,
        "Cox Lasso heart's subparts first-order dosiomics": df_cox_32X_firstorder,
        "Cox Lasso heart's subparts screened first-order dosiomics": df_cox_32X_filter_firstorder,
        "Cox Lasso whole-heart dosiomics": df_cox_1320,
        "Cox Lasso screened whole-heart dosiomics": df_cox_1320_filter,
        "Cox Lasso heart's subparts dosiomics": df_cox_32X,
        "Cox Lasso heart's subparts screened dosiomics": df_cox_32X_filter,
        "Cox Bootstrap Lasso whole-heart first-order dosiomics": df_cox_boot_1320_firstorder,
        "Cox Bootstrap Lasso screened whole-heart first-order dosiomics": df_cox_boot_1320_filter_firstorder,
        "Cox Bootstrap Lasso heart's subparts first-order dosiomics": df_cox_boot_32X_firstorder,
        "Cox Bootstrap Lasso heart's subparts screened first-order dosiomics": df_cox_boot_32X_filter_firstorder,
        "Cox Bootstrap Lasso whole-heart dosiomics": df_cox_boot_1320,
        "Cox Bootstrap Lasso screened whole-heart dosiomics": df_cox_boot_1320_filter,
        "Cox Bootstrap Lasso heart's subparts dosiomics": df_cox_boot_32X,
        "Cox Bootstrap Lasso heart's subparts screened dosiomics": df_cox_boot_32X_filter,
        "RSF whole-heart first-order dosiomics": df_rsf_1320_firstorder,
        "RSF screened whole-heart first-order dosiomics": df_rsf_1320_filter_firstorder,
        "RSF heart's subparts first-order dosiomics": df_rsf_32X_firstorder,
        "RSF heart's subparts screened first-order dosiomics": df_rsf_32X_filter_firstorder,
        "RSF whole-heart dosiomics": df_rsf_1320,
        "RSF screened whole-heart dosiomics": df_rsf_1320_filter,
        "RSF heart's subparts dosiomics": df_rsf_32X,
        "RSF heart's subparts screened dosiomics": df_rsf_32X_filter}
    idx_res = dict_results_test.keys()
    df_results_test = pd.DataFrame(data = {"model": idx_res})
    df_results_test.set_index("model", drop = False, inplace = True)
    df_results_test.loc[idx_res, "cindex_harrell"] = [dict_results_test[model].loc["C-index", "Test"] for model in idx_res]
    df_results_test.loc[idx_res, "cindex_ipcw"] = [dict_results_test[model].loc["IPCW C-index", "Test"] for model in idx_res]
    df_results_test.loc[:, "model type"] = df_results_test["model"].apply(get_color)
    # Harrell's C-index
    df_results_test.sort_values(by = ["cindex_harrell"], ascending = False, inplace = True)
    fig = px.scatter(df_results_test, x = "model", y = "cindex_harrell", color = "model type",
                     color_discrete_map = map_model_type)
    fig.update_xaxes(tickangle = -60, tickmode = "linear")
    fig.update_yaxes(range = [0.6, 0.8], title = "Harrell's C-index test set")
    fig.update_xaxes(categoryorder = "total descending", title = "Model", tickfont = {'size': 19})
    fig.update_layout(legend = {'font' : {'size' : 15}})
    fig.update_layout(title = model)
    # fig.update_layout(xaxis_categoryorder = "total descending")
    fig.write_image(f"plots/{model}/results_harrell_cindex.png", width = 1200, height = 900)
    # IPCW C-index
    df_results_test.sort_values(by = ["cindex_ipcw"], ascending = False, inplace = True)
    fig = px.scatter(df_results_test, x = "model", y = "cindex_ipcw", color = "model type",
                     color_discrete_map = map_model_type)
    fig.update_xaxes(tickangle = -60, tickmode = "linear")
    fig.update_yaxes(range = [0.6, 0.8], title = "IPCW C-index test set")
    fig.update_xaxes(categoryorder = "total descending", title = "Model", tickfont = {'size': 19})
    fig.update_layout(legend = {'font' : {'size' : 15}})
    fig.update_layout(title = model)
    # fig.update_layout(xaxis_categoryorder = "total descending")
    fig.write_image(f"plots/{model}/results_ipcw_cindex.png", width = 1200, height = 900)

    with open(f"tables/{model}/results_train.tex", "w") as f:
        f.write(table_train)
    with open(f"tables/{model}/results_test.tex", "w") as f:
        f.write(table_test)
    with open(f"tables/{model}/results_multiple_scores_{nb_estim}.tex", "w") as f:
        f.write(table_multiple)

