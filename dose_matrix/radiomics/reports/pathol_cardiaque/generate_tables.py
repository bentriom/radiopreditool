import pandas as pd

"""
for model in ["pathol_cardiaque", "pathol_cardiaque_chimio", "pathol_cardiaque_drugs"]:
    coxph_results_dir = f"../../slurm_results/analyzes/{model}/coxph_results/"
    rsf_results_dir = f"../../slurm_results/analyzes/{model}/rsf_results/"
    df_rsf_1320 = pd.read_csv(rsf_results_dir + "metrics_model_1320_all.csv")
    df_rsf_1320_filter = pd.read_csv(rsf_results_dir + "metrics_model_1320_features_hclust_corr.csv")
    df_rsf_32X = pd.read_csv(rsf_results_dir + "metrics_model_32X_all.csv")
    df_rsf_32X_filter = pd.read_csv(rsf_results_dir + "metrics_model_32X_features_hclust_corr.csv")
    df_rsf_1320.index = ["C-index", "IPCW C-index", "BS at 60", "IBS"]    
    df_rsf_1320_filter.index = ["C-index", "IPCW C-index", "BS at 60", "IBS"]    
    df_rsf_32X.index = ["C-index", "IPCW C-index", "BS at 60", "IBS"]    
    df_rsf_32X_filter.index = ["C-index", "IPCW C-index", "BS at 60", "IBS"]    
    df_rsf_1320.to_csv(rsf_results_dir + "metrics_model_1320_all.csv", index = True)
    df_rsf_1320_filter.to_csv(rsf_results_dir + "metrics_model_1320_features_hclust_corr.csv", index = True)
    df_rsf_32X.to_csv(rsf_results_dir + "metrics_model_32X_all.csv", index = True)
    df_rsf_32X_filter.to_csv(rsf_results_dir + "metrics_model_32X_features_hclust_corr.csv", index = True)
"""

def generate_line_set(df, set_type):
    return f"${df.loc['C-index',set_type]:.3f}$ & ${df.loc['IPCW C-index',set_type]:.3f}$ & ${df.loc['BS at 60',set_type]:.3f}$ & ${df.loc['IBS',set_type]:.3f}$"

def generate_line_multiple(df):
    return f"${df.loc['C-index','Mean']:.3f} \pm {df.loc['C-index','Std']:.3f}$ & ${df.loc['IPCW C-index','Mean']:.3f} \pm {df.loc['IPCW C-index','Std']:.3f}$ & \
            ${df.loc['BS at 60','Mean']:.3f} \pm {df.loc['BS at 60','Std']:.3f}$ & ${df.loc['IBS','Mean']:.3f} \pm {df.loc['IBS','Std']:.3f}$"

for model in ["pathol_cardiaque_chimio", "pathol_cardiaque_drugs", "pathol_cardiaque_grade3_chimio"]:
    coxph_results_dir = f"../../slurm_results/analyzes/{model}/coxph_R_results/"
    rsf_results_dir = f"../../slurm_results/analyzes/{model}/rsf_results/"

    df_cox_mean = pd.read_csv(coxph_results_dir + "metrics_1320_mean.csv", index_col = 0)
    df_dosesvol = pd.read_csv(coxph_results_dir + "metrics_1320_dosesvol.csv", index_col = 0)
    df_dosesvol_lasso = pd.read_csv(coxph_results_dir + "metrics_1320_dosesvol_lasso.csv", index_col = 0)
    # Firstorder radiomics Cox
    df_cox_1320_firstorder = pd.read_csv(coxph_results_dir + "metrics_1320_radiomics_firstorder_lasso_all.csv", index_col = 0)
    df_cox_1320_filter_firstorder = pd.read_csv(coxph_results_dir + "metrics_1320_radiomics_firstorder_lasso_features_hclust_corr.csv", index_col = 0)
    df_cox_32X_firstorder = pd.read_csv(coxph_results_dir + "metrics_32X_radiomics_firstorder_lasso_all.csv", index_col = 0)
    df_cox_32X_filter_firstorder = pd.read_csv(coxph_results_dir + "metrics_32X_radiomics_firstorder_lasso_features_hclust_corr.csv", index_col = 0)
    # Full radiomics Cox
    df_cox_1320 = pd.read_csv(coxph_results_dir + "metrics_1320_radiomics_full_lasso_all.csv", index_col = 0)
    df_cox_1320_filter = pd.read_csv(coxph_results_dir + "metrics_1320_radiomics_full_lasso_features_hclust_corr.csv", index_col = 0)
    df_cox_32X = pd.read_csv(coxph_results_dir + "metrics_32X_radiomics_full_lasso_all.csv", index_col = 0)
    df_cox_32X_filter = pd.read_csv(coxph_results_dir + "metrics_32X_radiomics_full_lasso_features_hclust_corr.csv", index_col = 0)
    # Firstorder radiomics RSF
    df_rsf_1320_firstorder = pd.read_csv(rsf_results_dir + "metrics_1320_radiomics_firstorder_all.csv", index_col = 0)
    df_rsf_1320_filter_firstorder = pd.read_csv(rsf_results_dir + "metrics_1320_radiomics_firstorder_features_hclust_corr.csv", index_col = 0)
    df_rsf_32X_firstorder = pd.read_csv(rsf_results_dir + "metrics_32X_radiomics_firstorder_all.csv", index_col = 0)
    df_rsf_32X_filter_firstorder = pd.read_csv(rsf_results_dir + "metrics_32X_radiomics_firstorder_features_hclust_corr.csv", index_col = 0)
    # Full radiomics RSF
    df_rsf_1320 = pd.read_csv(rsf_results_dir + "metrics_1320_radiomics_full_all.csv", index_col = 0)
    df_rsf_1320_filter = pd.read_csv(rsf_results_dir + "metrics_1320_radiomics_full_features_hclust_corr.csv", index_col = 0)
    df_rsf_32X = pd.read_csv(rsf_results_dir + "metrics_32X_radiomics_full_all.csv", index_col = 0)
    df_rsf_32X_filter = pd.read_csv(rsf_results_dir + "metrics_32X_radiomics_full_features_hclust_corr.csv", index_col = 0)
   
    table_train = "\\begin{tabular}{|c|c|c|c|c|}\n"
    table_train += "\\hline\n"
    table_train += "Model  & C-index & IPCW C-index & BS at 60 & IBS \\\\ \\hline\n"
    table_train += f"Cox mean heart dose & {generate_line_set(df_cox_mean, 'Train')} \\\\ \\hline\n"
    table_train += f"Cox doses volumes & {generate_line_set(df_dosesvol, 'Train')} \\\\ \\hline\n"
    table_train += f"Cox Lasso doses volumes & {generate_line_set(df_dosesvol_lasso, 'Train')}  \\\\ \\hline\n"
    # Firstorder radiomics Cox
    table_train += f"Cox Lasso firstorder 1320 (AllTrain) & {generate_line_set(df_cox_1320_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso firstorder 1320 (FETrain) & {generate_line_set(df_cox_1320_firstorder_filter, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso firstorder 32X (AllTrain) & {generate_line_set(df_cox_32X_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso firstorder 32X (FETrain) & {generate_line_set(df_cox_32X_filter_firstorder, 'Train')}  \\\\ \\hline\n"
    # Full radiomics Cox
    table_train += f"Cox Lasso 1320 (AllTrain) & {generate_line_set(df_cox_1320, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso 1320 (FETrain) & {generate_line_set(df_cox_1320_filter, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso 32X (AllTrain) & {generate_line_set(df_cox_32X, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso 32X (FETrain) & {generate_line_set(df_cox_32X_filter, 'Train')}  \\\\ \\hline\n"
    # Firstorder radiomics Cox
    table_train += f"RSF 1320 firstorder (AllTrain) & {generate_line_set(df_rsf_1320_firstorder, 'Train')} \\\\ \\hline\n"
    table_train += f"RSF 1320 firstorder (FETrain) &{ generate_line_set(df_rsf_1320_filter_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"RSF 32X firstorder (AllTrain) & {generate_line_set(df_rsf_32X_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"RSF 32X firstorder (FETrain) & {generate_line_set(df_rsf_32X_filter_firstorder, 'Train')}  \\\\ \\hline\n"
    # Full radiomics Cox
    table_train += f"RSF 1320 (AllTrain) & {generate_line_set(df_rsf_1320, 'Train')} \\\\ \\hline\n"
    table_train += f"RSF 1320 (FETrain) &{ generate_line_set(df_rsf_1320_filter, 'Train')}  \\\\ \\hline\n"
    table_train += f"RSF 32X (AllTrain) & {generate_line_set(df_rsf_32X, 'Train')}  \\\\ \\hline\n"
    table_train += f"RSF 32X (FETrain) & {generate_line_set(df_rsf_32X_filter, 'Train')}  \\\\ \\hline\n"
    table_train += "\\end{tabular}" 
    
    table_test = "\\begin{tabular}{|c|c|c|c|c|}\n"
    table_test += "\\hline\n"
    table_test += "Model  & C-index & IPCW C-index & BS at 60 & IBS \\\\ \\hline\n"
    table_test += f"Cox mean heart dose & {generate_line_set(df_cox_mean, 'Test')} \\\\ \\hline\n"
    table_test += f"Cox doses volumes & {generate_line_set(df_dosesvol, 'Test')} \\\\ \\hline\n"
    table_test += f"Cox Lasso doses volumes & {generate_line_set(df_dosesvol_lasso, 'Test')}  \\\\ \\hline\n"
    # Firstorder radiomics Cox
    table_test += f"Cox Lasso firstorder 1320 (AllTrain) & {generate_line_set(df_cox_1320_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso firstorder 1320 (FETrain) & {generate_line_set(df_cox_1320_firstorder_filter, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso firstorder 32X (AllTrain) & {generate_line_set(df_cox_32X_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso firstorder 32X (FETrain) & {generate_line_set(df_cox_32X_filter_firstorder, 'Test')}  \\\\ \\hline\n"
    # Full radiomics Cox
    table_test += f"Cox Lasso 1320 (AllTrain) & {generate_line_set(df_cox_1320, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso 1320 (FETrain) & {generate_line_set(df_cox_1320_filter, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso 32X (AllTrain) & {generate_line_set(df_cox_32X, 'Test')}  \\\\ \\hline\n"
    table_test += f"Cox Lasso 32X (FETrain) & {generate_line_set(df_cox_32X_filter, 'Test')}  \\\\ \\hline\n"
    # Firstorder radiomics RSF
    table_test += f"RSF 1320 firstorder (AllTrain) & {generate_line_set(df_rsf_1320_firstorder, 'Test')} \\\\ \\hline\n"
    table_test += f"RSF 1320 firstorder (FETrain) &{ generate_line_set(df_rsf_1320_filter_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"RSF 32X firstorder (AllTrain) & {generate_line_set(df_rsf_32X_firstorder, 'Test')}  \\\\ \\hline\n"
    table_test += f"RSF 32X firstorder (FETrain) & {generate_line_set(df_rsf_32X_filter_firstorder, 'Test')}  \\\\ \\hline\n"
    # Full radiomics RSF
    table_test += f"RSF 1320 (AllTrain) & {generate_line_set(df_rsf_1320, 'Test')} \\\\ \\hline\n"
    table_test += f"RSF 1320 (FETrain) &{ generate_line_set(df_rsf_1320_filter, 'Test')}  \\\\ \\hline\n"
    table_test += f"RSF 32X (AllTrain) & {generate_line_set(df_rsf_32X, 'Test')}  \\\\ \\hline\n"
    table_test += f"RSF 32X (FETrain) & {generate_line_set(df_rsf_32X_filter, 'Test')}  \\\\ \\hline\n"
    table_test += "\\end{tabular}" 
 
    nb_estim = 10
    df_multiple_cox_mean = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_1320_mean.csv", index_col = 0)
    df_multiple_dosesvol = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_1320_dosesvol.csv", index_col = 0)
    df_multiple_dosesvol_lasso = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_1320_dosesvol_lasso.csv", index_col = 0)
    # Firstorder radiomics Cox
    df_multiple_cox_1320_firstorder = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_1320_radiomics_firstorder_lasso_all.csv", index_col = 0)
    df_multiple_cox_1320_filter_firstorder = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_1320_radiomics_firstorder_lasso_features_hclust_corr.csv", index_col = 0)
    df_multiple_cox_32X_firstorder = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_32X_radiomics_firstorder_lasso_all.csv", index_col = 0)
    df_multiple_cox_32X_filter_firstorder = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_32X_radiomics_firstorder_lasso_features_hclust_corr.csv", index_col = 0)
    # Full radiomics Cox
    df_multiple_cox_1320 = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_1320_radiomics_full_lasso_all.csv", index_col = 0)
    df_multiple_cox_1320_filter = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_1320_radiomics_full_lasso_features_hclust_corr.csv", index_col = 0)
    df_multiple_cox_32X = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_32X_radiomics_full_lasso_all.csv", index_col = 0)
    df_multiple_cox_32X_filter = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_32X_radiomics_full_lasso_features_hclust_corr.csv", index_col = 0)
    # Firstorder radiomics RSF
    df_multiple_rsf_1320_firstorder = pd.read_csv(rsf_results_dir + str(nb_estim) + "_runs_test_metrics_1320_radiomics_firstorder_all.csv", index_col = 0)
    df_multiple_rsf_1320_filter_firstorder = pd.read_csv(rsf_results_dir + str(nb_estim) + "_runs_test_metrics_1320_radiomics_firstorder_features_hclust_corr.csv", index_col = 0)
    df_multiple_rsf_32X_firstorder = pd.read_csv(rsf_results_dir + str(nb_estim) + "_runs_test_metrics_32X_radiomics_firstorder_all.csv", index_col = 0)
    df_multiple_rsf_32X_filter_firstorder = pd.read_csv(rsf_results_dir + str(nb_estim) + "_runs_test_metrics_32X_radiomics_firstorder_features_hclust_corr.csv", index_col = 0)
    # Full radiomics RSF
    df_multiple_rsf_1320 = pd.read_csv(rsf_results_dir + str(nb_estim) + "_runs_test_metrics_1320_radiomics_full_all.csv", index_col = 0)
    df_multiple_rsf_1320_filter = pd.read_csv(rsf_results_dir + str(nb_estim) + "_runs_test_metrics_1320_radiomics_full_features_hclust_corr.csv", index_col = 0)
    df_multiple_rsf_32X = pd.read_csv(rsf_results_dir + str(nb_estim) + "_runs_test_metrics_32X_radiomics_full_all.csv", index_col = 0)
    df_multiple_rsf_32X_filter = pd.read_csv(rsf_results_dir + str(nb_estim) + "_runs_test_metrics_32X_radiomics_full_features_hclust_corr.csv", index_col = 0)
 
    table_multiple = "\\begin{tabular}{|c|c|c|c|c|}\n"
    table_multiple += "\\hline\n"
    table_multiple += "Model  & C-index & IPCW C-index & BS at 60 & IBS \\\\ \\hline\n"
    table_multiple += f"Cox mean heart dose & {generate_line_multiple(df_multiple_cox_mean)} \\\\ \\hline\n"
    table_multiple += f"Cox doses volumes & {generate_line_multiple(df_multiple_dosesvol)} \\\\ \\hline\n"
    table_multiple += f"Cox Lasso doses volumes & {generate_line_multiple(df_multiple_dosesvol_lasso)}  \\\\ \\hline\n"
    # Firstorder radiomics Cox
    table_multiple += f"Cox Lasso 1320 firstorder (AllTrain) & {generate_line_multiple(df_multiple_cox_1320_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso 1320 firstorder (FETrain) & {generate_line_multiple(df_multiple_cox_1320_filter_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso 32X firstorder (AllTrain) & {generate_line_multiple(df_multiple_cox_32X_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso 32X firstorder (FETrain) & {generate_line_multiple(df_multiple_cox_32X_filter_firstorder)}  \\\\ \\hline\n"
    # Full radiomics Cox
    table_multiple += f"Cox Lasso 1320 (AllTrain) & {generate_line_multiple(df_multiple_cox_1320)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso 1320 (FETrain) & {generate_line_multiple(df_multiple_cox_1320_filter)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso 32X (AllTrain) & {generate_line_multiple(df_multiple_cox_32X)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso 32X (FETrain) & {generate_line_multiple(df_multiple_cox_32X_filter)}  \\\\ \\hline\n"
    # Firstorder radiomics RSF
    table_multiple += f"RSF 1320 firstorder (AllTrain) & {generate_line_multiple(df_multiple_rsf_1320_firstorder)} \\\\ \\hline\n"
    table_multiple += f"RSF 1320 firstorder (FETrain) &{ generate_line_multiple(df_multiple_rsf_1320_filter_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"RSF 32X firstorder (AllTrain) & {generate_line_multiple(df_multiple_rsf_32X_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"RSF 32X firstorder (FETrain) & {generate_line_multiple(df_multiple_rsf_32X_filter_firstorder)}  \\\\ \\hline\n"
    # Full radiomics RSF
    table_multiple += f"RSF 1320 (AllTrain) & {generate_line_multiple(df_multiple_rsf_1320)} \\\\ \\hline\n"
    table_multiple += f"RSF 1320 (FETrain) &{ generate_line_multiple(df_multiple_rsf_1320_filter)}  \\\\ \\hline\n"
    table_multiple += f"RSF 32X (AllTrain) & {generate_line_multiple(df_multiple_rsf_32X)}  \\\\ \\hline\n"
    table_multiple += f"RSF 32X (FETrain) & {generate_line_multiple(df_multiple_rsf_32X_filter)}  \\\\ \\hline\n"
    table_multiple += "\\end{tabular}" 

    with open(f"tables/results_train_{model}.tex", "w") as f:
        f.write(table_train)
    with open(f"tables/results_test_{model}.tex", "w") as f:
        f.write(table_test)
    with open(f"tables/results_multiple_scores_{nb_estim}_{model}.tex", "w") as f:
        f.write(table_multiple)

