import pandas as pd
import plotly.express as px
import os

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
#     return f"${df.loc['C-index','Mean']:.3f} \pm {df.loc['C-index','Std']:.3f}$ & \
#             ${df.loc['IPCW C-index','Mean']:.3f} \pm {df.loc['IPCW C-index','Std']:.3f}$ & \
#             ${df.loc['BS at 60','Mean']:.3f} \pm {df.loc['BS at 60','Std']:.3f}$ & \
#             ${df.loc['IBS','Mean']:.3f} \pm {df.loc['IBS','Std']:.3f}$"
    return f"${df.loc['C-index','Mean']:.3f} \pm {df.loc['C-index','Std']:.3f}$ & \
             ${df.loc['IBS','Mean']:.3f} \pm {df.loc['IBS','Std']:.3f}$"

#for model in ["pathol_cardiaque_grade3_drugs_iccc_other"]:
local_results_dir = "/media/moud/LaCie/local_results/analyzes/"
slurm_results_dir = "../../slurm_results/analyzes/"
list_binwidth = [0.1, 0.2, 0.5, 1.0]
list_models = [(local_results_dir, f"pathol_cardiaque_grade3_drugs_iccc_other_bw_{binw}") for binw in list_binwidth]
list_models += [(local_results_dir, f"pathol_cardiaque_grade3_drugs_iccc_other_bw_{binw}_filter_entropy") for binw in list_binwidth]
for results_dir, model in list_models:
    coxph_results_dir = f"{results_dir}{model}/coxph_R_results/"
    rsf_results_dir = f"{results_dir}{model}/rsf_results/"
    os.makedirs(f"tables/{model}", exist_ok = True)

    df_cox_mean = pd.read_csv(coxph_results_dir + "metrics_1320_mean.csv", index_col = 0)
    df_dosesvol = pd.read_csv(coxph_results_dir + "metrics_1320_dosesvol.csv", index_col = 0)
    df_dosesvol_lasso = pd.read_csv(coxph_results_dir + "metrics_1320_dosesvol_lasso.csv", index_col = 0)
    df_rsf_dosesvol = pd.read_csv(rsf_results_dir + "metrics_1320_dosesvol.csv", index_col = 0)
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
    table_train += f"RSF doses volumes & {generate_line_set(df_rsf_dosesvol, 'Train')}  \\\\ \\hline\n"
    # Firstorder radiomics Cox
    table_train += f"Cox Lasso firstorder whole heart (AllTrain) & {generate_line_set(df_cox_1320_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso firstorder whole heart (FETrain) & {generate_line_set(df_cox_1320_filter_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso firstorder subparts of heart (AllTrain) & {generate_line_set(df_cox_32X_firstorder, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso firstorder subparts of heart (FETrain) & {generate_line_set(df_cox_32X_filter_firstorder, 'Train')}  \\\\ \\hline\n"
    # Full radiomics Cox
    table_train += f"Cox Lasso whole heart (AllTrain) & {generate_line_set(df_cox_1320, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso whole heart (FETrain) & {generate_line_set(df_cox_1320_filter, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso subparts of heart (AllTrain) & {generate_line_set(df_cox_32X, 'Train')}  \\\\ \\hline\n"
    table_train += f"Cox Lasso subparts of heart (FETrain) & {generate_line_set(df_cox_32X_filter, 'Train')}  \\\\ \\hline\n"
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
 
    nb_estim = 10
    df_multiple_cox_mean = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_1320_mean.csv", index_col = 0)
    df_multiple_dosesvol = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_1320_dosesvol.csv", index_col = 0)
    df_multiple_dosesvol_lasso = pd.read_csv(coxph_results_dir + str(nb_estim) + "_runs_test_metrics_1320_dosesvol_lasso.csv", index_col = 0)
    df_multiple_rsf_dosesvol = pd.read_csv(rsf_results_dir + str(nb_estim) + "_runs_test_metrics_1320_dosesvol.csv", index_col = 0)
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
    
    # Plot of C-index
    dict_results = {
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
        "RSF whole-heart first-order dosiomics": df_multiple_rsf_1320_firstorder,
        "RSF screened whole-heart first-order dosiomics": df_multiple_rsf_1320_filter_firstorder,
        "RSF heart's subparts first-order dosiomics": df_multiple_rsf_32X_firstorder,
        "RSF heart's subparts screened first-order dosiomics": df_multiple_rsf_32X_filter_firstorder,
        "RSF whole-heart dosiomics": df_multiple_rsf_1320,
        "RSF screened whole-heart dosiomics": df_multiple_rsf_1320_filter,
        "RSF heart's subparts dosiomics": df_multiple_rsf_32X,
        "RSF heart's subparts screened dosiomics": df_multiple_rsf_32X_filter}
    df_results = pd.DataFrame(data = {"model": dict_results.keys()})
    df_results.set_index("model", drop = False, inplace = True)
    df_results.loc[dict_results.keys(), "mean"] = [dict_results[model].loc["C-index", "Mean"] for model in dict_results.keys()]
    df_results.loc[dict_results.keys(), "std"] = [dict_results[model].loc["C-index", "Std"] for model in dict_results.keys()]
    def get_color(model):
        splits = model.split()
        if splits[0] == "RSF":
            return "Random Survival Forest"
        if splits[0] == "Cox":
            if splits[1] == "Lasso":
                return "Cox PH with Lasso penalty"
            else:
                return "Cox PH"
    df_results.loc[:, "model type"] = df_results["model"].apply(get_color)
    df_results.sort_values(by = ["mean"], ascending = False, inplace = True)
    map_model_type = {'Random Survival Forest': 'red', 'Cox PH with Lasso penalty': 'blue', 'Cox PH': 'green'}
    fig = px.scatter(df_results, x = "model", y = "mean", color = "model type", color_discrete_map = map_model_type, error_y = "std")
    fig.update_xaxes(tickangle = -60)
    fig.update_yaxes(range = [0.6, 0.8], title = "Mean C-index")
    fig.update_xaxes(categoryorder = "total descending", title = "Model", tickfont = {'size': 19})
    fig.update_layout(legend = {'font' : {'size' : 15}})
    fig.update_layout(title = model)
    # fig.update_layout(xaxis_categoryorder = "total descending")
    fig.write_image(f"tables/{model}/results_cindex.png", width = 1200, height = 900)

    #table_multiple = "\\begin{tabular}{|c|c|c|c|c|}\n"
    table_multiple = "\\begin{tabular}{|c|c|c|}\n"
    table_multiple += "\\hline\n"
    #table_multiple += "Model  & C-index & IPCW C-index & BS at 60 & IBS \\\\ \\hline\n"
    table_multiple += "Model  & C-index & IBS \\\\ \\hline\n"
    table_multiple += f"Cox mean heart dose & {generate_line_multiple(df_multiple_cox_mean)} \\\\ \\hline\n"
    table_multiple += f"Cox doses volumes & {generate_line_multiple(df_multiple_dosesvol)} \\\\ \\hline\n"
    table_multiple += f"Cox Lasso doses volumes & {generate_line_multiple(df_multiple_dosesvol_lasso)}  \\\\ \\hline\n"
    table_multiple += f"RSF doses volumes & {generate_line_multiple(df_multiple_rsf_dosesvol)}  \\\\ \\hline\n"
    # Firstorder radiomics Cox
    table_multiple += f"Cox Lasso whole heart firstorder (AllTrain) & {generate_line_multiple(df_multiple_cox_1320_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso whole heart firstorder (FETrain) & {generate_line_multiple(df_multiple_cox_1320_filter_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso subparts of heart firstorder (AllTrain) & {generate_line_multiple(df_multiple_cox_32X_firstorder)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso subparts of heart firstorder (FETrain) & {generate_line_multiple(df_multiple_cox_32X_filter_firstorder)}  \\\\ \\hline\n"
    # Full radiomics Cox
    table_multiple += f"Cox Lasso whole heart (AllTrain) & {generate_line_multiple(df_multiple_cox_1320)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso whole heart (FETrain) & {generate_line_multiple(df_multiple_cox_1320_filter)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso subparts of heart (AllTrain) & {generate_line_multiple(df_multiple_cox_32X)}  \\\\ \\hline\n"
    table_multiple += f"Cox Lasso subparts of heart (FETrain) & {generate_line_multiple(df_multiple_cox_32X_filter)}  \\\\ \\hline\n"
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

    with open(f"tables/{model}/results_train.tex", "w") as f:
        f.write(table_train)
    with open(f"tables/{model}/results_test.tex", "w") as f:
        f.write(table_test)
    with open(f"tables/{model}/results_multiple_scores_{nb_estim}.tex", "w") as f:
        f.write(table_multiple)

