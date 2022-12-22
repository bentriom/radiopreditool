
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

## C-index scores plots

def get_color(model):
    splits = model.split()
    if splits[0] == "RSF":
        return "Random Survival Forest"
    elif splits[0] == "C":
        return "Cox PH"
    elif splits[0] == "CL":
            return "Cox PH with Lasso penalty"
    elif splits[0] == "CBL":
        return "Cox PH with Bootstrap Lasso"

def cindex_plots(analyzes_dir, nb_estim):
    coxph_results_dir = f"{analyzes_dir}coxph_R/"
    rsf_results_dir = f"{analyzes_dir}rsf/"
    save_plots_dir = f"{analyzes_dir}plots/"
    os.makedirs(save_plots_dir, exist_ok = True)

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

    dict_results_multiple_all = {
        "C mean heart dose" : df_multiple_cox_mean,
        "C doses-volumes indicators": df_multiple_dosesvol,
        "CL doses-volumes indicators": df_multiple_dosesvol_lasso,
        "RSF doses-volumes indicators": df_multiple_rsf_dosesvol,
        "CL whole-heart first-order dosiomics": df_multiple_cox_1320_firstorder,
        "CL whole-heart dosiomics": df_multiple_cox_1320,
        "CL heart's subparts first-order dosiomics": df_multiple_cox_32X_firstorder,
        "CL heart's subparts dosiomics": df_multiple_cox_32X,
        "CBL whole-heart first-order dosiomics": df_multiple_cox_boot_1320_firstorder,
        "CBL whole-heart dosiomics": df_multiple_cox_boot_1320,
        "CBL heart's subparts first-order dosiomics": df_multiple_cox_boot_32X_firstorder,
        "CBL heart's subparts dosiomics": df_multiple_cox_boot_32X,
        "RSF whole-heart first-order dosiomics": df_multiple_rsf_1320_firstorder,
        "RSF whole-heart dosiomics": df_multiple_rsf_1320,
        "RSF heart's subparts first-order dosiomics": df_multiple_rsf_32X_firstorder,
        "RSF heart's subparts dosiomics": df_multiple_rsf_32X
    }
    dict_results_multiple_features_hclust_corr = {
        "CL screened whole-heart first-order dosiomics": df_multiple_cox_1320_filter_firstorder,
        "CL screened whole-heart dosiomics": df_multiple_cox_1320_filter,
        "CL screened heart's subparts first-order dosiomics": df_multiple_cox_32X_filter_firstorder,
        "CL screened heart's subparts dosiomics": df_multiple_cox_32X_filter,
        "CBL screened whole-heart first-order dosiomics": df_multiple_cox_boot_1320_filter_firstorder,
        "CBL screened whole-heart dosiomics": df_multiple_cox_boot_1320_filter,
        "CBL screened heart's subparts first-order dosiomics": df_multiple_cox_boot_32X_filter_firstorder,
        "CBL screened heart's subparts dosiomics": df_multiple_cox_boot_32X_filter,
        "RSF screened whole-heart first-order dosiomics": df_multiple_rsf_1320_filter_firstorder,
        "RSF screened whole-heart dosiomics": df_multiple_rsf_1320_filter,
        "RSF screened heart's subparts first-order dosiomics": df_multiple_rsf_32X_filter_firstorder,
        "RSF screened heart's subparts dosiomics": df_multiple_rsf_32X_filter
    }
    dict_results_multiple = dict_results_multiple_all | dict_results_multiple_features_hclust_corr
    idx_res = dict_results_multiple.keys()
    idx_res_all = dict_results_multiple_all.keys()
    idx_res_features_hclust_corr = dict_results_multiple_features_hclust_corr.keys()
    df_results_multiple = pd.DataFrame(data = {"model": idx_res})
    df_results_multiple.set_index("model", drop = False, inplace = True)
    df_results_multiple.loc[idx_res, "mean_harrell"] = [dict_results_multiple[model].loc["C-index", "Mean"] for model in idx_res]
    df_results_multiple.loc[idx_res, "std_harrell"] = [dict_results_multiple[model].loc["C-index", "Std"] for model in idx_res]
    df_results_multiple.loc[idx_res, "mean_ipcw"] = [dict_results_multiple[model].loc["IPCW C-index", "Mean"] for model in idx_res]
    df_results_multiple.loc[idx_res, "std_ipcw"] = [dict_results_multiple[model].loc["IPCW C-index", "Std"] for model in idx_res]
    df_results_multiple.loc[:, "model type"] = df_results_multiple["model"].apply(get_color)

    ## Harrell's C-index multiple runs
    df_results_multiple.sort_values(by = ["mean_harrell"], ascending = False, inplace = True)
    fig = make_subplots(rows = 2, cols = 2, horizontal_spacing = 0.15, vertical_spacing = 0.35,
                        subplot_titles = ("All features", "Pre-screening", "", ""))
    color_map = {'Random Survival Forest': 'red', 'Cox PH with Lasso penalty': 'blue',
                 'Cox PH with Bootstrap Lasso': 'purple', 'Cox PH': 'green'}
    symbol_map = {'Random Survival Forest': 'diamond-open', 'Cox PH with Lasso penalty': 'square-open',
                  'Cox PH with Bootstrap Lasso': 'x-open', 'Cox PH': 'circle-open'}
    xaxis_angle = -50
    xaxis_size = 16
    format_xaxis = lambda model: model.split(' ', 1)[1].replace('screened ', '').capitalize()

    # No screening
    fig_scatter = px.scatter(df_results_multiple.loc[idx_res_all, :], x = "model", y = "mean_harrell",
                             color = "model type", error_y = "std_harrell")
    fig_scatter.update_xaxes(tickangle = xaxis_angle, tickmode = "linear")
    fig_scatter.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_all, "model"],
                             ticktext = df_results_multiple.loc[idx_res_all, "model"].apply(format_xaxis))
    fig_scatter.update_yaxes(range = [0.5, 1.0], title = "Mean Harrell's C-index")
    fig_scatter.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size})
    fig_scatter.write_image(f"{save_plots_dir}multiple_scores_harrell_cindex_all.svg", width = 1200, height = 900)
    for i in range(len(fig_scatter.data)):
        name_scatter = fig_scatter.data[i]["name"]
        trace_scatter = go.Scatter(fig_scatter.data[i],
                                   marker_color = color_map[name_scatter],
                                   marker_size = 9,
                                   marker_line = dict(width = 2.5),
                                   marker_symbol = symbol_map[name_scatter])
        fig.add_trace(trace_scatter, row = 1, col = 1)
    fig.update_xaxes(tickangle = xaxis_angle, tickmode = "linear", row = 1, col = 1)
    fig.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_all, "model"],
                             ticktext = df_results_multiple.loc[idx_res_all, "model"].apply(format_xaxis), row = 1, col = 1)
    fig.update_yaxes(range = [0.5, 1.0], title = "Mean Harrell's C-index", row = 1, col = 1)
    fig.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size}, row = 1, col = 1)
    # Features hclust correlation screening
    fig_scatter = px.scatter(df_results_multiple.loc[idx_res_features_hclust_corr, :], x = "model", y = "mean_harrell",
                             color = "model type", error_y = "std_harrell")
    fig_scatter.update_xaxes(tickangle = xaxis_angle, tickmode = "linear")
    fig_scatter.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_features_hclust_corr, "model"],
                             ticktext = df_results_multiple.loc[idx_res_features_hclust_corr, "model"].apply(format_xaxis))
    fig_scatter.update_yaxes(range = [0.5, 1.0], title = "Mean Harrell's C-index", row = 1, col = 2)
    fig_scatter.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size})
    fig_scatter.write_image(f"{save_plots_dir}multiple_scores_harrell_cindex_features_hclust_corr.svg", width = 1200, height = 900)
    fig_scatter.update_traces(showlegend = False)
    for i in range(len(fig_scatter.data)):
        name_scatter = fig_scatter.data[i]["name"]
        trace_scatter = go.Scatter(fig_scatter.data[i],
                                   marker_color = color_map[name_scatter],
                                   marker_size = 9,
                                   marker_line = dict(width = 2.5),
                                   marker_symbol = symbol_map[name_scatter])
        fig.add_trace(trace_scatter, row = 1, col = 2)
    fig.update_xaxes(tickangle = xaxis_angle, tickmode = "linear", row = 1, col = 2)
    fig.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_features_hclust_corr, "model"],
                     ticktext = df_results_multiple.loc[idx_res_features_hclust_corr, "model"].apply(format_xaxis),
                     row = 1, col = 2)
    fig.update_yaxes(range = [0.5, 1.0], title = "Mean Harrell's C-index", row = 1, col = 2)
    fig.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size}, row = 1, col = 2)

    ## IPCW C-index multiple runs
    df_results_multiple.sort_values(by = ["mean_ipcw"], ascending = False, inplace = True)

    # No screening
    fig_scatter = px.scatter(df_results_multiple.loc[idx_res_all, :], x = "model", y = "mean_ipcw",
                             color = "model type", error_y = "std_ipcw")
    fig_scatter.update_xaxes(tickangle = xaxis_angle, tickmode = "linear")
    fig_scatter.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_all, "model"],
                     ticktext = df_results_multiple.loc[idx_res_all, "model"].apply(format_xaxis))
    fig_scatter.update_yaxes(range = [0.5, 1.0], title = "Mean IPCW C-index")
    fig_scatter.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size})
    fig_scatter.write_image(f"{save_plots_dir}multiple_scores_ipcw_cindex_all.svg", width = 1200, height = 900)
    fig_scatter.update_traces(showlegend = False)
    for i in range(len(fig_scatter.data)):
        name_scatter = fig_scatter.data[i]["name"]
        trace_scatter = go.Scatter(fig_scatter.data[i],
                                   marker_color = color_map[name_scatter],
                                   marker_size = 9,
                                   marker_line = dict(width = 2.5),
                                   marker_symbol = symbol_map[name_scatter])
        fig.add_trace(trace_scatter, row = 2, col = 1)
    fig.update_xaxes(tickangle = xaxis_angle, tickmode = "linear", row = 2, col = 1)
    fig.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_all, "model"],
                     ticktext = df_results_multiple.loc[idx_res_all, "model"].apply(format_xaxis), row = 2, col = 1)
    fig.update_yaxes(range = [0.5, 1.0], title = "Mean IPCW C-index", row = 2, col = 1)
    fig.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size}, row = 2, col = 1)

    # Features hclust correlation screening
    fig_scatter = px.scatter(df_results_multiple.loc[idx_res_features_hclust_corr, :], x = "model", y = "mean_ipcw",
                             color = "model type", error_y = "std_ipcw")
    fig_scatter.update_xaxes(tickangle = xaxis_angle, tickmode = "linear")
    fig_scatter.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_features_hclust_corr, "model"],
                             ticktext = df_results_multiple.loc[idx_res_features_hclust_corr, "model"].apply(format_xaxis))
    fig_scatter.update_yaxes(range = [0.5, 1.0], title = "Mean IPCW C-index")
    fig_scatter.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size})
    fig_scatter.write_image(f"{save_plots_dir}multiple_scores_ipcw_cindex_features_hclust_corr.svg", width = 1200, height = 900)
    fig_scatter.update_traces(showlegend = False)
    for i in range(len(fig_scatter.data)):
        name_scatter = fig_scatter.data[i]["name"]
        trace_scatter = go.Scatter(fig_scatter.data[i],
                                   marker_color = color_map[name_scatter],
                                   marker_size = 9,
                                   marker_line = dict(width = 2.5),
                                   marker_symbol = symbol_map[name_scatter])
        fig.add_trace(trace_scatter, row = 2, col = 2)
    fig.update_xaxes(tickangle = xaxis_angle, tickmode = "linear", row = 2, col = 2)
    fig.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_features_hclust_corr, "model"],
                     ticktext = df_results_multiple.loc[idx_res_features_hclust_corr, "model"].apply(format_xaxis),
                     row = 2, col = 2)
    fig.update_yaxes(range = [0.5, 1.0], title = "Mean IPCW C-index", row = 2, col = 2)
    fig.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size}, row = 2, col = 2)

    fig.update_layout(legend = {'font' : {'size' : 15}})
    fig.update_layout(title = "C-index score", title_x = 0.5)

    fig.write_image(f"{save_plots_dir}multiple_scores_cindex.svg", width = 1200, height = 1200)

## Latex tables

def generate_line_set(df, set_type):
    return f"${df.loc['C-index',set_type]:.3f}$ & ${df.loc['IPCW C-index',set_type]:.3f}$"\
           f"& ${df.loc['BS at 60',set_type]:.3f}$ & ${df.loc['IBS',set_type]:.3f}$"

def generate_line_multiple(df):
    return f"${df.loc['C-index','Mean']:.3f} \pm {df.loc['C-index','Std']:.3f}$ & \
             ${df.loc['IBS','Mean']:.3f} \pm {df.loc['IBS','Std']:.3f}$"

def latex_tables(analyzes_dir, nb_estim):
    coxph_results_dir = f"{analyzes_dir}coxph_R/"
    rsf_results_dir = f"{analyzes_dir}rsf/"

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

    # Save table
    os.makedirs(f"{analyzes_dir}tables", exist_ok = True)
    with open(f"{analyzes_dir}tables/multiple_scores_{nb_estim}_runs.tex", "w") as f:
        f.write(table_multiple)

