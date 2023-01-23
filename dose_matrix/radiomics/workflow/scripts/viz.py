
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
            return "Cox PH Lasso"
    elif splits[0] == "CBL":
        return "Cox PH Bootstrap Lasso"

def results_plots(analyzes_dir, nb_estim):
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
    df_results_multiple.loc[idx_res, "mean_ibs"] = [dict_results_multiple[model].loc["IBS", "Mean"] for model in idx_res]
    df_results_multiple.loc[idx_res, "std_ibs"] = [dict_results_multiple[model].loc["IBS", "Std"] for model in idx_res]
    df_results_multiple.loc[:, "model type"] = df_results_multiple["model"].apply(get_color)

    ## Plots settings
    color_map = {'Random Survival Forest': 'red', 'Cox PH Lasso': 'blue',
                 'Cox PH Bootstrap Lasso': 'purple', 'Cox PH': 'green'}
    symbol_map = {'Random Survival Forest': 'diamond-open', 'Cox PH Lasso': 'square-open',
                  'Cox PH Bootstrap Lasso': 'x-open', 'Cox PH': 'circle-open'}
    xaxis_angle = -50
    xaxis_size = 19.5
    legend_size = 18
    format_xaxis = lambda model: model.split(' ', 1)[1].replace('screened ', '').capitalize()

    ## Harrell's C-index multiple runs
    df_results_multiple.sort_values(by = ["mean_harrell"], ascending = False, inplace = True)
    max_harrell_cindex = df_results_multiple.iloc[0]["mean_harrell"]
    fig = make_subplots(rows = 2, cols = 2, horizontal_spacing = 0.13, vertical_spacing = 0.41,
                        subplot_titles = ("All features", "Pre-screening", "All features", "Pre-screening"))
    fig.update_layout(legend = dict(orientation = "h", font = {'size': legend_size},
                                    xanchor = "left", x = 0, yanchor = "bottom", y = 1.04))

    # No screening
    fig_scatter = px.scatter(df_results_multiple.loc[idx_res_all, :], x = "model", y = "mean_harrell",
                             color = "model type", error_y = "std_harrell")
    fig_scatter.update_xaxes(tickangle = xaxis_angle, tickmode = "linear")
    fig_scatter.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_all, "model"],
                             ticktext = df_results_multiple.loc[idx_res_all, "model"].apply(format_xaxis))
    fig_scatter.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size})
    fig_scatter.update_yaxes(range = [0.5, 1.0], title = "Mean Harrell's C-index")
    fig_scatter.write_image(f"{save_plots_dir}multiple_scores_harrell_cindex_all.svg", width = 1200, height = 900)
    for i in range(len(fig_scatter.data)):
        name_scatter = fig_scatter.data[i]["name"]
        trace_scatter = go.Scatter(fig_scatter.data[i],
                                   marker_color = color_map[name_scatter],
                                   marker_size = 9,
                                   marker_line = dict(width = 2.5),
                                   marker_symbol = symbol_map[name_scatter])
        fig.add_trace(trace_scatter, row = 1, col = 1)
        fig.add_hline(y = max_harrell_cindex, line_width = 1.5, line_dash = "dash",
                      line_color = "grey", opacity = 0.3, row = 1, col = 1)
    fig.update_xaxes(tickangle = xaxis_angle, tickmode = "linear", row = 1, col = 1)
    fig.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_all, "model"],
                             ticktext = df_results_multiple.loc[idx_res_all, "model"].apply(format_xaxis), row = 1, col = 1)
    fig.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size}, row = 1, col = 1)
    fig.update_yaxes(range = [0.5, 1.0], title = "Mean Harrell's C-index", row = 1, col = 1)
    # Features hclust correlation screening
    fig_scatter = px.scatter(df_results_multiple.loc[idx_res_features_hclust_corr, :], x = "model", y = "mean_harrell",
                             color = "model type", error_y = "std_harrell")
    fig_scatter.update_xaxes(tickangle = xaxis_angle, tickmode = "linear")
    fig_scatter.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_features_hclust_corr, "model"],
                             ticktext = df_results_multiple.loc[idx_res_features_hclust_corr, "model"].apply(format_xaxis))
    fig_scatter.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size})
    fig_scatter.update_yaxes(range = [0.5, 1.0], title = "Mean Harrell's C-index", row = 1, col = 2)
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
        fig.add_hline(y = max_harrell_cindex, line_width = 1.5, line_dash = "dash",
                      annotation_text = "max", annotation_position = "top right",
                      line_color = "grey", opacity = 0.3, row = 1, col = 2)
    fig.update_xaxes(tickangle = xaxis_angle, tickmode = "linear", row = 1, col = 2)
    fig.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_features_hclust_corr, "model"],
                     ticktext = df_results_multiple.loc[idx_res_features_hclust_corr, "model"].apply(format_xaxis),
                     row = 1, col = 2)
    fig.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size}, row = 1, col = 2)
    fig.update_yaxes(range = [0.5, 1.0], title = "Mean Harrell's C-index", row = 1, col = 2)

    ## IPCW C-index multiple runs
    df_results_multiple.sort_values(by = ["mean_ipcw"], ascending = False, inplace = True)
    max_ipcw_cindex = df_results_multiple.iloc[0]["mean_ipcw"]

    # No screening
    fig_scatter = px.scatter(df_results_multiple.loc[idx_res_all, :], x = "model", y = "mean_ipcw",
                             color = "model type", error_y = "std_ipcw")
    fig_scatter.update_xaxes(tickangle = xaxis_angle, tickmode = "linear")
    fig_scatter.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_all, "model"],
                     ticktext = df_results_multiple.loc[idx_res_all, "model"].apply(format_xaxis))
    fig_scatter.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size})
    fig_scatter.update_yaxes(range = [0.5, 1.0], title = "Mean IPCW C-index")
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
        fig.add_hline(y = max_ipcw_cindex, line_width = 1.5, line_dash = "dash",
                      line_color = "grey", opacity = 0.3, row = 2, col = 1)
    fig.update_xaxes(tickangle = xaxis_angle, tickmode = "linear", row = 2, col = 1)
    fig.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_all, "model"],
                     ticktext = df_results_multiple.loc[idx_res_all, "model"].apply(format_xaxis), row = 2, col = 1)
    fig.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size}, row = 2, col = 1)
    fig.update_yaxes(range = [0.5, 1.0], title = "Mean IPCW C-index", row = 2, col = 1)

    # Features hclust correlation screening
    fig_scatter = px.scatter(df_results_multiple.loc[idx_res_features_hclust_corr, :], x = "model", y = "mean_ipcw",
                             color = "model type", error_y = "std_ipcw")
    fig_scatter.update_xaxes(tickangle = xaxis_angle, tickmode = "linear")
    fig_scatter.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_features_hclust_corr, "model"],
                             ticktext = df_results_multiple.loc[idx_res_features_hclust_corr, "model"].apply(format_xaxis))
    fig_scatter.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size})
    fig_scatter.update_yaxes(range = [0.5, 1.0], title = "Mean IPCW C-index")
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
        fig.add_hline(y = max_ipcw_cindex, line_width = 1.5, line_dash = "dash",
                      annotation_text = "max", annotation_position = "top right",
                      line_color = "grey", opacity = 0.3, row = 2, col = 2)
    fig.update_xaxes(tickangle = xaxis_angle, tickmode = "linear", row = 2, col = 2)
    fig.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_features_hclust_corr, "model"],
                     ticktext = df_results_multiple.loc[idx_res_features_hclust_corr, "model"].apply(format_xaxis),
                     row = 2, col = 2)
    fig.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size}, row = 2, col = 2)
    fig.update_yaxes(range = [0.5, 1.0], title = "Mean IPCW C-index", row = 2, col = 2)

    # fig.update_layout(title = "C-index score", title_x = 0.5)

    fig.write_image(f"{save_plots_dir}multiple_scores_cindex.svg", width = 1200, height = 1200)
    fig.write_image(f"{save_plots_dir}multiple_scores_cindex.png", width = 1200, height = 1200)

    ## Plot IBS
    df_results_multiple.sort_values(by = ["mean_ibs"], ascending = False, inplace = True)
    max_ibs = df_results_multiple.iloc[0]["mean_ibs"]
    y_max_ibs = 1.1 * (df_results_multiple["mean_ibs"] + df_results_multiple["std_ibs"]).max()
    fig = make_subplots(rows = 1, cols = 2, horizontal_spacing = 0.15, vertical_spacing = 0.35,
                        subplot_titles = ("All features", "Pre-screening"))
    fig.update_layout(legend = dict(orientation = "h", font = {'size': legend_size},
                                    xanchor = "left", x = 0, yanchor = "bottom", y = 1.06))

    # No screening
    fig_scatter = px.scatter(df_results_multiple.loc[idx_res_all, :], x = "model", y = "mean_ibs",
                             color = "model type", error_y = "std_ibs")
    fig_scatter.update_xaxes(tickangle = xaxis_angle, tickmode = "linear")
    fig_scatter.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_all, "model"],
                             ticktext = df_results_multiple.loc[idx_res_all, "model"].apply(format_xaxis))
    fig_scatter.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size})
    fig_scatter.update_yaxes(range = [0, y_max_ibs], title = "Mean IBS")
    fig_scatter.write_image(f"{save_plots_dir}multiple_scores_ibs_all.svg", width = 1200, height = 800)
    for i in range(len(fig_scatter.data)):
        name_scatter = fig_scatter.data[i]["name"]
        trace_scatter = go.Scatter(fig_scatter.data[i],
                                   marker_color = color_map[name_scatter],
                                   marker_size = 9,
                                   marker_line = dict(width = 2.5),
                                   marker_symbol = symbol_map[name_scatter])
        fig.add_trace(trace_scatter, row = 1, col = 1)
        fig.add_hline(y = max_ibs, line_width = 1.5, line_dash = "dash",
                      line_color = "grey", opacity = 0.3, row = 1, col = 1)
    fig.update_xaxes(tickangle = xaxis_angle, tickmode = "linear", row = 1, col = 1)
    fig.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_all, "model"],
                             ticktext = df_results_multiple.loc[idx_res_all, "model"].apply(format_xaxis), row = 1, col = 1)
    fig.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size}, row = 1, col = 1)
    fig.update_yaxes(range = [0, y_max_ibs], title = "Mean IBS", row = 1, col = 1)
    # Features hclust correlation screening
    fig_scatter = px.scatter(df_results_multiple.loc[idx_res_features_hclust_corr, :], x = "model", y = "mean_ibs",
                             color = "model type", error_y = "std_ibs")
    fig_scatter.update_xaxes(tickangle = xaxis_angle, tickmode = "linear")
    fig_scatter.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_features_hclust_corr, "model"],
                             ticktext = df_results_multiple.loc[idx_res_features_hclust_corr, "model"].apply(format_xaxis))
    fig_scatter.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size})
    fig_scatter.update_yaxes(range = [0, y_max_ibs], title = "Mean IBS", row = 1, col = 2)
    fig_scatter.write_image(f"{save_plots_dir}multiple_scores_ibs_features_hclust_corr.svg", width = 1200, height = 800)
    fig_scatter.update_traces(showlegend = False)
    for i in range(len(fig_scatter.data)):
        name_scatter = fig_scatter.data[i]["name"]
        trace_scatter = go.Scatter(fig_scatter.data[i],
                                   marker_color = color_map[name_scatter],
                                   marker_size = 9,
                                   marker_line = dict(width = 2.5),
                                   marker_symbol = symbol_map[name_scatter])
        fig.add_trace(trace_scatter, row = 1, col = 2)
        fig.add_hline(y = max_ibs, line_width = 1.5, line_dash = "dash",
                      annotation_text = "max", annotation_position = "top right",
                      line_color = "grey", opacity = 0.3, row = 1, col = 2)
    fig.update_xaxes(tickangle = xaxis_angle, tickmode = "linear", row = 1, col = 2)
    fig.update_xaxes(tickmode = "array", tickvals = df_results_multiple.loc[idx_res_features_hclust_corr, "model"],
                     ticktext = df_results_multiple.loc[idx_res_features_hclust_corr, "model"].apply(format_xaxis),
                     row = 1, col = 2)
    fig.update_xaxes(categoryorder = "total descending", title = "", tickfont = {'size': xaxis_size}, row = 1, col = 2)
    fig.update_yaxes(range = [0, y_max_ibs], title = "Mean IBS", row = 1, col = 2)

    fig.write_image(f"{save_plots_dir}multiple_scores_ibs.svg", width = 1200, height = 800)
    fig.write_image(f"{save_plots_dir}multiple_scores_ibs.png", width = 1200, height = 800)

## Latex tables

def generate_line_set(df, set_type):
    return f"${df.loc['C-index',set_type]:.3f}$ & ${df.loc['IPCW C-index',set_type]:.3f}$"\
           f"& ${df.loc['BS at 60',set_type]:.3f}$ & ${df.loc['IBS',set_type]:.3f}$"

def generate_line_multiple(df):
    return f"${df.loc['C-index','Mean']:.3f} \pm {df.loc['C-index','Std']:.3f}$ & " \
           f"${df.loc['IPCW C-index','Mean']:.3f} \pm {df.loc['IPCW C-index','Std']:.3f}$ & " \
           f"${df.loc['IBS','Mean']:.3f} \pm {df.loc['IBS','Std']:.3f}$"

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
    table_multiple = "\\begin{tabular}{|l|c|c|c|c|c|c|}\n"
    table_multiple += "\\hline\n"
    table_multiple += "\\multirow{2}{*}{Model} & \\multicolumn{3}{c}{All features} & \\multicolumn{3}{c}{Pre-screening} \\\\ \\cline{2-7}" \
                      "& Harrell's C-index & IPCW C-index & IBS & Harrell's C-index & IPCW C-index & IBS  \\\\ \hline"
    table_multiple += "\\begin{tabular}{@{}l@{}}Cox \\\\ Mean heart dose\\end{tabular}  & " \
                      f"{generate_line_multiple(df_multiple_cox_mean)} & & & \\\\ \\hline\n"
    table_multiple += "\\begin{tabular}{@{}l@{}}Cox \\\\ Dose-volume indicators\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_dosesvol)} & & & \\\\ \\hline\n"
    table_multiple += "\\begin{tabular}{@{}l@{}}Cox Lasso \\\\ Dose-volume indicators\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_dosesvol_lasso)} & & & \\\\ \\hline\n"
    table_multiple += "\\begin{tabular}{@{}l@{}}RSF \\\\ Dose-volume indicators\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_rsf_dosesvol)} & & & \\\\ \\hline\n"
    # Firstorder radiomics Cox Lasso
    table_multiple += "\\begin{tabular}{@{}l@{}}Cox Lasso \\\\ Whole heart firstorder dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_cox_1320_firstorder)} & " \
                      f"{generate_line_multiple(df_multiple_cox_1320_filter_firstorder)} \\\\ \\hline\n"
    table_multiple += "\\begin{tabular}{@{}l@{}}Cox Lasso \\\\ Heart's subparts firstorder dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_cox_32X_firstorder)} & " \
                      f"{generate_line_multiple(df_multiple_cox_32X_filter_firstorder)} \\\\ \\hline\n"
    # Full radiomics Cox Lasso
    table_multiple += "\\begin{tabular}{@{}l@{}}Cox Lasso \\\\ Whole heart dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_cox_1320)} & " \
                      f"{generate_line_multiple(df_multiple_cox_1320_filter)} \\\\ \\hline\n"
    table_multiple += "\\begin{tabular}{@{}l@{}}Cox Lasso \\\\ Heart's subparts dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_cox_32X)} & " \
                      f"{generate_line_multiple(df_multiple_cox_32X_filter)} \\\\ \\hline\n"
    # Firstorder radiomics Cox Bootstrap Lasso
    table_multiple += "\\begin{tabular}{@{}l@{}}Cox Bootstrap Lasso \\\\ Whole heart firstorder dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_cox_boot_1320_firstorder)} & " \
                      f"{generate_line_multiple(df_multiple_cox_boot_1320_filter_firstorder)} \\\\ \\hline\n"
    table_multiple += "\\begin{tabular}{@{}l@{}}Cox Bootstrap Lasso \\\\ Heart's subparts firstorder dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_cox_boot_32X_firstorder)} & " \
                      f"{generate_line_multiple(df_multiple_cox_boot_32X_filter_firstorder)} \\\\ \\hline\n"
    # Full radiomics Cox Bootstrap Lasso
    table_multiple += "\\begin{tabular}{@{}l@{}}Cox Bootstrap Lasso \\\\ Whole heart dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_cox_boot_1320)} & " \
                      f"{generate_line_multiple(df_multiple_cox_boot_1320_filter)} \\\\ \\hline\n"
    table_multiple += "\\begin{tabular}{@{}l@{}}Cox Bootstrap Lasso \\\\ Heart's subparts dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_cox_boot_32X)} & " \
                      f"{generate_line_multiple(df_multiple_cox_boot_32X_filter)} \\\\ \\hline\n"
    # Firstorder radiomics RSF
    table_multiple += "\\begin{tabular}{@{}l@{}}RSF \\\\ Whole heart firstorder dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_rsf_1320_firstorder)} & " \
                      f"{generate_line_multiple(df_multiple_rsf_1320_filter_firstorder)} \\\\ \\hline\n"
    table_multiple += "\\begin{tabular}{@{}l@{}}RSF \\\\ Heart's subparts firstorder dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_rsf_32X_firstorder)} & " \
                      f"{generate_line_multiple(df_multiple_rsf_32X_filter_firstorder)} \\\\ \\hline\n"
    # Full radiomics RSF
    table_multiple += "\\begin{tabular}{@{}l@{}}RSF \\\\ Whole heart dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_rsf_1320)} & " \
                      f"{generate_line_multiple(df_multiple_rsf_1320_filter)} \\\\ \\hline\n"
    table_multiple += "\\begin{tabular}{@{}l@{}}RSF \\\\ Heart's subparts dosiomics\\end{tabular} & " \
                      f"{generate_line_multiple(df_multiple_rsf_32X)} & " \
                      f"{generate_line_multiple(df_multiple_rsf_32X_filter)} \\\\ \\hline\n"
    table_multiple += "\\end{tabular}\n"


    # Save table
    os.makedirs(f"{analyzes_dir}tables", exist_ok = True)
    with open(f"{analyzes_dir}tables/multiple_scores_{nb_estim}_runs.tex", "w") as f:
        f.write(table_multiple)

