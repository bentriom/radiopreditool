
import io, sys
import pandas as pd
import numpy as np
from scipy.stats.distributions import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radiopreditool_utils import *

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sksurv.util import Surv, check_y_survival
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score, brier_score
from lifelines import CoxPHFitter

# Helpers for compatibility between lifelines / sksurv
def get_risk_scores(cph_object, X):
    if type(cph_object) is CoxPHSurvivalAnalysis or type(cph_object) is CoxnetSurvivalAnalysis:
        return cph_object.predict(X)
    elif type(cph_object) is CoxPHFitter:
        return cph_object.predict_partial_hazard(X)
    else:
        raise TypeError(f"Unrecognized Cox object type: {type(cph_object)}")

def get_probs_bs(cph_object, X, bs_time, ibs_timeline):
    if type(cph_object) is CoxPHSurvivalAnalysis or type(cph_object) is CoxnetSurvivalAnalysis:
        coxph_surv_func = cph_object.predict_survival_function(X)
        prob_surv = [coxph_surv_func[i](bs_time) for i in range(X.shape[0])]
        ibs_preds = [[surv_func(t) for t in ibs_timeline] for surv_func in coxph_surv_func]
        return prob_surv, ibs_preds 
    elif type(cph_object) is CoxPHFitter:
        prob_surv = cph_object.predict_survival_function(X, bs_time).loc[bs_time, :].values
        df_full_ibs_preds = cph_object.predict_survival_function(X, ibs_timeline)
        ibs_preds = [df_full_ibs_preds[patient].values for patient in df_full_ibs_preds.columns]
        return prob_surv, ibs_preds
    else:
        raise TypeError(f"Unrecognized Cox object type: {type(cph_object)}")

# Plot regularization path in Lasso
def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(11, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns.values
    cmap = plt.cm.get_cmap('tab20', n_features).colors
    color_plot = {coefs.index[i]: cmap[i] for i in range(len(coefs.index))}
    for index, row in coefs.iterrows():
        ax.semilogx(alphas, row[alphas], ".-", label = index, color = color_plot[index])
    sorted_alphas = np.sort(alphas)
    alpha_min = sorted_alphas[0]
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    # Sort by the first on alpha_min for ytick labels
    top_coefs = top_coefs[np.argsort(coefs.loc[top_coefs.index, alpha_min])]
    df_top_coefs = pd.DataFrame({"loc_ytick": coefs.loc[top_coefs.index, alpha_min] - 0.01}, index = top_coefs.index)
    for i, name in enumerate(top_coefs.index):
        # loc_ytick = np.mean(coefs.loc[name, sorted_alphas[0:2]])
        if i < n_highlight - 1:
            future_name = df_top_coefs.index[i+1]
            ytick_gap = df_top_coefs.loc[future_name, "loc_ytick"] - df_top_coefs.loc[name, "loc_ytick"] 
            if abs(ytick_gap) <= 0.02:
                df_top_coefs.loc[name, "loc_ytick"] -= 0.01
                df_top_coefs.loc[future_name, "loc_ytick"] += 0.01
        loc_ytick = df_top_coefs.loc[name, "loc_ytick"]
        plt.text(
            alpha_min, loc_ytick, name + "   ",
            horizontalalignment = "right",
            verticalalignment = "center",
            fontsize = "small",
            color = color_plot[name],
            rotation = 5
        )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("penalty")
    ax.set_ylabel("Cox model's coefficient")
    plt.margins(tight = True)

# Plot CV results
def plot_cv_results(df, alpha_lr):
    df_plot = df.sort_values("alpha")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df_plot["alpha"], df_plot["mean_score"])
    ax.fill_between(df_plot["alpha"], df_plot["mean_score"] - df_plot["std_score"], 
                    df_plot["mean_score"] + df_plot["std_score"], alpha = .15)
    df_nbr_features = df_plot.drop_duplicates(subset = ["non_zero_coefs"], keep = "first")
    for i, row in enumerate(df_nbr_features.itertuples()):
        sign_text = 1 if i % 2 == 0 else -1
        ax.text(row.alpha, row.mean_score + sign_text * row.std_score, row.non_zero_coefs, horizontalalignment = "center")
        ax.scatter(row.alpha, row.mean_score, marker = "x", color = "red")
    ax.set_xscale("log")
    ax.set_ylabel("concordance index")
    ax.set_xlabel("alpha")
    ax.axvline(alpha_lr, c = "C2")
    ax.axvline(df.iloc[0]["alpha"], c = "C1", linestyle = "--")
    ax.axhline(0.5, color = "grey", linestyle = "--")
    ax.grid(True)

# Estimate the maximum alpha, debuting with the one that maximizes the score,
# that is not rejected by likelihood ratio test
def max_alpha_lr_test(df_cv_res, df_coxph_train, event_col, duration_col, l1_ratio, log_name = ""):
    # covariates = [col for col in df_coxph_train if col not in [event_col, duration_col]]
    # X_train, y_train = df_coxph_train.loc[:,covariates], df_coxph_train.loc[:,[duration_col, event_col]] 
    # y_train[event_col].replace({1.0: True, 0.0: False}, inplace = True)
    # surv_y_train = Surv.from_dataframe(event_col, duration_col, y_train)
    logger = logging.getLogger(log_name)
    alpha_max_cv = df_cv_res.iloc[0]["alpha"]
    nbr_features_max_cv = df_cv_res.iloc[0]["non_zero_coefs"]
    df_alpha_candidates = df_cv_res.copy()
    df_alpha_candidates = df_alpha_candidates.loc[(df_alpha_candidates["alpha"] > alpha_max_cv) &  
                                                  (df_alpha_candidates["non_zero_coefs"] < nbr_features_max_cv), :].sort_values("alpha")
    df_alpha_candidates.drop_duplicates(subset = ["non_zero_coefs"], keep = "first", inplace = True)
    df_alpha_candidates.index = df_alpha_candidates["alpha"]
    coxph_ref = CoxPHFitter(penalizer = alpha_max_cv, l1_ratio = l1_ratio)
    coxph_ref.fit(df_coxph_train, duration_col, event_col, step_size = 0.6)
    loglik_ref = coxph_ref.log_likelihood_
    new_alpha, ncoefs_new_alpha = alpha_max_cv, nbr_features_max_cv
    for index, row in df_alpha_candidates.iterrows():
        coxph = CoxPHFitter(penalizer = row["alpha"], l1_ratio = l1_ratio)
        coxph.fit(df_coxph_train, duration_col, event_col, step_size = 0.5)
        loglik_alpha = coxph.log_likelihood_
        loglik_ratio = -2*(loglik_alpha - loglik_ref)
        pvalue = chi2.sf(loglik_ratio, 1)
        # coxnet = CoxnetSurvivalAnalysis(alphas = [new_alpha], l1_ratio = l1_ratio)
        # coxnet.fit(X_train, surv_y_train)
        # print(f"alpha: {new_alpha}, non zero coefs: {sum(np.abs(coxph.params_) > 0)} or {row['non_zero_coefs']}")
        # print(f"loglik coxnet: {sum(coxnet.predict(X_train))}")
        ncoefs_candidate = df_alpha_candidates.loc[row['alpha'],'non_zero_coefs']
        logger.info(f"{new_alpha} ({ncoefs_new_alpha}) vs {row['alpha']} ({ncoefs_candidate}): {pvalue}")
        if pvalue < 0.05:
            break
        new_alpha, ncoefs_new_alpha = row["alpha"], df_alpha_candidates.loc[new_alpha,'non_zero_coefs']
    return new_alpha

# Cross-validation for penalty estimation in regularized Cox
def cv_fit_cox(X_train, surv_y_train, df_coxph_train, event_col, duration_col, analyzes_dir, penalty, name, n_splits = 5, log_name = ""):
    covariates = X_train.columns.values
    stratified_cv = StratifiedKFold(n_splits = n_splits)
    split_cv = stratified_cv.split(X_train, get_events(surv_y_train))
    l1_ratio = 1.0
    if penalty == "ridge":
        coxph = CoxPHSurvivalAnalysis()
        dict_params_coxph = {'alpha': [0.0001, 0.5, 1.0, 1.5, 2.0, 5.0]}
        cv_coxph = GridSearchCV(estimator = coxph, param_grid = dict_params_coxph, cv = split_cv, n_jobs = get_ncpus())
        cv_coxph.fit(X_train, surv_y_train)
        cv_results = cv_coxph.cv_results_
        cv_alphas = [param["alphas"][0] for param in cv_results["params"]]
        cv_mean = cv_results["mean_test_score"]
        cv_std = cv_results["std_test_score"]
        df_cv_res = pd.DataFrame({"alpha": cv_alphas, "mean_score": cv_mean, "std_score": cv_std})
        df_cv_res.sort_values("mean_score", ascending = False, inplace = True)
        df_cv_res.to_csv(analyzes_dir + f"coxph_results/cv_{name}.csv", index = False)
        return cv_coxph.best_estimator_
    elif penalty == "lasso":
        coxnet = CoxnetSurvivalAnalysis(n_alphas = 40, l1_ratio = l1_ratio, tol = 1e-5)
        coxnet.fit(X_train, surv_y_train)
        # Plot coefs
        coefficients_lasso = pd.DataFrame(coxnet.coef_, index = pretty_labels(covariates), columns = coxnet.alphas_)
        plot_coefficients(coefficients_lasso, n_highlight = min(7, len(covariates)))
        plt.savefig(analyzes_dir + f"coxph_plots/regularization_path_{name}.png", dpi = 480, bbox_inches = 'tight')
        plt.close()
        # Gridsearch CV
        list_alphas = coxnet.alphas_
        dict_params_coxnet = {'alphas': [[a] for a in list_alphas]}
        coxnet_grid = CoxnetSurvivalAnalysis(l1_ratio = l1_ratio, tol = 1e-6)
        cv_coxnet = GridSearchCV(estimator = coxnet_grid, param_grid = dict_params_coxnet, cv = split_cv, refit = False, n_jobs = get_ncpus())
        cv_coxnet.fit(X_train, surv_y_train)
        cv_results = cv_coxnet.cv_results_
        cv_alphas = [param["alphas"][0] for param in cv_results["params"]]
        cv_mean = cv_results["mean_test_score"]
        cv_std = cv_results["std_test_score"]
        # Add number of non zeros coefs
        series_nbr_features = (coefficients_lasso.transpose().abs() > 0).sum(axis = 1)
        df_cv_res = pd.DataFrame({"alpha": cv_alphas, "mean_score": cv_mean, "std_score": cv_std})
        df_cv_res.insert(0, "non_zero_coefs", series_nbr_features[df_cv_res["alpha"]].values)
        df_cv_res.sort_values("mean_score", ascending = False, inplace = True)
        df_cv_res.to_csv(analyzes_dir + f"coxph_results/cv_{name}.csv", index = False)
        # Successive likelihood ratio tests
        alpha_lr = max_alpha_lr_test(df_cv_res, df_coxph_train, event_col, duration_col, l1_ratio, log_name = log_name)
        # Plot CV
        plot_cv_results(df_cv_res, alpha_lr)
        plt.savefig(analyzes_dir + f"coxph_plots/mean_error_alphas_{name}.png", dpi = 480, bbox_inches = 'tight')
        plt.close()
        # Refit with best alpha
        best_coxnet = CoxnetSurvivalAnalysis(alphas = [alpha_lr], l1_ratio = l1_ratio, fit_baseline_model = True)
        best_coxnet.fit(X_train, surv_y_train)
        return best_coxnet

# Run a Cox analysis
def coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, 
                   penalty = "ridge", seed = None, test_size = 0.3, name = "", log_name = ""):
    logger = logging.getLogger(log_name)
    logger.info(covariates)
    df_trainset = pd.read_csv(file_trainset)
    df_testset = pd.read_csv(file_testset)
    # Select variables and drop NAs
    df_model_train = df_trainset[covariates + [event_col, duration_col]].dropna()
    df_model_test = df_testset[covariates + [event_col, duration_col]].dropna()
    logger.info(f"Trainset file: {file_trainset}")
    logger.info(f"Testset file: {file_testset}")
    logger.info(f"Penalty: {penalty}")
    logger.info(f"Trainset number of samples: {df_model_train.shape[0]}")
    logger.info(f"Testset number of samples: {df_model_test.shape[0]}")
    logger.info("NAs are dropped")
    logger.info(f"Train/test events (non-censored data): {df_model_train[event_col].sum()} {df_model_test[event_col].sum()}")
    # Prepare train and test datasets
    norm_scaler = StandardScaler()
    df_model_train.loc[:, covariates] = norm_scaler.fit_transform(df_model_train.loc[:, covariates])
    df_model_test.loc[:, covariates] = norm_scaler.transform(df_model_test.loc[:, covariates])
    X_train, y_train = df_model_train.loc[:,covariates], df_model_train.loc[:,[duration_col, event_col]] 
    X_test, y_test = df_model_test.loc[:,covariates], df_model_test.loc[:,[duration_col, event_col]] 
    y_train[event_col].replace({1.0: True, 0.0: False}, inplace = True)
    y_test[event_col].replace({1.0: True, 0.0: False}, inplace = True)
    surv_y_train = Surv.from_dataframe(event_col, duration_col, y_train)
    surv_y_test = Surv.from_dataframe(event_col, duration_col, y_test)
    logger.info(f"Balance train/test events: {event_balance(surv_y_train)} {event_balance(surv_y_test)}")

    # CoxPH model
    if penalty == None:
        best_coxph = CoxPHFitter(penalizer = 0.0, alpha = 0.05)
        best_coxph.fit(df_model_train, duration_col, event_col, step_size = 0.5)
    elif penalty in ["lasso", "ridge"]:
        best_coxph = cv_fit_cox(X_train, surv_y_train, df_model_train, event_col, duration_col, analyzes_dir, penalty, name, log_name = log_name)

    # Predictions of risk score (\beta^t exp(x)) / survival probabilities
    risk_scores_train = get_risk_scores(best_coxph, X_train)
    risk_scores_test = get_risk_scores(best_coxph, X_test)
    ibs_timeline = np.arange(1, 61, step = 1)
    final_time = ibs_timeline[-1]
    prob_surv_train, ibs_preds_train = get_probs_bs(best_coxph, X_train, final_time, ibs_timeline)
    prob_surv_test, ibs_preds_test = get_probs_bs(best_coxph, X_test, final_time, ibs_timeline)
    
    # Summary / coefficients plots
    if penalty == None:
        logger.info("\n" + best_coxph.summary.to_markdown())
        best_coxph.plot()
        fig, ax = plt.gcf(), plt.gca()
        fig.set_size_inches(12, 6)
        plotted_yticks = ax.get_yticklabels()
        [t.set_text(pretty_label(t.get_text())) for t in plotted_yticks]
        ax.set_yticklabels(plotted_yticks)
        plt.yticks(fontsize = "small")
        plt.savefig(analyzes_dir + f"coxph_plots/coefs_{name}.png", dpi = 480, bbox_inches = 'tight')
        plt.close()
    elif penalty in ["lasso", "ridge"]:
        best_coefs = pd.DataFrame(best_coxph.coef_, index = pretty_labels(covariates), columns = ["coefficient"])
        mask_non_zero = best_coefs.iloc[:, 0] != 0
        nbr_non_zero = np.sum(mask_non_zero)
        logger.info(f"Number of non-zero coefficients: {nbr_non_zero}")
        non_zero_coefs = best_coefs.loc[mask_non_zero, :]
        coef_order = non_zero_coefs.abs().sort_values("coefficient").index
        fig, ax = plt.subplots(figsize=(12, 6))
        non_zero_coefs.loc[coef_order].plot.barh(ax = ax, legend = False)
        ax.set_xlabel("coefficient")
        plt.title(name + f" (non zero coeffs: {nbr_non_zero}/{len(covariates)})")
        ax.grid(True)
        fig.savefig(analyzes_dir + f"coxph_plots/coefs_{name}.png", dpi = 480, bbox_inches = 'tight')
        plt.close()
    
    # Metrics
    # Harell's C-index
    coxph_cindex_train = concordance_index_censored(y_train[event_col], y_train[duration_col], risk_scores_train)
    coxph_cindex_test = concordance_index_censored(y_test[event_col], y_test[duration_col], risk_scores_test)
    logger.info(f"C-index trainset: {coxph_cindex_train}")
    logger.info(f"C-index testset: {coxph_cindex_test}")
    # Uno's C-index
    coxph_cindex_uno_train = concordance_index_ipcw(surv_y_train, surv_y_train, risk_scores_train)
    coxph_cindex_uno_test = concordance_index_ipcw(surv_y_train, surv_y_test, risk_scores_test)
    logger.info(f"IPCW C-index trainset: {coxph_cindex_uno_train}")
    logger.info(f"IPCW C-index testset: {coxph_cindex_uno_test}")
    # Brier score
    times_brier_train, brier_score_train = brier_score(surv_y_train, surv_y_train, prob_surv_train, final_time)
    times_brier_test, brier_score_test = brier_score(surv_y_train, surv_y_test, prob_surv_test, final_time)
    ibs_score_train = integrated_brier_score(surv_y_train, surv_y_train, ibs_preds_train, ibs_timeline)
    ibs_score_test = integrated_brier_score(surv_y_train, surv_y_test, ibs_preds_test, ibs_timeline)
    results_train = [coxph_cindex_train[0], coxph_cindex_uno_train[0], brier_score_train[0], ibs_score_train]
    results_train = [round(x, 3) for x in results_train]
    results_test = [coxph_cindex_test[0], coxph_cindex_uno_test[0], brier_score_test[0], ibs_score_test]
    results_test = [round(x, 3) for x in results_test]
    logger.info(f"Brier score at time {final_time} trainset: {brier_score_train}")
    logger.info(f"Brier score at time {final_time} testset: {brier_score_test}")
    logger.info(f"IBS trainset: {ibs_score_train}")
    logger.info(f"IBS testset: {ibs_score_test}")
    logger.info(f"Train: {results_train[0]} & {results_train[1]} & {results_train[2]} & {results_train[3]}") 
    logger.info(f"Test: {results_test[0]} & {results_test[1]} & {results_test[2]} & {results_test[3]}") 
    df_results = pd.DataFrame({"Train": results_train, "Test": results_test}, index = ["C-index", "IPCW C-index", "BS at 60", "IBS"])
    df_results.to_csv(analyzes_dir + f"coxph_results/metrics_{name}.csv", index = True)
    return best_coxph

# Run baseline models
def baseline_models_analysis(file_trainset, file_features_hclust_corr, file_testset, event_col, analyzes_dir):
    log_name = "baseline_models"
    logger = setup_logger(log_name, analyzes_dir + f"{log_name}.log")
    duration_col = "survival_time_years"
    df_trainset = pd.read_csv(file_trainset)
    features_hclust_corr = pd.read_csv(file_features_hclust_corr, header = None)[0].values
    clinical_vars = get_clinical_features(df_trainset, event_col, duration_col)
    os.makedirs(analyzes_dir + "coxph_plots", exist_ok = True)
    os.makedirs(analyzes_dir + "coxph_results", exist_ok = True)

    # Coxph mean dose of heart (1320)
    model_name = "1320_mean"
    covariates = ["1320_original_firstorder_Mean"] + clinical_vars
    logger.info("Model heart mean dose (1320)")
    coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, penalty = None, name = model_name, log_name = log_name)
    
    # Coxph doses volumes indicators of heart (1320)
    model_name = "1320_dosesvol"
    covariates = [feature for feature in df_trainset.columns if re.match("dv_\w+_1320", feature)] + clinical_vars
    logger.info("Model heart doses volumes (1320)")
    coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, penalty = None, name = model_name, log_name = log_name)
    
    # Coxph doses volumes indicators of heart Lasso (1320)
    model_name = "1320_dosesvol_lasso"
    covariates = [feature for feature in df_trainset.columns if re.match("dv_\w+_1320", feature)] + clinical_vars
    logger.info("Model heart doses volumes lasso (1320)")
    coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, penalty = "lasso", name = model_name, log_name = log_name)

def cox_lasso_radiomics(file_trainset, file_features_hclust_corr, file_testset, event_col, analyzes_dir):
    log_name = "cox_lasso_radiomics"
    logger = setup_logger(log_name, analyzes_dir + f"{log_name}.log")
    duration_col = "survival_time_years"
    df_trainset = pd.read_csv(file_trainset)
    features_hclust_corr = pd.read_csv(file_features_hclust_corr, header = None)[0].values
    clinical_vars = get_clinical_features(df_trainset, event_col, duration_col)
    os.makedirs(analyzes_dir + "coxph_plots", exist_ok = True)
    os.makedirs(analyzes_dir + "coxph_results", exist_ok = True)
    
    # Coxph radiomics heart 32X full trainset lasso
    model_name = "32X_radiomics_lasso"
    covariates = [feature for feature in df_trainset.columns if re.match("^32[0-9]_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 32X (full trainset lasso)")
    coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, penalty = "lasso", name = model_name, log_name = log_name)
    
    # Coxph radiomics heart 32X hclust corr features trainset lasso
    model_name = "32X_radiomics_features_hclust_lasso"
    covariates = [feature for feature in features_hclust_corr if re.match("^32[0-9]_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 32X (hclust corr feature elimination trainset lasso)")
    coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, penalty = "lasso", name = model_name, log_name = log_name)
    
    # Coxph radiomics heart 1320 full trainset lasso
    model_name = "1320_radiomics_lasso"
    covariates = [feature for feature in df_trainset.columns if re.match("^1320_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 1320 (full trainset lasso)")
    coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, penalty = "lasso", name = model_name, log_name = log_name)

    # Coxph radiomics heart 1320 hclust corr features trainset lasso
    model_name = "1320_radiomics_features_hclust_lasso"
    covariates = [feature for feature in features_hclust_corr if re.match("^1320_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 1320 (hclust corr feature elimination trainset lasso)")
    coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, penalty = "lasso", name = model_name, log_name = log_name)

