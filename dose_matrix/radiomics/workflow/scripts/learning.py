
import pandas as pd
import numpy as np
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

def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(13, 5))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(
            alpha_min, coef, name + "   ",
            horizontalalignment="right",
            verticalalignment="center"
        )

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")

def cv_fit_cox(X_train, surv_y_train, covariates, analyzes_dir, penalty, name):
    stratified_cv = StratifiedKFold(n_splits = 3)
    split_cv = stratified_cv.split(X_train, get_events(surv_y_train))
    if penalty == "ridge":
        coxph = CoxPHSurvivalAnalysis()
        dict_params_coxph = {'alpha': [0.0001, 0.5, 1.0, 1.5, 2.0, 5.0]}
        cv_coxph = GridSearchCV(estimator = coxph, param_grid = dict_params_coxph, cv = split_cv, n_jobs = get_ncpus())
        cv_coxph.fit(X_train, surv_y_train)
        return cv_coxph.best_estimator_
    elif penalty == "lasso":
        coxnet = CoxnetSurvivalAnalysis(n_alphas = 100, l1_ratio = 1.0, alpha_min_ratio = 0.01, max_iter = 100)
        coxnet.fit(X_train, surv_y_train)
        # Plot coefs
        coefficients_lasso = pd.DataFrame(coxnet.coef_, index = covariates, columns = np.round(coxnet.alphas_, 5))
        plot_coefficients(coefficients_lasso, n_highlight = 5)
        plt.savefig(analyzes_dir + f"coxph_plots/coef_select_alphas_{name}.png", dpi = 480)
        plt.close()
        # Gridsearch CV
        list_alphas = coxnet.alphas_
        dict_params_coxnet = {'alphas': [[a] for a in list_alphas]}
        coxnet_grid = CoxnetSurvivalAnalysis(l1_ratio = 0.95)
        cv_coxnet = GridSearchCV(estimator = coxnet_grid, param_grid = dict_params_coxnet, cv = split_cv, refit = False, n_jobs = get_ncpus())
        cv_coxnet.fit(X_train, surv_y_train)
        cv_results = cv_coxnet.cv_results_
        # Plot CV
        cv_alphas = [param["alphas"][0] for param in cv_results["params"]]
        cv_mean = cv_results["mean_test_score"]
        cv_std = cv_results["std_test_score"]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(cv_alphas, cv_mean)
        ax.fill_between(cv_alphas, cv_mean - cv_std, cv_mean + cv_std, alpha = .15)
        ax.set_xscale("log")
        ax.set_ylabel("concordance index")
        ax.set_xlabel("alpha")
        ax.axvline(cv_coxnet.best_params_["alphas"][0], c = "C1")
        ax.axhline(0.5, color = "grey", linestyle = "--")
        ax.grid(True)
        plt.savefig(analyzes_dir + f"coxph_plots/mean_error_alphas_{name}.png", dpi = 480)
        plt.close()
        best_coxnet = CoxnetSurvivalAnalysis(**cv_coxnet.best_params_, fit_baseline_model = True)
        best_coxnet.fit(X_train, surv_y_train)
        return best_coxnet

def coxph_analysis(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, penalty = "lasso", seed = None, test_size = 0.3, name = ""):
    logger = logging.getLogger("baseline_models")
    logger.info(covariates)
    df_model_train = df_trainset[covariates + [event_col, duration_col]].dropna()
    df_model_test = df_testset[covariates + [event_col, duration_col]].dropna()
    logger.info(f"Penalty: {penalty}")
    logger.info(f"Trainset number of samples: {df_model_train.shape[0]}")
    logger.info(f"Testset number of samples: {df_model_test.shape[0]}")
    logger.info(f"Train/test events (non-censored data): {df_model_train[event_col].sum()} {df_model_test[event_col].sum()}")
    # Prepare train and test datasets
    X_train, y_train = df_model_train.loc[:,covariates], df_model_train.loc[:,[duration_col, event_col]] 
    X_test, y_test = df_model_test.loc[:,covariates], df_model_test.loc[:,[duration_col, event_col]] 
    norm_scaler = StandardScaler()
    X_train = norm_scaler.fit_transform(X_train)
    X_test = norm_scaler.transform(X_test)
    y_train[event_col].replace({1.0: True, 0.0: False}, inplace = True)
    y_test[event_col].replace({1.0: True, 0.0: False}, inplace = True)
    surv_y_train = Surv.from_dataframe(event_col, duration_col, y_train)
    surv_y_test = Surv.from_dataframe(event_col, duration_col, y_test)
    logger.info(f"Balance train/test events: {event_balance(surv_y_train)} {event_balance(surv_y_test)}")

    # CoxPH model
    best_coxph = cv_fit_cox(X_train, surv_y_train, covariates, analyzes_dir, penalty, name)

    # Coefficients
    best_coefs = pd.DataFrame(best_coxph.coef_, index = covariates, columns = ["coefficient"])
    mask_non_zero = best_coefs.iloc[:, 0] != 0
    nbr_non_zero = np.sum(mask_non_zero)
    logger.info(f"Number of non-zero coefficients: {nbr_non_zero}")
    non_zero_coefs = best_coefs.loc[mask_non_zero, :]
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index
    fig, ax = plt.subplots(figsize=(12, 6))
    non_zero_coefs.loc[coef_order].plot.barh(ax = ax, legend = False)
    ax.set_xlabel("coefficient")
    plt.title(name + f" (non zero coeffs: {nbr_non_zero}/{len(covariates)}")
    ax.grid(True)
    fig.savefig(analyzes_dir + f"coxph_plots/coefs_{name}.png", dpi = 480)

    # Metrics
    # Harell's C-index
    # CoxPH.predict() predcits risk score
    coxph_cindex_train = concordance_index_censored(y_train[event_col], y_train[duration_col], best_coxph.predict(X_train))
    coxph_cindex_test = concordance_index_censored(y_test[event_col], y_test[duration_col], best_coxph.predict(X_test))
    logger.info(f"C-index trainset: {coxph_cindex_train}")
    logger.info(f"C-index testset: {coxph_cindex_test}")
    # Uno's C-index
    coxph_cindex_uno_train = concordance_index_ipcw(surv_y_train, surv_y_train, best_coxph.predict(X_train))
    coxph_cindex_uno_test = concordance_index_ipcw(surv_y_train, surv_y_test, best_coxph.predict(X_test))
    logger.info(f"Uno's C-index trainset: {coxph_cindex_uno_train}")
    logger.info(f"Uno's C-index testset: {coxph_cindex_uno_test}")
    # Brier score
    ibs_timeline = np.arange(1, 61, step = 1)
    final_time = ibs_timeline[-1]
    coxph_surv_func_train = best_coxph.predict_survival_function(X_train)
    coxph_surv_func_test = best_coxph.predict_survival_function(X_test)
    prob_surv_train = [coxph_surv_func_train[i](final_time) for i in range(len(surv_y_train))]
    prob_surv_test = [coxph_surv_func_test[i](final_time) for i in range(len(surv_y_test))]
    times_brier_train, scores_brier_train = brier_score(surv_y_train, surv_y_train, prob_surv_train, final_time)
    times_brier_test, scores_brier_test = brier_score(surv_y_train, surv_y_test, prob_surv_test, final_time)
    logger.info(f"Brier score at time {final_time} trainset: {scores_brier_train}")
    logger.info(f"Brier score at time {final_time} testset: {scores_brier_test}")
    ibs_preds_train = [[surv_func(t) for t in ibs_timeline] for surv_func in coxph_surv_func_train]
    ibs_preds_test = [[surv_func(t) for t in ibs_timeline] for surv_func in coxph_surv_func_test]
    ibs_score_train = integrated_brier_score(surv_y_train, surv_y_train, ibs_preds_train, ibs_timeline)
    ibs_score_test = integrated_brier_score(surv_y_train, surv_y_test, ibs_preds_test, ibs_timeline)
    logger.info(f"IBS trainset: {ibs_score_train}")
    logger.info(f"IBS testset: {ibs_score_test}")

def baseline_models_analysis(file_trainset, file_preprocessed_trainset, file_testset, event_col, analyzes_dir):
    duration_col = "survival_time_years"
    logger = setup_logger("baseline_models", analyzes_dir + "baseline_models.log")
    df_trainset = pd.read_csv(file_trainset)
    df_preprocessed_trainset = pd.read_csv(file_preprocessed_trainset)
    df_testset = pd.read_csv(file_testset)
    clinical_vars = get_clinical_features(df_trainset, event_col, duration_col)
    os.makedirs(analyzes_dir + "coxph_plots", exist_ok = True)

    # Coxph mean dose of heart (1320)
    model_name = "1320_mean_lasso"
    covariates = ["1320_original_firstorder_Mean"] + clinical_vars
    logger.info("Model heart mean dose (1320)")
    coxph_analysis(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, name = model_name)

    # Coxph radiomics heart 32X
    model_name = "32X_radiomics_lasso"
    covariates = [feature for feature in df_preprocessed_trainset.columns if re.match("^32[0-9]_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 32X (filtered)")
    coxph_analysis(df_preprocessed_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, name = model_name)
    
    # Coxph radiomics heart 1320
    model_name = "1320_radiomics_lasso"
    covariates = [feature for feature in df_preprocessed_trainset.columns if re.match("^1320_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 1320 (filtered)")
    coxph_analysis(df_preprocessed_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, name = model_name)
   
