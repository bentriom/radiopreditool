
import io, sys
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool
from scipy.stats.distributions import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radiopreditool_utils import *
from coxph_utils import *
from itertools import starmap

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sksurv.util import Surv, check_y_survival
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score, brier_score
from lifelines import CoxPHFitter

# Run a complete Cox analysis with cross-validation
def coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, 
                   penalty = "lasso", seed = None, test_size = 0.3, model_name = "", log_name = ""):
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
        best_coxph = cv_fit_cox(X_train, surv_y_train, df_model_train, event_col, duration_col, analyzes_dir, penalty, model_name, log_name = log_name)

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
        plt.savefig(analyzes_dir + f"coxph_plots/coefs_{model_name}.png", dpi = 480, bbox_inches = 'tight')
        plt.close()
    elif penalty in ["lasso", "ridge"]:
        plot_non_zero_coefs(best_coxph.coef_, pretty_labels(covariates), model_name)
        plt.savefig(analyzes_dir + f"coxph_plots/coefs_{model_name}.png", dpi = 480, bbox_inches = 'tight')
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
    df_results.to_csv(analyzes_dir + f"coxph_results/metrics_{model_name}.csv", index = True)
    return best_coxph

# Run baseline models analysis: cross-validation, plots and scores over testset
def baseline_models_analysis(file_trainset, file_testset, event_col, analyzes_dir, only_plots = False):
    log_name = "baseline_models"
    logger = setup_logger(log_name, analyzes_dir + f"{log_name}.log")
    duration_col = "survival_time_years"
    df_trainset = pd.read_csv(file_trainset)
    clinical_vars = get_clinical_features(df_trainset, event_col, duration_col)
    os.makedirs(analyzes_dir + "coxph_plots", exist_ok = True)
    os.makedirs(analyzes_dir + "coxph_results", exist_ok = True)

    # Coxph mean dose of heart (1320)
    model_name = "1320_mean"
    covariates = ["1320_original_firstorder_Mean"] + clinical_vars
    logger.info("Model heart mean dose (1320)")
    coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, penalty = None, model_name = model_name, log_name = log_name)
    
    # Coxph doses volumes indicators of heart (1320)
    model_name = "1320_dosesvol"
    covariates = [feature for feature in df_trainset.columns if re.match("dv_\w+_1320", feature)] + clinical_vars
    logger.info("Model heart doses volumes (1320)")
    coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, 
                   penalty = None, model_name = model_name, log_name = log_name)
    
    # Coxph doses volumes indicators of heart Lasso (1320)
    model_name = "1320_dosesvol_lasso"
    covariates = [feature for feature in df_trainset.columns if re.match("dv_\w+_1320", feature)] + clinical_vars
    logger.info("Model heart doses volumes lasso (1320)")
    if only_plots:
        redo_plot_lasso_model(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, model_name)
    else:
        coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, 
                       penalty = "lasso", model_name = model_name, log_name = log_name)

# Score of baseline models over several testsets
def multiple_scores_baseline_models(nb_estim, event_col, analyzes_dir, duration_col = "survival_time_years"):
    log_name = "multiple_scores_baseline_models"
    logger = setup_logger(log_name, analyzes_dir + f"{log_name}.log")
    logger.info(f"Number of CPUs: {get_ncpus()}")
    df_trainset0 = pd.read_csv(analyzes_dir + "datasets/trainset_0.csv.gz")
    clinical_vars = get_clinical_features(df_trainset0, event_col, duration_col)
    os.makedirs(analyzes_dir + "coxph_results", exist_ok = True)
    index_results = ["C-index", "IPCW C-index", "BS at 60", "IBS"]
    
    # Coxph mean dose of heart (1320)
    model_name = "1320_mean"
    covariates = ["1320_original_firstorder_Mean"] + clinical_vars
    logger.info("Model heart mean dose (1320)")
    with Pool(get_ncpus()-1) as p:
        results = p.map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col, 
                                                    analyzes_dir = analyzes_dir, penalty = None, model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(results)    
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = index_results)
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)

    # Coxph doses volumes indicators of heart (1320)
    model_name = "1320_dosesvol"
    covariates = [feature for feature in df_trainset0.columns if re.match("dv_\w+_1320", feature)] + clinical_vars
    logger.info("Model heart doses volumes (1320)")
    with Pool(get_ncpus()-1) as p:
        results = p.map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col, 
                                                    analyzes_dir = analyzes_dir, penalty = None, model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(results)    
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = index_results)
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)
    
    # Coxph doses volumes indicators of heart Lasso (1320)
    model_name = "1320_dosesvol_lasso"
    covariates = [feature for feature in df_trainset0.columns if re.match("dv_\w+_1320", feature)] + clinical_vars
    logger.info("Model heart doses volumes lasso (1320)")
    with Pool(get_ncpus()-1) as p:
        results = p.map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col, 
                                                    analyzes_dir = analyzes_dir, penalty = "lasso", model_name = model_name, log_name = log_name), range(nb_estim))
    
    results = np.asarray(results)    
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = index_results)
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)

# Run Lasso models on dosiomics: cross-validation, plots and metrics over testset
def cox_lasso_radiomics(file_trainset, file_features_hclust_corr, file_testset, event_col, analyzes_dir, only_plots = False):
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
    if only_plots:
        redo_plot_lasso_model(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, model_name)
    else:
        coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, 
                       penalty = "lasso", model_name = model_name, log_name = log_name)
    
    # Coxph radiomics heart 32X hclust corr features trainset lasso
    model_name = "32X_radiomics_features_hclust_lasso"
    covariates = [feature for feature in features_hclust_corr if re.match("^32[0-9]_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 32X (hclust corr feature elimination trainset lasso)")
    if only_plots:
        redo_plot_lasso_model(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, model_name)
    else:
        coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, 
                       penalty = "lasso", model_name = model_name, log_name = log_name)
    
    # Coxph radiomics heart 1320 full trainset lasso
    model_name = "1320_radiomics_lasso"
    covariates = [feature for feature in df_trainset.columns if re.match("^1320_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 1320 (full trainset lasso)")
    if only_plots:
        redo_plot_lasso_model(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, model_name)
    else:
        coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, 
                       penalty = "lasso", model_name = model_name, log_name = log_name)

    # Coxph radiomics heart 1320 hclust corr features trainset lasso
    model_name = "1320_radiomics_features_hclust_lasso"
    covariates = [feature for feature in features_hclust_corr if re.match("^1320_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 1320 (hclust corr feature elimination trainset lasso)")
    if only_plots:
        redo_plot_lasso_model(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, model_name)
    else:
        coxph_analysis(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, 
                       penalty = "lasso", model_name = model_name, log_name = log_name)

# Score of baseline models over several testsets
def multiple_scores_cox_lasso_radiomics(nb_estim, file_features_hclust_corr, event_col, analyzes_dir, duration_col = "survival_time_years"):
    log_name = "multiple_scores_cox_lasso_radiomics"
    logger = setup_logger(log_name, analyzes_dir + f"{log_name}.log")
    logger.info(f"Number of CPUs: {get_ncpus()}")
    df_trainset0 = pd.read_csv(analyzes_dir + "datasets/trainset_0.csv.gz")
    features_hclust_corr = pd.read_csv(file_features_hclust_corr, header = None)[0].values
    clinical_vars = get_clinical_features(df_trainset0, event_col, duration_col)
    os.makedirs(analyzes_dir + "coxph_results", exist_ok = True)
    index_results = ["C-index", "IPCW C-index", "BS at 60", "IBS"]
    
    # Coxph radiomics heart 32X full trainset lasso
    model_name = "32X_radiomics_lasso"
    covariates = [feature for feature in df_trainset0.columns if re.match("^32[0-9]_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 32X (full trainset lasso)")
    with Pool(get_ncpus()-1) as p:
        results = p.map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col, 
                                                    analyzes_dir = analyzes_dir, penalty = "lasso", model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(results)    
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = index_results)
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)
    
    # Coxph radiomics heart 32X hclust corr features trainset lasso
    model_name = "32X_radiomics_features_hclust_lasso"
    covariates = [feature for feature in features_hclust_corr if re.match("^32[0-9]_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 32X (hclust corr feature elimination trainset lasso)")
    with Pool(get_ncpus()-1) as p:
        results = p.map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col, 
                                                    analyzes_dir = analyzes_dir, penalty = "lasso", model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(results)    
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = index_results)
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)
    
    # Coxph radiomics heart 1320 full trainset lasso
    model_name = "1320_radiomics_lasso"
    covariates = [feature for feature in df_trainset0.columns if re.match("^1320_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 1320 (full trainset lasso)")
    with Pool(get_ncpus()-1) as p:
        results = p.map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col, 
                                                    analyzes_dir = analyzes_dir, penalty = "lasso", model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(results)    
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = index_results)
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)

    # Coxph radiomics heart 1320 hclust corr features trainset lasso
    model_name = "1320_radiomics_features_hclust_lasso"
    covariates = [feature for feature in features_hclust_corr if re.match("^1320_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 1320 (hclust corr feature elimination trainset lasso)")
    with Pool(get_ncpus()-1) as p:
        results = p.map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col, 
                                                    analyzes_dir = analyzes_dir, penalty = "lasso", model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(results)    
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = index_results)
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)


# Sequential
def _seq_multiple_scores_baseline_models(nb_estim, event_col, analyzes_dir):
    log_name = "multiple_scores_baseline_models"
    logger = setup_logger(log_name, analyzes_dir + f"{log_name}.log")
    logger.info(f"Number of CPUs: {get_ncpus()}")
    duration_col = "survival_time_years"
    df_trainset0 = pd.read_csv(analyzes_dir + "datasets/trainset_0.csv.gz")
    clinical_vars = get_clinical_features(df_trainset0, event_col, duration_col)
    os.makedirs(analyzes_dir + "coxph_results", exist_ok = True)

    # Coxph mean dose of heart (1320)
    model_name = "1320_mean"
    covariates = ["1320_original_firstorder_Mean"] + clinical_vars
    logger.info("Model heart mean dose (1320)")
    results = map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col,
                                              analyzes_dir = analyzes_dir, penalty = None, model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(list(results))
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = ["C-index", "IPCW C-index", "BS at 60", "IBS"])
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)

    # Coxph doses volumes indicators of heart (1320)
    model_name = "1320_dosesvol"
    covariates = [feature for feature in df_trainset0.columns if re.match("dv_\w+_1320", feature)] + clinical_vars
    logger.info("Model heart doses volumes (1320)")
    results = map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col,
                                              analyzes_dir = analyzes_dir, penalty = None, model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(list(results))
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = ["C-index", "IPCW C-index", "BS at 60", "IBS"])
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)

    # Coxph doses volumes indicators of heart Lasso (1320)
    model_name = "1320_dosesvol_lasso"
    covariates = [feature for feature in df_trainset0.columns if re.match("dv_\w+_1320", feature)] + clinical_vars
    logger.info("Model heart doses volumes lasso (1320)")
    results = map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col,
                                              analyzes_dir = analyzes_dir, penalty = "lasso", model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(list(results))
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = ["C-index", "IPCW C-index", "BS at 60", "IBS"])
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)

def _seq_multiple_scores_cox_lasso_radiomics(nb_estim, file_features_hclust_corr, event_col, analyzes_dir):
    log_name = "multiple_scores_cox_lasso_radiomics"
    logger = setup_logger(log_name, analyzes_dir + f"{log_name}.log")
    duration_col = "survival_time_years"
    df_trainset0 = pd.read_csv(analyzes_dir + "datasets/trainset_0.csv.gz")
    features_hclust_corr = pd.read_csv(file_features_hclust_corr, header = None)[0].values
    clinical_vars = get_clinical_features(df_trainset0, event_col, duration_col)
    os.makedirs(analyzes_dir + "coxph_results", exist_ok = True)

    # Coxph radiomics heart 32X full trainset lasso
    model_name = "32X_radiomics_lasso"
    covariates = [feature for feature in df_trainset0.columns if re.match("^32[0-9]_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 32X (full trainset lasso)")
    results = map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col,
                                              analyzes_dir = analyzes_dir, penalty = "lasso", model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(list(results))
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = ["C-index", "IPCW C-index", "BS at 60", "IBS"])
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)

    # Coxph radiomics heart 32X hclust corr features trainset lasso
    model_name = "32X_radiomics_features_hclust_lasso"
    covariates = [feature for feature in features_hclust_corr if re.match("^32[0-9]_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 32X (hclust corr feature elimination trainset lasso)")
    results = map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col,
                                              analyzes_dir = analyzes_dir, penalty = "lasso", model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(list(results))
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = ["C-index", "IPCW C-index", "BS at 60", "IBS"])
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)

    # Coxph radiomics heart 1320 full trainset lasso
    model_name = "1320_radiomics_lasso"
    covariates = [feature for feature in df_trainset0.columns if re.match("^1320_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 1320 (full trainset lasso)")
    results = map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col,
                                              analyzes_dir = analyzes_dir, penalty = "lasso", model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(list(results))
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = ["C-index", "IPCW C-index", "BS at 60", "IBS"])
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)

    # Coxph radiomics heart 1320 hclust corr features trainset lasso
    model_name = "1320_radiomics_features_hclust_lasso"
    covariates = [feature for feature in features_hclust_corr if re.match("^1320_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 1320 (hclust corr feature elimination trainset lasso)")
    results = map(partial(refit_best_cox_id, covariates = covariates, event_col = event_col, duration_col = duration_col,
                                              analyzes_dir = analyzes_dir, penalty = "lasso", model_name = model_name, log_name = log_name), range(nb_estim))
    results = np.asarray(list(results))
    df_results = pd.DataFrame({"Mean": np.mean(results, axis = 0), "Std": np.std(results, axis = 0)}, index = ["C-index", "IPCW C-index", "BS at 60", "IBS"])
    df_results.to_csv(analyzes_dir + f"coxph_results/{nb_estim}_runs_test_metrics_{model_name}.csv", index = True)

