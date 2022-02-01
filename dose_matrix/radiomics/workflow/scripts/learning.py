
import pandas as pd
import numpy as np
from radiopreditool_utils import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sksurv.util import Surv, check_y_survival
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, brier_score

def coxph_analysis(df_trainset, df_testset, covariates, event_col, duration_col, seed = None, test_size = 0.3):
    logger = logging.getLogger("baseline_models")
    logger.info(covariates)
    df_model_train = df_trainset[covariates + [event_col, duration_col]].dropna()
    df_model_test = df_testset[covariates + [event_col, duration_col]].dropna()
    logger.info(f"Trainset number of samples: {df_model_train.shape[0]}")
    logger.info(f"Testset number of samples: {df_model_test.shape[0]}")
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
    logger.info(f"Balance train/test event: {event_balance(surv_y_train)} {event_balance(surv_y_test)}")

    # CoxPH model
    coxph = CoxPHSurvivalAnalysis()
    stratified_cv = StratifiedKFold(n_splits = 3)
    split_cv = stratified_cv.split(X_train, get_events(surv_y_train))
    dict_params_coxph = {'alpha': [0.0001, 0.5, 1.0, 1.5, 2.0, 5.0]}
    cv_coxph = GridSearchCV(estimator = coxph, param_grid = dict_params_coxph, cv = split_cv)
    cv_coxph.fit(X_train, surv_y_train)
    logger.info(str(cv_coxph.cv_results_))

    # Metrics
    # C-index
    coxph_cindex_train = cv_coxph.best_estimator_.score(X_train, surv_y_train)
    coxph_cindex_test = cv_coxph.best_estimator_.score(X_test, surv_y_test)
    logger.info(f"C-index trainset: {coxph_cindex_train}")
    logger.info(f"C-index testset: {coxph_cindex_test}")
    # Brier score
    final_time = 50
    coxph_surv_func_train = cv_coxph.best_estimator_.predict_survival_function(X_train)
    coxph_surv_func_test = cv_coxph.best_estimator_.predict_survival_function(X_test)
    prob_surv_train = [coxph_surv_func_train[i](final_time) for i in range(len(surv_y_train))]
    prob_surv_test = [coxph_surv_func_test[i](final_time) for i in range(len(surv_y_test))]
    times_brier_train, scores_brier_train = brier_score(surv_y_train, surv_y_train, prob_surv_train, final_time)
    times_brier_test, scores_brier_test = brier_score(surv_y_train, surv_y_test, prob_surv_test, final_time)
    logger.info(f"Brier score at time {final_time} trainset: {scores_brier_train}")
    logger.info(f"Brier score at time {final_time} testset: {scores_brier_test}")
    ibs_timeline = np.arange(5, 51, step = 5)
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

    # Coxph mean dose of heart (1320)
    covariates = ["1320_original_firstorder_Mean"] + clinical_vars
    logger.info("Model heart mean dose (1320)")
    coxph_analysis(df_trainset, df_testset, covariates, event_col, duration_col)

    # Coxph radiomics heart 32X
    covariates = [feature for feature in df_preprocessed_trainset.columns if re.match("^32[0-9]_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 32X (filtered)")
    coxph_analysis(df_preprocessed_trainset, df_testset, covariates, event_col, duration_col)
    
    # Coxph radiomics heart 1320
    covariates = [feature for feature in df_preprocessed_trainset.columns if re.match("^1320_.*", feature)] + clinical_vars
    logger.info("Model heart dosiomics 1320 (filtered)")
    coxph_analysis(df_preprocessed_trainset, df_testset, covariates, event_col, duration_col)
    
