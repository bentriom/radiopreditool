
import pandas as pd
import numpy as np
from radiopreditool_utils import *
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sksurv.util import Surv, check_y_survival
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score

def coxph_analysis(df_trainset, covariates, event_col, duration_col, seed = 21, test_size = 0.3):
    logger = logging.getLogger("baseline_models")
    df_model = df_trainset[covariates + [event_col, duration_col]]
    df_model = df_model.dropna()
    # Prepare train and test datasets
    data_X, data_y = df_model[covariates], df_model[[duration_col, event_col]] 
    data_X_num = OneHotEncoder().fit_transform(data_X)
    data_y[event_col] = data_y[event_col].replace({1.0: True, 0.0: False})
    structured_y = Surv.from_dataframe(event_col, duration_col, data_y)
    X, y = data_X_num, structured_y
    X_train, X_test, y_train, y_test = train_test_split(data_X_num, structured_y,
                                                        random_state = seed, shuffle = True,
                                                        test_size = test_size, stratify = data_y[event_col])
    logger.info(f"Balance train/test vhd: {event_balance(y_train)} {event_balance(y_test)}")

    # CoxPH model
    coxph = CoxPHSurvivalAnalysis()
    stratified_cv = StratifiedKFold(n_splits=3)
    split_cv = stratified_cv.split(X_train, get_events(y_train))
    dict_params_coxph = {'alpha': [0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]}
    cv_coxph = GridSearchCV(estimator = coxph, param_grid = dict_params_coxph, cv = split_cv)
    cv_coxph.fit(X_train, y_train)
    logger.info(str(cv_coxph.cv_results_))

    # Metrics
    # C-index
    coxph_cindex_train = cv_coxph.best_estimator_.score(X_train, y_train)
    coxph_cindex_test = cv_coxph.best_estimator_.score(X_test, y_test)
    logger.info(f"C-index. Train: {coxph_cindex_train} Test: {coxph_cindex_test}")
    # Brier score
    coxph_surv_func_train = cv_coxph.best_estimator_.predict_survival_function(X_train)
    coxph_surv_func_test = cv_coxph.best_estimator_.predict_survival_function(X_test)
    final_time = 0.99 * max(data_y[duration_col])
    logger.info(f"Brier score end time: {final_time}")
    prob_surv_test = [coxph_surv_func_test[i](final_time) for i in range(len(y_test))]
    times_test = get_times(y_test)
    logger.info(f"min: {min(times_test)}, max: {max(times_test)}")
    times_brier, scores_brier = brier_score(y_train, y_test, prob_surv_test, final_time)
    logger.info(f"Test: {scores_brier}")

def baseline_models_analysis(file_trainset, analyzes_dir): 
    event_col = "Pathologie_cardiaque"
    duration_col = "survival_time_years"
    logger = setup_logger("baseline_models", analyzes_dir + "baseline_models.log")
    df_trainset = pd.read_csv(file_trainset)

    # Mean dose 320 (heart)
    covariates = ["320_original_firstorder_Mean"]
    logger.info("Model 320 mean dose")
    coxph_analysis(df_trainset, covariates, event_col, duration_col)

