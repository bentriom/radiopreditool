
## Cox learning in python

BASELINE_MODELS_COX = ["1320_mean", "1320_dosesvol"]
BASELINE_MODELS_LASSO = ["1320_dosesvol_lasso"]
COX_RADIOMICS_LASSO = ["32X_radiomics_lasso", "32X_radiomics_features_hclust_lasso", \
                       "1320_radiomics_lasso", "1320_radiomics_features_hclust_lasso"]

rule baseline_analysis_python:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz"
    output:
        ANALYZES_DIR + "baseline_models.log",
        expand(ANALYZES_DIR + "coxph_plots/coefs_{model}.png", model = BASELINE_MODELS_COX),
        expand(ANALYZES_DIR + "coxph_results/metrics_{model}.csv", model = BASELINE_MODELS_COX),
        expand(ANALYZES_DIR + "coxph_plots/coefs_{model}.png", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_plots/cv_mean_error_{model}.png", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_plots/regularization_path_{model}.png", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_results/cv_{model}.csv", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_results/best_params_{model}.csv", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_results/metrics_{model}.csv", model = BASELINE_MODELS_LASSO)
    threads:
        get_ncpus() - 1
    run:
        learning.baseline_models_analysis(ANALYZES_DIR + "datasets/trainset.csv.gz", ANALYZES_DIR + "datasets/testset.csv.gz", EVENT_COL, ANALYZES_DIR)

rule multiple_scores_baseline_models_python:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_results/best_params_{model}.csv", model = BASELINE_MODELS_LASSO)
    output: 
        ANALYZES_DIR + "multiple_scores_baseline_models.log",
        expand(ANALYZES_DIR + "coxph_results/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics_{model}.csv", model = BASELINE_MODELS_COX + BASELINE_MODELS_LASSO)
    threads:
        get_ncpus() - 1
    run:
        learning.multiple_scores_baseline_models(NB_ESTIM_SCORE_MODELS, EVENT_COL, ANALYZES_DIR)

rule cox_lasso_radiomics_python:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
        ANALYZES_DIR + "features_hclust_corr.csv"
    output:
        ANALYZES_DIR + "cox_lasso_radiomics.log",
        expand(ANALYZES_DIR + "coxph_plots/coefs_{model}.png", model = COX_RADIOMICS_LASSO),
        expand(ANALYZES_DIR + "coxph_plots/cv_mean_error_{model}.png", model = COX_RADIOMICS_LASSO),
        expand(ANALYZES_DIR + "coxph_plots/regularization_path_{model}.png", model = COX_RADIOMICS_LASSO),
        expand(ANALYZES_DIR + "coxph_results/cv_{model}.csv", model = COX_RADIOMICS_LASSO),
        expand(ANALYZES_DIR + "coxph_results/best_params_{model}.csv", model = COX_RADIOMICS_LASSO),
        expand(ANALYZES_DIR + "coxph_results/metrics_{model}.csv", model = COX_RADIOMICS_LASSO)
    threads:
        get_ncpus() - 1
    run:
        learning.cox_lasso_radiomics(ANALYZES_DIR + "datasets/trainset.csv.gz", ANALYZES_DIR + "features_hclust_corr.csv", ANALYZES_DIR + "datasets/testset.csv.gz", EVENT_COL, ANALYZES_DIR)

rule multiple_scores_cox_lasso_radiomics_python:
    input:
        ANALYZES_DIR + "features_hclust_corr.csv",
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_results/best_params_{model}.csv", model = COX_RADIOMICS_LASSO)
    output: 
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics.log",
        expand(ANALYZES_DIR + "coxph_results/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics_{model}.csv", model = COX_RADIOMICS_LASSO)
    threads:
        get_ncpus() - 1
    run:
        learning.multiple_scores_cox_lasso_radiomics(NB_ESTIM_SCORE_MODELS, ANALYZES_DIR + "features_hclust_corr.csv", EVENT_COL, ANALYZES_DIR)

