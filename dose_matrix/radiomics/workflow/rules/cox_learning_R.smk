
## Cox learning in R

BASELINE_MODELS_COX = ["1320_mean", "1320_dosesvol"]
BASELINE_MODELS_LASSO = ["1320_dosesvol_lasso"]
COX_RADIOMICS_LASSO = ["32X_radiomics_lasso_all", "32X_radiomics_lasso_features_hclust_corr", \
                       "1320_radiomics_lasso_all", "1320_radiomics_lasso_features_hclust_corr"]

rule baseline_analysis_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz"
    output:
        ANALYZES_DIR + "baseline_models_R.log",
        expand(ANALYZES_DIR + "coxph_R_plots/coefs_{model}.png", model = BASELINE_MODELS_COX),
        expand(ANALYZES_DIR + "coxph_R_results/metrics_{model}.csv", model = BASELINE_MODELS_COX),
        expand(ANALYZES_DIR + "coxph_R_plots/coefs_{model}.png", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R_plots/cv_mean_error_{model}.png", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R_plots/regularization_path_{model}.png", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R_results/cv_{model}.csv", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R_results/best_params_{model}.csv", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R_results/metrics_{model}.csv", model = BASELINE_MODELS_LASSO)
    threads:
        1
    conda:
        "envs/cox_R_env.yaml"
    run:
        f"Rscript workflow/scripts/cox_learning.R baseline_models {ANALYZES_DIR} {EVENT_COL} survival_time_years"

rule multiple_scores_baseline_analysis_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R_results/best_params_{model}.csv", model = BASELINE_MODELS_LASSO)
    output: 
        ANALYZES_DIR + "multiple_scores_baseline_models_R.log",
        expand(ANALYZES_DIR + "coxph_R_results/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics_{model}.csv", model = BASELINE_MODELS_COX + BASELINE_MODELS_LASSO)
    threads:
        get_ncpus() - 1
    conda:
        "envs/cox_R_env.yaml"
    run:
        f"Rscript workflow/scripts/multiple_scores_cox.R multiple_scores_baseline_models {NB_ESTIM_SCORE_MODELS} {ANALYZES_DIR} {EVENT_COL} survival_time_years"

rule cox_lasso_radiomics_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
        ANALYZES_DIR + "features_hclust_corr.csv"
    output:
        ANALYZES_DIR + "cox_lasso_radiomics_R_all.log",
        ANALYZES_DIR + "cox_lasso_radiomics_R_features_hclust_corr.log",
        expand(ANALYZES_DIR + "coxph_R_plots/coefs_{model}.png", model = COX_RADIOMICS_LASSO),
        expand(ANALYZES_DIR + "coxph_R_plots/cv_mean_error_{model}.png", model = COX_RADIOMICS_LASSO),
        expand(ANALYZES_DIR + "coxph_R_plots/regularization_path_{model}.png", model = COX_RADIOMICS_LASSO),
        expand(ANALYZES_DIR + "coxph_R_results/cv_{model}.csv", model = COX_RADIOMICS_LASSO),
        expand(ANALYZES_DIR + "coxph_R_results/best_params_{model}.csv", model = COX_RADIOMICS_LASSO),
        expand(ANALYZES_DIR + "coxph_R_results/metrics_{model}.csv", model = COX_RADIOMICS_LASSO)
    threads:
        1
    conda:
        "envs/cox_R_env.yaml"
    run:
        f"Rscript workflow/scripts/cox_learning.R cox_lasso_radiomics_all {ANALYZES_DIR} {EVENT_COL} survival_time_years"
        f"Rscript workflow/scripts/cox_learning.R cox_lasso_radiomics_features_hclust_corr {ANALYZES_DIR} {EVENT_COL} survival_time_years {ANALYZES_DIR}features_hclust_corr.csv"

rule multiple_scores_cox_lasso_radiomics_R:
    input:
        ANALYZES_DIR + "features_hclust_corr.csv",
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R_results/best_params_{model}.csv", model = COX_RADIOMICS_LASSO)
    output: 
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics_R_all.log",
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics_R_features_hclust_corr.log",
        expand(ANALYZES_DIR + "coxph_R_results/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics_{model}.csv", model = COX_RADIOMICS_LASSO)
    threads:
        get_ncpus() - 1
    conda:
        "envs/cox_R_env.yaml"
    run:
        f"Rscript workflow/scripts/multiple_scores_cox.R multiple_scores_cox_lasso_radiomics_all {NB_ESTIM_SCORE_MODELS} {ANALYZES_DIR} {EVENT_COL} survival_time_years"
        f"Rscript workflow/scripts/multiple_scores_cox.R multiple_scores_cox_lasso_radiomics_features_hclust_corr {NB_ESTIM_SCORE_MODELS} {ANALYZES_DIR} {EVENT_COL} survival_time_years {ANALYZES_DIR}features_hclust_corr.csv"

