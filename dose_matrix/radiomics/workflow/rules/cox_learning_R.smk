
## Cox learning in R

BASELINE_MODELS_COX = ["1320_mean", "1320_dosesvol"]
BASELINE_MODELS_LASSO = ["1320_dosesvol_lasso"]
COX_RADIOMICS_LASSO_32X = ["32X_radiomics_firstorder_lasso_all", "32X_radiomics_firstorder_lasso_features_hclust_corr", \
                           "32X_radiomics_full_lasso_all", "32X_radiomics_full_lasso_features_hclust_corr"]
COX_RADIOMICS_LASSO_1320 = ["1320_radiomics_firstorder_lasso_all", "1320_radiomics_firstorder_lasso_features_hclust_corr", \
                            "1320_radiomics_full_lasso_all", "1320_radiomics_full_lasso_features_hclust_corr"]
COX_RADIOMICS_LASSO = COX_RADIOMICS_LASSO_32X + COX_RADIOMICS_LASSO_1320

# Baseline models

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
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R baseline_models {ANALYZES_DIR} {EVENT_COL}"

rule multiple_scores_baseline_analysis_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R_results/best_params_{model}.csv", model = BASELINE_MODELS_LASSO)
    output: 
        ANALYZES_DIR + "multiple_scores_baseline_models_R.log",
        expand(ANALYZES_DIR + "coxph_R_results/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics_{model}.csv", model = BASELINE_MODELS_COX + BASELINE_MODELS_LASSO)
    threads:
        min(get_ncpus() - 1, NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R multiple_scores_baseline_models {NB_ESTIM_SCORE_MODELS} {ANALYZES_DIR} {EVENT_COL} survival_time_years"

# Cox Lasso radiomics no feature elimination

rule cox_lasso_radiomics_whole_heart_all_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
        ANALYZES_DIR + "features_hclust_corr.csv"
    output:
        ANALYZES_DIR + "cox_lasso_radiomics_R_1320_features_hclust_corr.log",
        expand(ANALYZES_DIR + "coxph_R_plots/coefs_{model}.png", model = COX_RADIOMICS_LASSO_1320),
        expand(ANALYZES_DIR + "coxph_R_plots/cv_mean_error_{model}.png", model = COX_RADIOMICS_LASSO_1320),
        expand(ANALYZES_DIR + "coxph_R_plots/regularization_path_{model}.png", model = COX_RADIOMICS_LASSO_1320),
        expand(ANALYZES_DIR + "coxph_R_results/cv_{model}.csv", model = COX_RADIOMICS_LASSO_1320),
        expand(ANALYZES_DIR + "coxph_R_results/best_params_{model}.csv", model = COX_RADIOMICS_LASSO_1320),
        expand(ANALYZES_DIR + "coxph_R_results/metrics_{model}.csv", model = COX_RADIOMICS_LASSO_1320)
    threads:
        1
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R cox_lasso_radiomics_all {ANALYZES_DIR} {EVENT_COL} 1320"

rule cox_lasso_radiomics_subparts_heart_all_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
        ANALYZES_DIR + "features_hclust_corr.csv"
    output:
        ANALYZES_DIR + "cox_lasso_radiomics_R_32X_features_hclust_corr.log",
        expand(ANALYZES_DIR + "coxph_R_plots/coefs_{model}.png", model = COX_RADIOMICS_LASSO_32X),
        expand(ANALYZES_DIR + "coxph_R_plots/cv_mean_error_{model}.png", model = COX_RADIOMICS_LASSO_32X),
        expand(ANALYZES_DIR + "coxph_R_plots/regularization_path_{model}.png", model = COX_RADIOMICS_LASSO_32X),
        expand(ANALYZES_DIR + "coxph_R_results/cv_{model}.csv", model = COX_RADIOMICS_LASSO_32X),
        expand(ANALYZES_DIR + "coxph_R_results/best_params_{model}.csv", model = COX_RADIOMICS_LASSO_32X),
        expand(ANALYZES_DIR + "coxph_R_results/metrics_{model}.csv", model = COX_RADIOMICS_LASSO_32X)
    threads:
        1
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R cox_lasso_radiomics_all {ANALYZES_DIR} {EVENT_COL} 32X"

# Cox Lasso radiomics features hclust corr elimination

rule cox_lasso_radiomics_whole_heart_features_hclust_corr_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
    output:
        ANALYZES_DIR + "cox_lasso_radiomics_R_1320_all.log",
        expand(ANALYZES_DIR + "coxph_R_plots/coefs_{model}.png", model = COX_RADIOMICS_LASSO_1320),
        expand(ANALYZES_DIR + "coxph_R_plots/cv_mean_error_{model}.png", model = COX_RADIOMICS_LASSO_1320),
        expand(ANALYZES_DIR + "coxph_R_plots/regularization_path_{model}.png", model = COX_RADIOMICS_LASSO_1320),
        expand(ANALYZES_DIR + "coxph_R_results/cv_{model}.csv", model = COX_RADIOMICS_LASSO_1320),
        expand(ANALYZES_DIR + "coxph_R_results/best_params_{model}.csv", model = COX_RADIOMICS_LASSO_1320),
        expand(ANALYZES_DIR + "coxph_R_results/metrics_{model}.csv", model = COX_RADIOMICS_LASSO_1320)
    threads:
        1
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R cox_lasso_radiomics_features_hclust_corr {ANALYZES_DIR} {EVENT_COL} {ANALYZES_DIR}features_hclust_corr.csv 1320"

rule cox_lasso_radiomics_subparts_heart_features_hclust_corr_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
    output:
        ANALYZES_DIR + "cox_lasso_radiomics_R_32X_all.log",
        expand(ANALYZES_DIR + "coxph_R_plots/coefs_{model}.png", model = COX_RADIOMICS_LASSO_32X),
        expand(ANALYZES_DIR + "coxph_R_plots/cv_mean_error_{model}.png", model = COX_RADIOMICS_LASSO_32X),
        expand(ANALYZES_DIR + "coxph_R_plots/regularization_path_{model}.png", model = COX_RADIOMICS_LASSO_32X),
        expand(ANALYZES_DIR + "coxph_R_results/cv_{model}.csv", model = COX_RADIOMICS_LASSO_32X),
        expand(ANALYZES_DIR + "coxph_R_results/best_params_{model}.csv", model = COX_RADIOMICS_LASSO_32X),
        expand(ANALYZES_DIR + "coxph_R_results/metrics_{model}.csv", model = COX_RADIOMICS_LASSO_32X)
    threads:
        1
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R cox_lasso_radiomics_features_hclust_corr {ANALYZES_DIR} {EVENT_COL} {ANALYZES_DIR}features_hclust_corr.csv 32X"

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
        min(get_ncpus() - 1, NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R multiple_scores_cox_lasso_radiomics_all {NB_ESTIM_SCORE_MODELS} {ANALYZES_DIR} {EVENT_COL} survival_time_years && "
        f"Rscript workflow/scripts/multiple_scores_cox.R multiple_scores_cox_lasso_radiomics_features_hclust_corr {NB_ESTIM_SCORE_MODELS} {ANALYZES_DIR} {EVENT_COL} survival_time_years {ANALYZES_DIR}features_hclust_corr.csv"

