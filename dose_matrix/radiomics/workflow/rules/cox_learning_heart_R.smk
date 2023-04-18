
## Cox learning in R

HEART_BASELINE_MODELS_COX = ["1320_mean", "1320_dosesvol"]
HEART_BASELINE_MODELS_LASSO = ["1320_dosesvol_lasso"]
HEART_COX_RADIOMICS_LASSO_1320_ALL = ["1320_radiomics_firstorder_lasso_all", "1320_radiomics_full_lasso_all"]
HEART_COX_RADIOMICS_LASSO_32X_ALL = ["32X_radiomics_firstorder_lasso_all", "32X_radiomics_full_lasso_all"]
HEART_COX_RADIOMICS_LASSO_1320_FE_HCLUST = ["1320_radiomics_firstorder_lasso_features_hclust_corr", \
                                            "1320_radiomics_full_lasso_features_hclust_corr"]
HEART_COX_RADIOMICS_LASSO_32X_FE_HCLUST = [ "32X_radiomics_firstorder_lasso_features_hclust_corr", \
                                            "32X_radiomics_full_lasso_features_hclust_corr"]
HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL = ["1320_radiomics_firstorder_bootstrap_lasso_all", \
                                                "1320_radiomics_full_bootstrap_lasso_all"]
HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL = ["32X_radiomics_firstorder_bootstrap_lasso_all", \
                                               "32X_radiomics_full_bootstrap_lasso_all"]
HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST = ["1320_radiomics_firstorder_bootstrap_lasso_features_hclust_corr", \
                                                      "1320_radiomics_full_bootstrap_lasso_features_hclust_corr"]
HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST = [ "32X_radiomics_firstorder_bootstrap_lasso_features_hclust_corr", \
                                                      "32X_radiomics_full_bootstrap_lasso_features_hclust_corr"]
HEART_COX_RADIOMICS_LASSO_ALL = HEART_COX_RADIOMICS_LASSO_32X_ALL + HEART_COX_RADIOMICS_LASSO_1320_ALL
HEART_COX_RADIOMICS_LASSO_FE_HCLUST = HEART_COX_RADIOMICS_LASSO_32X_FE_HCLUST + HEART_COX_RADIOMICS_LASSO_1320_FE_HCLUST
HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL + \
                                          HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL
HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST =  HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST + \
                                                 HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST
HEART_COX_RADIOMICS_LASSO = HEART_COX_RADIOMICS_LASSO_ALL + HEART_COX_RADIOMICS_LASSO_FE_HCLUST
HEART_COX_RADIOMICS_SIS_ALL = ["32X_radiomics_full_sis_all", "1320_marrow_radiomics_full_sis_all", \
                               "32X_marrow_radiomics_full_sis_all"]
HEART_COX_RADIOMICS_SIS_FE_HCLUST = ["32X_radiomics_full_sis_features_hclust_corr", \
                                     "1320_marrow_radiomics_full_sis_features_hclust_corr", \
                                     "32X_marrow_radiomics_full_sis_features_hclust_corr"]

##### Run Cox models learning with cross-validation #####

rule multiple_scores_baseline_analysis_heart_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_baseline_models_heart_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = HEART_BASELINE_MODELS_COX + HEART_BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = HEART_BASELINE_MODELS_COX + HEART_BASELINE_MODELS_LASSO),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = HEART_BASELINE_MODELS_COX, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = HEART_BASELINE_MODELS_COX, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = HEART_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv_mean_error.png",
               model = HEART_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/regularization_path.png",
               model = HEART_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv.csv",
               model = HEART_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/best_params.csv",
               model = HEART_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = HEART_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} baseline_models heart"

rule multiple_scores_cox_lasso_radiomics_all_heart_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics_heart_all_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = HEART_COX_RADIOMICS_LASSO_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = HEART_COX_RADIOMICS_LASSO_ALL),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = HEART_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv_mean_error.png",
               model = HEART_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/regularization_path.png",
               model = HEART_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv.csv",
               model = HEART_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/best_params.csv",
               model = HEART_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = HEART_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_lasso_radiomics_all heart"

rule multiple_scores_cox_lasso_radiomics_features_hclust_corr_heart_R:
    input:
        expand(ANALYZES_DIR + "screening/features_hclust_corr_{nb_set}.csv", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics_heart_features_hclust_corr_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = HEART_COX_RADIOMICS_LASSO_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = HEART_COX_RADIOMICS_LASSO_FE_HCLUST),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = HEART_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv_mean_error.png",
               model = HEART_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/regularization_path.png",
               model = HEART_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv.csv",
               model = HEART_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/best_params.csv",
               model = HEART_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = HEART_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_lasso_radiomics_features_hclust_corr heart"

rule multiple_scores_cox_bootstrap_lasso_radiomics_all_heart_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_bootstrap_lasso_radiomics_heart_all_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/freq_selected_features.png",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/bootstrap_selected_features.csv",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/final_selected_features.csv",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_bootstrap_lasso_radiomics_all heart"

rule multiple_scores_cox_bootstrap_lasso_radiomics_features_hclust_corr_heart_R:
    input:
        expand(ANALYZES_DIR + "screening/features_hclust_corr_{nb_set}.csv", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_bootstrap_lasso_radiomics_heart_features_hclust_corr_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/freq_selected_features.png",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/bootstrap_selected_features.csv",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/final_selected_features.csv",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_bootstrap_lasso_radiomics_features_hclust_corr heart"

rule multiple_scores_cox_sis_radiomics_all_heart_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_sis_radiomics_heart_all_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = HEART_COX_RADIOMICS_SIS_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = HEART_COX_RADIOMICS_SIS_ALL),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = HEART_COX_RADIOMICS_SIS_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/final_selected_features.csv",
               model = HEART_COX_RADIOMICS_SIS_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = HEART_COX_RADIOMICS_SIS_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} cox_sis_radiomics_all heart"

rule multiple_scores_cox_sis_radiomics_features_hclust_corr_heart_R:
    input:
        expand(ANALYZES_DIR + "screening/features_hclust_corr_{nb_set}.csv", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_sis_radiomics_heart_features_hclust_corr_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = HEART_COX_RADIOMICS_SIS_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = HEART_COX_RADIOMICS_SIS_FE_HCLUST),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = HEART_COX_RADIOMICS_SIS_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/final_selected_features.csv",
               model = HEART_COX_RADIOMICS_SIS_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = HEART_COX_RADIOMICS_SIS_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} cox_sis_radiomics_features_hclust_corr heart"

##### Deprecated rules: learning with only one train/test set #####

# Baseline models
rule baseline_analysis_heart_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz"
    output:
        ANALYZES_DIR + "baseline_models_R_heart.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = HEART_BASELINE_MODELS_COX),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = HEART_BASELINE_MODELS_COX),
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = HEART_BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv_mean_error.png", model = HEART_BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/regularization_path.png", model = HEART_BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv.csv", model = HEART_BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = HEART_BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = HEART_BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = HEART_BASELINE_MODELS_LASSO)
    threads:
        5
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R {CONFIGFILE_PATH} baseline_models heart"

# Cox Lasso radiomics no feature elimination
rule cox_lasso_radiomics_whole_heart_all_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
    output:
        ANALYZES_DIR + "cox_lasso_radiomics_R_1320_all.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = HEART_COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv_mean_error.png", model = HEART_COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/regularization_path.png", model = HEART_COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv.csv", model = HEART_COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = HEART_COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = HEART_COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = HEART_COX_RADIOMICS_LASSO_1320_ALL)
    threads:
        5
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R {CONFIGFILE_PATH} cox_lasso_radiomics_all 1320"

rule cox_bootstrap_lasso_radiomics_whole_heart_all_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
    output:
        ANALYZES_DIR + "cox_bootstrap_lasso_radiomics_R_1320_all.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/freq_selected_features.png", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/bootstrap_selected_features.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/final_selected_features.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL)
    threads:
        1 if is_slurm_run() else get_ncpus()
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R {CONFIGFILE_PATH} cox_bootstrap_lasso_radiomics_all 1320"

rule cox_lasso_radiomics_subparts_heart_all_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
    output:
        ANALYZES_DIR + "cox_lasso_radiomics_R_32X_all.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = HEART_COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv_mean_error.png", model = HEART_COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/regularization_path.png", model = HEART_COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv.csv", model = HEART_COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = HEART_COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = HEART_COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = HEART_COX_RADIOMICS_LASSO_32X_ALL)
    threads:
        5
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R {CONFIGFILE_PATH} cox_lasso_radiomics_all 32X"

rule cox_bootstrap_lasso_radiomics_subparts_heart_all_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
    output:
        ANALYZES_DIR + "cox_bootstrap_lasso_radiomics_R_32X_all.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/freq_selected_features.png", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/bootstrap_selected_features.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/final_selected_features.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL)
    threads:
        1 if is_slurm_run() else get_ncpus()
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R {CONFIGFILE_PATH} cox_bootstrap_lasso_radiomics_all 32X"

# Cox Lasso radiomics features hclust corr elimination
rule cox_lasso_radiomics_whole_heart_features_hclust_corr_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
        ANALYZES_DIR + "features_hclust_corr.csv"
    output:
        ANALYZES_DIR + "cox_lasso_radiomics_R_1320_features_hclust_corr.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = HEART_COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv_mean_error.png", model = HEART_COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/regularization_path.png", model = HEART_COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv.csv", model = HEART_COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = HEART_COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = HEART_COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = HEART_COX_RADIOMICS_LASSO_1320_FE_HCLUST)
    threads:
        5
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R {CONFIGFILE_PATH} cox_lasso_radiomics_features_hclust_corr 1320"

rule cox_bootstrap_lasso_radiomics_whole_heart_features_hclust_corr_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
        ANALYZES_DIR + "features_hclust_corr.csv"
    output:
        ANALYZES_DIR + "cox_bootstrap_lasso_radiomics_R_1320_features_hclust_corr.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/freq_selected_features.png", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/bootstrap_selected_features.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/final_selected_features.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST)
    threads:
        1 if is_slurm_run() else get_ncpus()
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R {CONFIGFILE_PATH} cox_bootstrap_lasso_radiomics_features_hclust_corr 1320"

rule cox_lasso_radiomics_subparts_heart_features_hclust_corr_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
        ANALYZES_DIR + "features_hclust_corr.csv"
    output:
        ANALYZES_DIR + "cox_lasso_radiomics_R_32X_features_hclust_corr.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = HEART_COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv_mean_error.png", model = HEART_COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/regularization_path.png", model = HEART_COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv.csv", model = HEART_COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = HEART_COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = HEART_COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = HEART_COX_RADIOMICS_LASSO_32X_FE_HCLUST)
    threads:
        5
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R {CONFIGFILE_PATH} cox_lasso_radiomics_features_hclust_corr 32X"

rule cox_bootstrap_lasso_radiomics_subparts_heart_features_hclust_corr_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
        ANALYZES_DIR + "features_hclust_corr.csv"
    output:
        ANALYZES_DIR + "cox_bootstrap_lasso_radiomics_R_32X_features_hclust_corr.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/freq_selected_features.png", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/bootstrap_selected_features.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/final_selected_features.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = HEART_COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST)
    threads:
        1 if is_slurm_run() else get_ncpus()
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R {CONFIGFILE_PATH} cox_bootstrap_lasso_radiomics_features_hclust_corr 32X"

