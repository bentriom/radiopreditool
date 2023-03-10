
BREASTS_BASELINE_MODELS_COX = ["breasts_mean", "breasts_dosesvol"]
BREASTS_BASELINE_MODELS_LASSO = ["breasts_dosesvol_lasso"]
THORAX_COX_RADIOMICS_LASSO_ALL = ["breasts_radiomics_full_lasso_all", "thorax_radiomics_full_lasso_all"]
THORAX_COX_RADIOMICS_LASSO_FE_HCLUST = ["breasts_radiomics_full_lasso_features_hclust_corr", \
                                        "thorax_radiomics_full_lasso_features_hclust_corr"]
THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL = ["breasts_radiomics_full_bootstrap_lasso_all", \
                                            "thorax_radiomics_full_bootstrap_lasso_all"]
THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST = ["breasts_radiomics_full_bootstrap_lasso_features_hclust_corr", \
                                                  "thorax_radiomics_full_bootstrap_lasso_features_hclust_corr"]
THORAX_COX_RADIOMICS_SIS_ALL = ["thorax_radiomics_full_sis_all"]
THORAX_COX_RADIOMICS_SIS_FE_HCLUST = ["thorax_radiomics_full_sis_features_hclust_corr"]

##### Run Cox models learning in R with cross-validation #####

rule multiple_scores_baseline_analysis_breasts_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_baseline_models_breasts_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = BREASTS_BASELINE_MODELS_COX + BREASTS_BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = BREASTS_BASELINE_MODELS_COX + BREASTS_BASELINE_MODELS_LASSO),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = BREASTS_BASELINE_MODELS_COX, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = BREASTS_BASELINE_MODELS_COX, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = BREASTS_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv_mean_error.png",
               model = BREASTS_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/regularization_path.png",
               model = BREASTS_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv.csv",
               model = BREASTS_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/best_params.csv",
               model = BREASTS_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = BREASTS_BASELINE_MODELS_LASSO, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} baseline_models breasts"

rule multiple_scores_cox_lasso_radiomics_all_thorax_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics_breasts_all_R.log",
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics_thorax_all_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_LASSO_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_LASSO_ALL),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = THORAX_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv_mean_error.png",
               model = THORAX_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/regularization_path.png",
               model = THORAX_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv.csv",
               model = THORAX_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/best_params.csv",
               model = THORAX_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = THORAX_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_lasso_radiomics_all breasts && "
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_lasso_radiomics_all thorax"

rule multiple_scores_cox_lasso_radiomics_features_hclust_corr_thorax_R:
    input:
        expand(ANALYZES_DIR + "screening/features_hclust_corr_{nb_set}.csv", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics_breasts_features_hclust_corr_R.log",
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics_thorax_features_hclust_corr_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_LASSO_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_LASSO_FE_HCLUST),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = THORAX_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv_mean_error.png",
               model = THORAX_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/regularization_path.png",
               model = THORAX_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv.csv",
               model = THORAX_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/best_params.csv",
               model = THORAX_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = THORAX_COX_RADIOMICS_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_lasso_radiomics_features_hclust_corr breasts && "
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_lasso_radiomics_features_hclust_corr thorax"

rule multiple_scores_cox_bootstrap_lasso_radiomics_all_thorax_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_bootstrap_lasso_radiomics_breasts_all_R.log",
        ANALYZES_DIR + "multiple_scores_cox_bootstrap_lasso_radiomics_thorax_all_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/freq_selected_features.png",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/bootstrap_selected_features.csv",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/final_selected_features.csv",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_bootstrap_lasso_radiomics_all breasts && "
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_bootstrap_lasso_radiomics_all thorax"

rule multiple_scores_cox_bootstrap_lasso_radiomics_features_hclust_corr_thorax_R:
    input:
        expand(ANALYZES_DIR + "screening/features_hclust_corr_{nb_set}.csv", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_bootstrap_lasso_radiomics_breasts_features_hclust_corr_R.log",
        ANALYZES_DIR + "multiple_scores_cox_bootstrap_lasso_radiomics_thorax_features_hclust_corr_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/freq_selected_features.png",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/bootstrap_selected_features.csv",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/final_selected_features.csv",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = THORAX_COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_bootstrap_lasso_radiomics_features_hclust_corr breasts && "
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_bootstrap_lasso_radiomics_features_hclust_corr thorax"

rule multiple_scores_cox_sis_radiomics_all_thorax_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_sis_radiomics_thorax_all_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_SIS_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_SIS_ALL),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = THORAX_COX_RADIOMICS_SIS_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/final_selected_features.csv",
               model = THORAX_COX_RADIOMICS_SIS_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = THORAX_COX_RADIOMICS_SIS_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} cox_sis_radiomics_all thorax"

rule multiple_scores_cox_sis_radiomics_features_hclust_corr_thorax_R:
    input:
        expand(ANALYZES_DIR + "screening/features_hclust_corr_{nb_set}.csv", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_sis_radiomics_breasts_features_hclust_corr_R.log",
        ANALYZES_DIR + "multiple_scores_cox_sis_radiomics_thorax_features_hclust_corr_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_SIS_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = THORAX_COX_RADIOMICS_SIS_FE_HCLUST),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = THORAX_COX_RADIOMICS_SIS_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/final_selected_features.csv",
               model = THORAX_COX_RADIOMICS_SIS_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = THORAX_COX_RADIOMICS_SIS_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} cox_sis_radiomics_features_hclust_corr thorax"

