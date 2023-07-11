
WHOLE_BODY_COX_RADIOMICS_LASSO_ALL = ["whole_body_radiomics_full_lasso_all"]
WHOLE_BODY_COX_RADIOMICS_SIS_ALL = ["whole_body_radiomics_full_sis_all"]

rule multiple_scores_cox_lasso_radiomics_all_whole_body_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics_whole_body_all_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = WHOLE_BODY_COX_RADIOMICS_LASSO_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = WHOLE_BODY_COX_RADIOMICS_LASSO_ALL),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = WHOLE_BODY_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv_mean_error.png",
               model = WHOLE_BODY_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/regularization_path.png",
               model = WHOLE_BODY_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/cv.csv",
               model = WHOLE_BODY_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/best_params.csv",
               model = WHOLE_BODY_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = WHOLE_BODY_COX_RADIOMICS_LASSO_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} " + \
        f"cox_lasso_radiomics_all whole_body"

rule multiple_scores_cox_sis_radiomics_all_whole_body_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_cox_sis_radiomics_whole_body_all_R.log",
        # Results that summaries all cv runs
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = WHOLE_BODY_COX_RADIOMICS_SIS_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = WHOLE_BODY_COX_RADIOMICS_SIS_ALL),
        # Results for each run
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/coefs.png",
               model = WHOLE_BODY_COX_RADIOMICS_SIS_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/final_selected_features.csv",
               model = WHOLE_BODY_COX_RADIOMICS_SIS_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/{nb_set}/metrics.csv",
               model = WHOLE_BODY_COX_RADIOMICS_SIS_ALL, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} cox_sis_radiomics_all whole_body"

