
THORAX_RSF_RADIOMICS_ALL = ["breasts_dosesvol", "breasts_radiomics_full_all", "thorax_radiomics_full_all"]
THORAX_RSF_RADIOMICS_FE_HCLUST = ["breasts_radiomics_full_features_hclust_corr", \
                                 "thorax_radiomics_full_features_hclust_corr"]

##### Run RSF models learning in R with cross-validation #####

rule multiple_scores_rsf_thorax:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_rsf_thorax_all.log",
        expand(ANALYZES_DIR + "rsf/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = THORAX_RSF_RADIOMICS_ALL),
        expand(ANALYZES_DIR + "rsf/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = THORAX_RSF_RADIOMICS_ALL)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    shell:
        f"Rscript workflow/scripts/multiple_scores_rsf.R {CONFIGFILE_PATH} rsf_radiomics_all breasts && "
        f"Rscript workflow/scripts/multiple_scores_rsf.R {CONFIGFILE_PATH} rsf_radiomics_all thorax"

rule multiple_scores_rsf_features_hclust_corr_thorax:
    input:
        expand(ANALYZES_DIR + "screening/features_hclust_corr_{nb_set}.csv", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_rsf_thorax_features_hclust_corr.log",
        expand(ANALYZES_DIR + "rsf/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = THORAX_RSF_RADIOMICS_FE_HCLUST),
        expand(ANALYZES_DIR + "rsf/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = THORAX_RSF_RADIOMICS_FE_HCLUST)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    shell:
        f"Rscript workflow/scripts/multiple_scores_rsf.R {CONFIGFILE_PATH} rsf_radiomics_features_hclust_corr breasts && "
        f"Rscript workflow/scripts/multiple_scores_rsf.R {CONFIGFILE_PATH} rsf_radiomics_features_hclust_corr thorax"

