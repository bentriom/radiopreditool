
WHOLE_BODY_RSF_RADIOMICS_ALL = ["whole_body_radiomics_full_all"]

##### Run RSF models learning in R with cross-validation #####

rule multiple_scores_rsf_whole_body:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        ANALYZES_DIR + "multiple_scores_rsf_whole_body_all.log",
        expand(ANALYZES_DIR + "rsf/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = WHOLE_BODY_RSF_RADIOMICS_ALL),
        expand(ANALYZES_DIR + "rsf/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = WHOLE_BODY_RSF_RADIOMICS_ALL)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    shell:
        f"Rscript workflow/scripts/multiple_scores_rsf.R {CONFIGFILE_PATH} rsf_radiomics_all whole_body"

