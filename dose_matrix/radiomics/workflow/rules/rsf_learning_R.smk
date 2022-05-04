
## Random Survival Forest learning in R

RSF_RADIOMICS_ALL = ["32X_radiomics_firstorder_all", "1320_radiomics_firstorder_all", \
                     "32X_radiomics_full_all", "1320_radiomics_full_all"]
RSF_RADIOMICS_FE_HCLUST = ["32X_radiomics_firstorder_features_hclust_corr", "1320_radiomics_firstorder_features_hclust_corr", \
                           "32X_radiomics_full_features_hclust_corr", "1320_radiomics_full_features_hclust_corr"]

rule rsf_analysis:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz"
    output:
        ANALYZES_DIR + "rsf_all.log",
        expand(ANALYZES_DIR + "rsf_plots/rsf_vimp_{model}.png", model = RSF_RADIOMICS_ALL),
        expand(ANALYZES_DIR + "rsf_results/cv_{model}.csv", model = RSF_RADIOMICS_ALL),
        expand(ANALYZES_DIR + "rsf_results/metrics_{model}.csv", model = RSF_RADIOMICS_ALL)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        get_ncpus() - 1
    shell:
        f"Rscript workflow/scripts/rsf_learning.R {ANALYZES_DIR}datasets/trainset.csv.gz {ANALYZES_DIR}datasets/testset.csv.gz all {EVENT_COL} {ANALYZES_DIR} all"
 
rule multiple_scores_rsf:
    input:
        ANALYZES_DIR + "features_hclust_corr.csv",
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "rsf_results/cv_{model}.csv", model = RSF_RADIOMICS_ALL)
    output: 
        ANALYZES_DIR + "multiple_scores_rsf_all.log",
        expand(ANALYZES_DIR + "rsf_results/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics_{model}.csv", model = RSF_RADIOMICS_ALL)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        min(get_ncpus() - 1, NB_ESTIM_SCORE_MODELS)
    shell:
        f"Rscript workflow/scripts/multiple_scores_rsf.R {NB_ESTIM_SCORE_MODELS} all {EVENT_COL} {ANALYZES_DIR} all"
 
rule rsf_features_hclust_corr_analysis:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
        ANALYZES_DIR + "features_hclust_corr.csv"
    output:
        ANALYZES_DIR + "rsf_features_hclust_corr.log",
        expand(ANALYZES_DIR + "rsf_plots/rsf_vimp_{model}.png", model = RSF_RADIOMICS_FE_HCLUST),
        expand(ANALYZES_DIR + "rsf_results/cv_{model}.csv", model = RSF_RADIOMICS_FE_HCLUST),
        expand(ANALYZES_DIR + "rsf_results/metrics_{model}.csv", model = RSF_RADIOMICS_FE_HCLUST)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        get_ncpus() - 1
    shell:
        f"Rscript workflow/scripts/rsf_learning.R {ANALYZES_DIR}datasets/trainset.csv.gz {ANALYZES_DIR}datasets/testset.csv.gz {ANALYZES_DIR}features_hclust_corr.csv {EVENT_COL} {ANALYZES_DIR} features_hclust_corr"

rule multiple_scores_rsf_features_hclust_corr:
    input:
        ANALYZES_DIR + "features_hclust_corr.csv",
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "rsf_results/cv_{model}.csv", model = RSF_RADIOMICS_FE_HCLUST)
    output: 
        ANALYZES_DIR + "multiple_scores_rsf_features_hclust_corr.log",
        expand(ANALYZES_DIR + "rsf_results/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics_{model}.csv", model = RSF_RADIOMICS_FE_HCLUST)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        min(get_ncpus() - 1, NB_ESTIM_SCORE_MODELS)
    shell:
        f"Rscript workflow/scripts/multiple_scores_rsf.R {NB_ESTIM_SCORE_MODELS} {ANALYZES_DIR}features_hclust_corr.csv {EVENT_COL} {ANALYZES_DIR} features_hclust_corr"

