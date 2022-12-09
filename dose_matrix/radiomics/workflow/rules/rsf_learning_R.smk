
## Random Survival Forest learning in R

RSF_RADIOMICS_32X_ALL = ["32X_radiomics_firstorder_all", "32X_radiomics_full_all"]
RSF_RADIOMICS_1320_ALL = ["1320_radiomics_firstorder_all", "1320_radiomics_full_all", "1320_dosesvol"]
RSF_RADIOMICS_ALL = RSF_RADIOMICS_32X_ALL + RSF_RADIOMICS_1320_ALL
RSF_RADIOMICS_32X_FE_HCLUST = ["32X_radiomics_firstorder_features_hclust_corr", "32X_radiomics_full_features_hclust_corr"]
RSF_RADIOMICS_1320_FE_HCLUST = ["1320_radiomics_firstorder_features_hclust_corr", "1320_radiomics_full_features_hclust_corr"]
RSF_RADIOMICS_FE_HCLUST = RSF_RADIOMICS_32X_FE_HCLUST + RSF_RADIOMICS_1320_FE_HCLUST

# RSF no feature elimination

rule rsf_subparts_heart_analysis:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz"
    output:
        ANALYZES_DIR + "rsf_all_32X.log",
        expand(ANALYZES_DIR + "rsf/{model}/cv.csv", model = RSF_RADIOMICS_32X_ALL),
        expand(ANALYZES_DIR + "rsf/{model}/metrics.csv", model = RSF_RADIOMICS_32X_ALL),
        expand(ANALYZES_DIR + "rsf/{model}/model.rds", model = RSF_RADIOMICS_32X_ALL)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else get_ncpus()
    shell:
        f"Rscript workflow/scripts/rsf_learning.R {CONFIGFILE_PATH} rsf_radiomics_all 32X"

rule rsf_subparts_heart_vimp:
    input:
        expand(ANALYZES_DIR + "rsf/{model}/model.rds", model = RSF_RADIOMICS_32X_ALL)
    output:
        expand(ANALYZES_DIR + "rsf/{model}/rsf_vimp.png", model = RSF_RADIOMICS_32X_ALL),
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else get_ncpus()
    shell:
        f"Rscript workflow/scripts/rsf_vimp.R {ANALYZES_DIR} {' '.join(RSF_RADIOMICS_32X_ALL)}"

rule rsf_whole_heart_analysis:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz"
    output:
        ANALYZES_DIR + "rsf_all_1320.log",
        expand(ANALYZES_DIR + "rsf/{model}/cv.csv", model = RSF_RADIOMICS_1320_ALL),
        expand(ANALYZES_DIR + "rsf/{model}/metrics.csv", model = RSF_RADIOMICS_1320_ALL),
        expand(ANALYZES_DIR + "rsf/{model}/model.rds", model = RSF_RADIOMICS_1320_ALL)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else get_ncpus()
    shell:
        f"Rscript workflow/scripts/rsf_learning.R {CONFIGFILE_PATH} rsf_radiomics_all 1320"

rule rsf_whole_heart_vimp:
    input:
        expand(ANALYZES_DIR + "rsf/{model}/model.rds", model = RSF_RADIOMICS_1320_ALL)
    output:
        expand(ANALYZES_DIR + "rsf/{model}/rsf_vimp.png", model = RSF_RADIOMICS_1320_ALL),
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else get_ncpus()
    shell:
        f"Rscript workflow/scripts/rsf_vimp.R {ANALYZES_DIR} {' '.join(RSF_RADIOMICS_1320_ALL)}"

rule multiple_scores_rsf:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
        # expand(ANALYZES_DIR + "rsf/{model}/cv.csv", model = RSF_RADIOMICS_ALL)
    output:
        ANALYZES_DIR + "multiple_scores_rsf_all.log",
        expand(ANALYZES_DIR + "rsf/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = RSF_RADIOMICS_ALL),
        expand(ANALYZES_DIR + "rsf/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = RSF_RADIOMICS_ALL)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    shell:
        f"Rscript workflow/scripts/multiple_scores_rsf.R {CONFIGFILE_PATH} rsf_radiomics_all"

# RSF feature elimination with hclustering / correlation

rule rsf_subparts_heart_features_hclust_corr_analysis:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
        ANALYZES_DIR + "screening/features_hclust_corr.csv"
    output:
        ANALYZES_DIR + "rsf_features_hclust_corr_32X.log",
        expand(ANALYZES_DIR + "rsf/{model}/cv.csv", model = RSF_RADIOMICS_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "rsf/{model}/metrics.csv", model = RSF_RADIOMICS_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "rsf/{model}/model.rds", model = RSF_RADIOMICS_32X_FE_HCLUST)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else get_ncpus()
    shell:
        f"Rscript workflow/scripts/rsf_learning.R {CONFIGFILE_PATH} rsf_radiomics_features_hclust_corr 32X"

rule rsf_subparts_heart_features_hclust_corr_vimp:
    input:
        expand(ANALYZES_DIR + "rsf/{model}/model.rds", model = RSF_RADIOMICS_32X_FE_HCLUST)
    output:
        expand(ANALYZES_DIR + "rsf/{model}/rsf_vimp.png", model = RSF_RADIOMICS_32X_FE_HCLUST),
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else get_ncpus()
    shell:
        f"Rscript workflow/scripts/rsf_vimp.R {ANALYZES_DIR} {' '.join(RSF_RADIOMICS_32X_FE_HCLUST)}"


rule rsf_whole_heart_features_hclust_corr_analysis:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
        ANALYZES_DIR + "screening/features_hclust_corr.csv"
    output:
        ANALYZES_DIR + "rsf_features_hclust_corr_1320.log",
        expand(ANALYZES_DIR + "rsf/{model}/cv.csv", model = RSF_RADIOMICS_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "rsf/{model}/metrics.csv", model = RSF_RADIOMICS_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "rsf/{model}/model.rds", model = RSF_RADIOMICS_1320_FE_HCLUST)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else get_ncpus()
    shell:
        f"Rscript workflow/scripts/rsf_learning.R {CONFIGFILE_PATH} rsf_radiomics_features_hclust_corr 1320"

rule rsf_whole_heart_features_hclust_corr_vimp:
    input:
        expand(ANALYZES_DIR + "rsf/{model}/model.rds", model = RSF_RADIOMICS_1320_FE_HCLUST)
    output:
        expand(ANALYZES_DIR + "rsf/{model}/rsf_vimp.png", model = RSF_RADIOMICS_1320_FE_HCLUST),
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else get_ncpus()
    shell:
        f"Rscript workflow/scripts/rsf_vimp.R {ANALYZES_DIR} {' '.join(RSF_RADIOMICS_1320_FE_HCLUST)}"

rule multiple_scores_rsf_features_hclust_corr:
    input:
        expand(ANALYZES_DIR + "screening/features_hclust_corr_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
        # expand(ANALYZES_DIR + "rsf/{model}/cv.csv", model = RSF_RADIOMICS_FE_HCLUST)
    output:
        ANALYZES_DIR + "multiple_scores_rsf_features_hclust_corr.log",
        expand(ANALYZES_DIR + "rsf/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = RSF_RADIOMICS_FE_HCLUST),
        expand(ANALYZES_DIR + "rsf/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_full_test_metrics.csv",
               model = RSF_RADIOMICS_FE_HCLUST)
    conda:
        "../envs/rsf_R_env.yaml"
    threads:
        1 if is_slurm_run() else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    shell:
        f"Rscript workflow/scripts/multiple_scores_rsf.R {CONFIGFILE_PATH} rsf_radiomics_features_hclust_corr"

rule rsf_vimp:
    input:
        expand(ANALYZES_DIR + "rsf/{model}/rsf_vimp.png", model = RSF_RADIOMICS_32X_ALL),
        expand(ANALYZES_DIR + "rsf/{model}/rsf_vimp.png", model = RSF_RADIOMICS_1320_ALL),
        expand(ANALYZES_DIR + "rsf/{model}/rsf_vimp.png", model = RSF_RADIOMICS_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "rsf/{model}/rsf_vimp.png", model = RSF_RADIOMICS_1320_FE_HCLUST)
    output:
        f"{ANALYZES_DIR}rsf_vimp.log"
    shell:
        f"touch {ANALYZES_DIR}rsf_vimp.log"

