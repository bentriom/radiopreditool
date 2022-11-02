
## Cox learning in R

BASELINE_MODELS_COX = ["1320_mean", "1320_dosesvol"]
BASELINE_MODELS_LASSO = ["1320_dosesvol_lasso"]
COX_RADIOMICS_LASSO_1320_ALL = ["1320_radiomics_firstorder_lasso_all", "1320_radiomics_full_lasso_all"]
COX_RADIOMICS_LASSO_32X_ALL = ["32X_radiomics_firstorder_lasso_all", "32X_radiomics_full_lasso_all"]
COX_RADIOMICS_LASSO_1320_FE_HCLUST = ["1320_radiomics_firstorder_lasso_features_hclust_corr", \
                                      "1320_radiomics_full_lasso_features_hclust_corr"]
COX_RADIOMICS_LASSO_32X_FE_HCLUST = [ "32X_radiomics_firstorder_lasso_features_hclust_corr", \
                                      "32X_radiomics_full_lasso_features_hclust_corr"]
COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL = ["1320_radiomics_firstorder_bootstrap_lasso_all", \
                                          "1320_radiomics_full_bootstrap_lasso_all"]
COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL = ["32X_radiomics_firstorder_bootstrap_lasso_all", \
                                         "32X_radiomics_full_bootstrap_lasso_all"]
COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST = ["1320_radiomics_firstorder_bootstrap_lasso_features_hclust_corr", \
                                      "1320_radiomics_full_bootstrap_lasso_features_hclust_corr"]
COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST = [ "32X_radiomics_firstorder_bootstrap_lasso_features_hclust_corr", \
                                      "32X_radiomics_full_bootstrap_lasso_features_hclust_corr"]
COX_RADIOMICS_LASSO_ALL = COX_RADIOMICS_LASSO_32X_ALL + COX_RADIOMICS_LASSO_1320_ALL
COX_RADIOMICS_LASSO_FE_HCLUST =  COX_RADIOMICS_LASSO_32X_FE_HCLUST + COX_RADIOMICS_LASSO_1320_FE_HCLUST
COX_RADIOMICS_LASSO = COX_RADIOMICS_LASSO_ALL + COX_RADIOMICS_LASSO_FE_HCLUST

# Baseline models

rule baseline_analysis_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz"
    output:
        ANALYZES_DIR + "baseline_models_R.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = BASELINE_MODELS_COX),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = BASELINE_MODELS_COX),
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv_mean_error.png", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/regularization_path.png", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv.csv", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = BASELINE_MODELS_LASSO),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = BASELINE_MODELS_LASSO)
    threads:
        5
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R {CONFIGFILE_PATH} baseline_models"

# Cox Lasso radiomics no feature elimination

rule cox_lasso_radiomics_whole_heart_all_R:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz",
    output:
        ANALYZES_DIR + "cox_lasso_radiomics_R_1320_all.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv_mean_error.png", model = COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/regularization_path.png", model = COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv.csv", model = COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = COX_RADIOMICS_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = COX_RADIOMICS_LASSO_1320_ALL)
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
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/freq_selected_features.png", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/bootstrap_selected_features.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/final_selected_features.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_ALL)
    threads:
        1 if "SLURM_CPUS_PER_TASK" in os.environ else get_ncpus()
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
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv_mean_error.png", model = COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/regularization_path.png", model = COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv.csv", model = COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = COX_RADIOMICS_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = COX_RADIOMICS_LASSO_32X_ALL)
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
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/freq_selected_features.png", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/bootstrap_selected_features.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/final_selected_features.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_ALL)
    threads:
        1 if "SLURM_CPUS_PER_TASK" in os.environ else get_ncpus()
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
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv_mean_error.png", model = COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/regularization_path.png", model = COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv.csv", model = COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = COX_RADIOMICS_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = COX_RADIOMICS_LASSO_1320_FE_HCLUST)
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
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/freq_selected_features.png", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/bootstrap_selected_features.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/final_selected_features.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST)
    threads:
        1 if "SLURM_CPUS_PER_TASK" in os.environ else get_ncpus()
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
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv_mean_error.png", model = COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/regularization_path.png", model = COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/cv.csv", model = COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = COX_RADIOMICS_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = COX_RADIOMICS_LASSO_32X_FE_HCLUST)
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
        expand(ANALYZES_DIR + "coxph_R/{model}/coefs.png", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/freq_selected_features.png", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/bootstrap_selected_features.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/final_selected_features.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/metrics.csv", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST),
        expand(ANALYZES_DIR + "coxph_R/{model}/model.rds", model = COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST)
    threads:
        1 if "SLURM_CPUS_PER_TASK" in os.environ else get_ncpus()
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/cox_learning.R {CONFIGFILE_PATH} cox_bootstrap_lasso_radiomics_features_hclust_corr 32X"

rule multiple_scores_baseline_analysis_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = BASELINE_MODELS_LASSO)
    output:
        ANALYZES_DIR + "multiple_scores_baseline_models_R.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv",
               model = BASELINE_MODELS_COX + BASELINE_MODELS_LASSO)
    threads:
        min(get_ncpus() - 1, NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} baseline_models"


rule multiple_scores_cox_lasso_radiomics_all_R:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = COX_RADIOMICS_LASSO_ALL)
    output:
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics_R_all.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv", model = COX_RADIOMICS_LASSO_ALL)
    threads:
        1 if "SLURM_CPUS_PER_TASK" in os.environ else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} cox_lasso_radiomics_all"

rule multiple_scores_cox_lasso_radiomics_features_hclust_corr_R:
    input:
        ANALYZES_DIR + "features_hclust_corr.csv",
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "coxph_R/{model}/best_params.csv", model = COX_RADIOMICS_LASSO_FE_HCLUST)
    output:
        ANALYZES_DIR + "multiple_scores_cox_lasso_radiomics_R_features_hclust_corr.log",
        expand(ANALYZES_DIR + "coxph_R/{model}/" + str(NB_ESTIM_SCORE_MODELS) + "_runs_test_metrics.csv", model = COX_RADIOMICS_LASSO_FE_HCLUST)
    threads:
        1 if "SLURM_CPUS_PER_TASK" in os.environ else min(get_ncpus(), NB_ESTIM_SCORE_MODELS)
    conda:
        "../envs/cox_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/multiple_scores_cox.R {CONFIGFILE_PATH} cox_lasso_radiomics_features_hclust_corr"

