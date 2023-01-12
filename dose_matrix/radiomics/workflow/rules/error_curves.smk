
COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST = ["1320_radiomics_firstorder_bootstrap_lasso_features_hclust_corr", \
                                      "1320_radiomics_full_bootstrap_lasso_features_hclust_corr"]
COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST = [ "32X_radiomics_firstorder_bootstrap_lasso_features_hclust_corr", \
                                      "32X_radiomics_full_bootstrap_lasso_features_hclust_corr"]
COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST =  COX_RADIOMICS_BOOTSTRAP_LASSO_32X_FE_HCLUST + \
                                           COX_RADIOMICS_BOOTSTRAP_LASSO_1320_FE_HCLUST
rule error_curves_models:
    input:
        ANALYZES_DIR + "screening/features_hclust_corr.csv",
        rules.multiple_scores_baseline_analysis_R.output,
        rules.multiple_scores_rsf.output,
        rules.multiple_scores_cox_lasso_radiomics_all_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_features_hclust_corr_R.output,
        expand(ANALYZES_DIR + "coxph_R/{model}/5_runs_full_test_metrics.csv",
               model = COX_RADIOMICS_BOOTSTRAP_LASSO_FE_HCLUST, nb_set = range(NB_ESTIM_SCORE_MODELS)),
    output:
        ANALYZES_DIR + "error_curves/boot_error.rds",
        ANALYZES_DIR + "plots/error_curve_ipcw_cindex.png",
        ANALYZES_DIR + "plots/error_curve_brier_score.png"
    threads:
        1
    conda:
        "../envs/error_curves_R_env.yaml"
    shell:
        f"Rscript workflow/scripts/error_curves.R {ANALYZES_DIR} {EVENT_COL}"

