
rule pec_models:
    input:
        rules.multiple_scores_baseline_analysis_R.output,
        rules.multiple_scores_rsf.output,
        rules.multiple_scores_cox_lasso_radiomics_all_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_features_hclust_corr_R.output,
    output:
        ANALYZES_DIR + "pec_results/boot_error.rds",
        ANALYZES_DIR + "plots/error_curve_ipcw_cindex.png",
        ANALYZES_DIR + "plots/error_curve_brier_score.png"
    threads:
        get_ncpus() - 1
    conda:
        "../envs/pec_R_env.yaml"
    run:
        f"Rscript workflow/scripts/error_curves.R {ANALYZES_DIR} {EVENT_COL}"

