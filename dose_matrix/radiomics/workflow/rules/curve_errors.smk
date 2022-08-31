
COXLASSO = ["1320_dosesvol_lasso", "32X_radiomics_full_lasso_features_hclust_corr"]
rule pec_models:
    input:
        ANALYZES_DIR + "features_hclust_corr.csv",
        ANALYZES_DIR + "datasets/dataset.csv.gz",
        ANALYZES_DIR + "rsf_results/cv_1320_radiomics_full_all.csv",
        expand(ANALYZES_DIR + "coxph_R_results/best_params_{model}csv", model = COXLASSO),
        expand(ANALYZES_DIR + "coxph_R_results/path_lambda_{model}.csv", model = COXLASSO)
    output:
        ANALYZES_DIR + "pec_plots/bootcv.png"
    threads:
        get_ncpus() - 1
    conda:
        "../envs/pec_R_env.yaml"
    run:
        f"Rscript workflow/scripts/curve_errors.R {ANALYZES_DIR} {EVENT_COL}"

