## Visualisation

rule pca_visualisation:
    input:
        ANALYZES_DIR + "datasets/dataset.csv.gz"
    output:
        ANALYZES_DIR + "pca_viz.log",
        ANALYZES_DIR + "pca/pca_radiomics_all.png"
    run:
        trainset.pca_viz(ANALYZES_DIR + "datasets/dataset.csv.gz", EVENT_COL, ANALYZES_DIR)

rule scores_plots:
    input:
        rules.multiple_scores_baseline_analysis_R.output,
        rules.multiple_scores_cox_lasso_radiomics_all_R.output,
        rules.multiple_scores_cox_lasso_radiomics_features_hclust_corr_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_all_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_features_hclust_corr_R.output,
        rules.multiple_scores_rsf.output,
        rules.multiple_scores_rsf_features_hclust_corr.output,
    output:
        ANALYZES_DIR + "plots/multiple_scores_cindex.svg",
        ANALYZES_DIR + "plots/multiple_scores_harrell_cindex_all.svg",
        ANALYZES_DIR + "plots/multiple_scores_harrell_cindex_features_hclust_corr.svg",
        ANALYZES_DIR + "plots/multiple_scores_ipcw_cindex_all.svg",
        ANALYZES_DIR + "plots/multiple_scores_ipcw_cindex_features_hclust_corr.svg",
    threads:
        1
    run:
       viz.results_plots(ANALYZES_DIR, NB_ESTIM_SCORE_MODELS)

rule scores_tables:
    input:
        rules.multiple_scores_baseline_analysis_R.output,
        rules.multiple_scores_cox_lasso_radiomics_all_R.output,
        rules.multiple_scores_cox_lasso_radiomics_features_hclust_corr_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_all_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_features_hclust_corr_R.output,
        rules.multiple_scores_rsf.output,
        rules.multiple_scores_rsf_features_hclust_corr.output,
    output:
        ANALYZES_DIR + f"tables/multiple_scores_{NB_ESTIM_SCORE_MODELS}_runs.tex",
    threads:
        1
    run:
       viz.latex_tables(ANALYZES_DIR, NB_ESTIM_SCORE_MODELS)

