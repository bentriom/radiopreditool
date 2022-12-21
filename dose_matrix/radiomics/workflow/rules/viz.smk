## Visualisation

rule pca_visualisation:
    input:
        ANALYZES_DIR + "datasets/dataset.csv.gz"
    output:
        ANALYZES_DIR + "pca_viz.log",
        ANALYZES_DIR + "pca/pca_radiomics_all.png"
    run:
        trainset.pca_viz(ANALYZES_DIR + "datasets/dataset.csv.gz", EVENT_COL, ANALYZES_DIR)

rule scores_plot:
    input:
        rules.multiple_scores_baseline_analysis_R.output
        rules.multiple_scores_cox_lasso_radiomics_all_R.output
        rules.multiple_scores_cox_lasso_radiomics_features_hclust_corr_R.output
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_all_R.output
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_features_hclust_corr_R.output
        rules.multiple_scores_rsf.output
        rules.multiple_scores_rsf_features_hclust_corr.output
    output:
        ANALYZES_DIR + "plots/multiple"

