## Visualisation

rule pca_visualisation:
    input:
        ANALYZES_DIR + "datasets/dataset.csv.gz"
    output:
        ANALYZES_DIR + "pca_viz.log",
        ANALYZES_DIR + "pca/pca_radiomics_all.png"
    run:
        trainset.pca_viz(ANALYZES_DIR + "datasets/dataset.csv.gz", EVENT_COL, ANALYZES_DIR)

rule scores_plots_heart:
    input:
        rules.multiple_scores_baseline_analysis_heart_R.output,
        rules.multiple_scores_cox_lasso_radiomics_all_heart_R.output,
        rules.multiple_scores_cox_lasso_radiomics_features_hclust_corr_heart_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_all_heart_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_features_hclust_corr_heart_R.output,
        rules.multiple_scores_rsf_heart.output,
        rules.multiple_scores_rsf_features_hclust_corr_heart.output,
    output:
        expand(ANALYZES_DIR + "plots/heart/multiple_scores_cindex.{format}", format = ["png", "svg"]),
        expand(ANALYZES_DIR + "plots/heart/multiple_scores_ibs.{format}", format = ["png", "svg"]),
    threads:
        1
    run:
       viz.results_plots_heart(ANALYZES_DIR, NB_ESTIM_SCORE_MODELS)

rule scores_plots_heart_two:
    input:
        rules.multiple_scores_baseline_analysis_heart_R.output,
        rules.multiple_scores_cox_lasso_radiomics_all_heart_R.output,
        rules.multiple_scores_cox_lasso_radiomics_features_hclust_corr_heart_R.output,
        rules.multiple_scores_cox_sis_radiomics_all_heart_R.output,
        rules.multiple_scores_rsf_heart.output,
        rules.multiple_scores_rsf_features_hclust_corr_heart.output,
    output:
        expand(ANALYZES_DIR + "plots/heart/multiple_scores_cindex.{format}", format = ["png", "svg"]),
        expand(ANALYZES_DIR + "plots/heart/multiple_scores_ibs.{format}", format = ["png", "svg"]),
    threads:
        1
    run:
       viz.results_plots_heart_2(ANALYZES_DIR, NB_ESTIM_SCORE_MODELS)

rule scores_tables_heart:
    input:
        rules.multiple_scores_baseline_analysis_heart_R.output,
        rules.multiple_scores_cox_lasso_radiomics_all_heart_R.output,
        rules.multiple_scores_cox_lasso_radiomics_features_hclust_corr_heart_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_all_heart_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_features_hclust_corr_heart_R.output,
        rules.multiple_scores_rsf_heart.output,
        rules.multiple_scores_rsf_features_hclust_corr_heart.output,
    output:
        ANALYZES_DIR + f"tables/heart/multiple_scores_{NB_ESTIM_SCORE_MODELS}_runs.tex",
    threads:
        1
    run:
       viz.latex_tables_heart(ANALYZES_DIR, NB_ESTIM_SCORE_MODELS)

rule scores_plots_thorax:
    input:
        rules.multiple_scores_baseline_analysis_breasts_R.output,
        rules.multiple_scores_cox_lasso_radiomics_all_thorax_R.output,
        rules.multiple_scores_cox_lasso_radiomics_features_hclust_corr_thorax_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_all_thorax_R.output,
        rules.multiple_scores_cox_bootstrap_lasso_radiomics_features_hclust_corr_thorax_R.output,
        rules.multiple_scores_cox_sis_radiomics_all_thorax_R.output,
        rules.multiple_scores_rsf_thorax.output,
        rules.multiple_scores_rsf_features_hclust_corr_thorax.output,
    output:
        expand(ANALYZES_DIR + "plots/thorax/multiple_scores_cindex.{format}", format = ["png", "svg"]),
        expand(ANALYZES_DIR + "plots/thorax/multiple_scores_ibs.{format}", format = ["png", "svg"]),
    threads:
        1
    run:
        viz.results_plots_thorax(ANALYZES_DIR, NB_ESTIM_SCORE_MODELS)

