
## Feature elimination

rule feature_elimination_dataset_hclust_corr:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz"
    output:
        ANALYZES_DIR + "screening/features_hclust_corr.csv",
    run:
        trainset.feature_elimination_hclust_corr(ANALYZES_DIR + "datasets/trainset.csv.gz", EVENT_COL, ANALYZES_DIR)

rule feature_elimination_dataset_hclust_corr_sets:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        expand(ANALYZES_DIR + "screening/features_hclust_corr_{nb_set}.csv", nb_set = range(NB_ESTIM_SCORE_MODELS))
    run:
        for nb_set in range(NB_ESTIM_SCORE_MODELS):
            trainset.feature_elimination_hclust_corr(ANALYZES_DIR + f"datasets/trainset_{nb_set}.csv.gz",
                                                     EVENT_COL, ANALYZES_DIR)

