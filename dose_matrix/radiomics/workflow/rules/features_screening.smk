
## Feature elimination

rule feature_screening_dataset_hclust_corr:
    input:
        ANALYZES_DIR + "datasets/dataset.csv.gz"
    output:
        ANALYZES_DIR + "screening/features_hclust_corr.csv",
    run:
        trainset.feature_elimination_hclust_corr(EVENT_COL, ANALYZES_DIR)

rule feature_screening_dataset_hclust_corr_sets:
    input:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    output:
        expand(ANALYZES_DIR + "screening/features_hclust_corr_{nb_set}.csv", nb_set = range(NB_ESTIM_SCORE_MODELS))
    run:
        for nb_set in range(NB_ESTIM_SCORE_MODELS):
            trainset.feature_elimination_hclust_corr(EVENT_COL, ANALYZES_DIR, id_set = nb_set)

