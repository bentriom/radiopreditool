
## Feature elimination

rule feature_elimination_dataset_hclust_corr:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz"
    output:
        ANALYZES_DIR + "screening/features_hclust_corr.csv",
    run:
        trainset.feature_elimination_hclust_corr(ANALYZES_DIR + "datasets/trainset.csv.gz", EVENT_COL, ANALYZES_DIR)

rule feature_elimination_dataset_hclust_corr_nb_set:
    input:
        ANALYZES_DIR + f"datasets/trainset_{nb_set}.csv.gz"
    output:
        ANALYZES_DIR + "screening/features_hclust_corr_{nb_set}.csv",
    run:
        trainset.feature_elimination_hclust_corr(ANALYZES_DIR + f"datasets/trainset_{nb_set}.csv.gz",
                                                 EVENT_COL, ANALYZES_DIR)

