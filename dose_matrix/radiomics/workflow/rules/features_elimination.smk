
## Feature elimination

rule feature_elimination_dataset_hclust_corr:
    input:
        ANALYZES_DIR + "datasets/trainset.csv.gz"
    output: 
        ANALYZES_DIR + "screening/features_hclust_corr.csv",
        ANALYZES_DIR + "feature_elimination_hclust_corr.log"
    run:
        trainset.feature_elimination_hclust_corr(ANALYZES_DIR + "datasets/trainset.csv.gz", EVENT_COL, ANALYZES_DIR)

