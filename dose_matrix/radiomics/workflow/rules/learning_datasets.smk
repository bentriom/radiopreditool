
# Learning datasets
rule create_dataset:
    input:
        RADIOMICS_DIR + "dose_matrix_radiomics.csv.gz",
        FCCSS_CLINICAL_DATASET
    output:
        ANALYZES_DIR + "datasets/dataset.csv.gz",
    run:
        file_radiomics = RADIOMICS_DIR + "dose_matrix_radiomics.csv.gz"
        trainset.create_dataset(file_radiomics, FCCSS_CLINICAL_DATASET, ANALYZES_DIR, FCCSS_CLINICAL_VARIABLES, EVENT_COL, DATE_EVENT_COL)

rule split_dataset:
    input:
        ANALYZES_DIR + "datasets/dataset.csv.gz",
        FCCSS_CLINICAL_DATASET
    output:
        ANALYZES_DIR + "datasets/trainset.csv.gz",
        ANALYZES_DIR + "datasets/testset.csv.gz"
    run:
        file_radiomics = RADIOMICS_DIR + "dose_matrix_radiomics.csv.gz"
        trainset.split_dataset(file_radiomics, FCCSS_CLINICAL_DATASET, ANALYZES_DIR, FCCSS_CLINICAL_VARIABLES, EVENT_COL, DATE_EVENT_COL)

rule multiple_splits_dataset:
    input:
        ANALYZES_DIR + "datasets/dataset.csv.gz",
        FCCSS_CLINICAL_DATASET
    output:
        expand(ANALYZES_DIR + "datasets/trainset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS)),
        expand(ANALYZES_DIR + "datasets/testset_{nb_set}.csv.gz", nb_set = range(NB_ESTIM_SCORE_MODELS))
    run:
        file_radiomics = RADIOMICS_DIR + "dose_matrix_radiomics.csv.gz"
        for i in range(NB_ESTIM_SCORE_MODELS):
            trainset.split_dataset(file_radiomics, FCCSS_CLINICAL_DATASET, ANALYZES_DIR, FCCSS_CLINICAL_VARIABLES, EVENT_COL, DATE_EVENT_COL, end_name_sets = f"_{i}")

