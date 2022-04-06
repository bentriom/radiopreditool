## Visualisation

rule pca_visualisation:
    input:
        ANALYZES_DIR + "datasets/dataset.csv.gz"
    output:
        ANALYZES_DIR + "pca_viz.log",
        ANALYZES_DIR + "pca/pca_radiomics_all.png"
    run:
        trainset.pca_viz(ANALYZES_DIR + "datasets/dataset.csv.gz", EVENT_COL, ANALYZES_DIR)

