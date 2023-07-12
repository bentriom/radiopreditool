
## Metadata

rule list_newdosi_files:
    input:
        expand(DOSES_DATASET_DIR + "{newdosi_file}.csv.gz", newdosi_file = list_newdosi_files)
    output:
        METADATA_DIR + "list_newdosi_files.csv"
    shell:
        "./workflow/scripts/awk_list_newdosi_files.sh '" + DOSES_DATASET_DIR + \
        "' > " + METADATA_DIR + "list_newdosi_files.csv"

rule list_newdosi_checks:
    input:
        METADATA_DIR + "list_newdosi_files.csv"
    output:
        METADATA_DIR + "list_newdosi_checks.csv",
        METADATA_DIR + "biggest_image_size.csv"
    run:
        check_dataset.analyze_dataset(DOSES_DATASET_DIR, METADATA_DIR)

rule report_checks:
    input:
        METADATA_DIR + "list_newdosi_checks.csv",
        LABELS_T_ORGANS_FILE
    output:
        METADATA_DIR + "report_checks.txt",
        expand(METADATA_DIR + "fccss_missing_labels_t_{name}.csv", name = ["all", "women", "men"]),
    run:
        report_checks.print_report(METADATA_DIR, FCCSS_CLINICAL_DATASET, LABELS_T_ORGANS_FILE)

rule entropy_analysis:
    input:
        expand(NII_DIR + "{newdosi_patient}_ID2013A.nii.gz", newdosi_patient = list_newdosi_patients),
        expand(NII_DIR + "{newdosi_patient}_mask_t.nii.gz", newdosi_patient = list_newdosi_patients),
        expand(NII_DIR + "{newdosi_patient}_mask_super_t.nii.gz", newdosi_patient = list_newdosi_patients)
    output:
        METADATA_DIR + "entropy_newdosi.csv.gz"
    threads:
        get_ncpus()
    run:
        entropy_analysis.compute_entropy(DOSES_DATASET_SUBDIRS, NII_DIR, METADATA_DIR)

