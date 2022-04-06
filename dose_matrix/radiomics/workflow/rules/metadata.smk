
## Metadata
rule list_newdosi_files:
    input:
        expand(DOSES_DATASET_DIR + "{newdosi_file}.csv.gz", newdosi_file = list_newdosi_files) 
    output:
        METADATA_DIR + "list_newdosi_files.csv"
    shell:
        "./workflow/scripts/awk_list_newdosi_files.sh '" + DOSES_DATASET_DIR + "' > " + METADATA_DIR + "list_newdosi_files.csv"

rule list_newdosi_checks:
    input:
        METADATA_DIR + "list_newdosi_files.csv"
    output:
        METADATA_DIR + "list_newdosi_checks.csv"
    run:
        check_dataset.analyze_dataset(DOSES_DATASET_DIR, METADATA_DIR)

rule report_checks:
    input:
        METADATA_DIR + "list_newdosi_checks.csv"
    output:
        METADATA_DIR + "report_checks.txt"
    run:
        report_checks.print_report(METADATA_DIR)

