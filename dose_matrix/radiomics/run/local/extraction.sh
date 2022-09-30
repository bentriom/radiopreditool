
conda activate radiopreditool
export LOCAL_SNAKEMAKE_NCPUS=5

snakemake --rerun-incomplete --use-conda --configfile config/local/pathol_cardiaque_grade3_drugs_iccc_other_bw_1.0.yaml -c${LOCAL_SNAKEMAKE_NCPUS} gather_radiomics

