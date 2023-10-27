
conda activate radiopreditool
export LOCAL_SNAKEMAKE_NCPUS=4

# snakemake -n --keep-incomplete --configfile config/local_mac/pathol_cardiaque_grade3_drugs_iccc_other_bw_0.5.yaml -c${LOCAL_SNAKEMAKE_NCPUS} scores_plots_heart_two
snakemake --rerun-incomplete --configfile config/local_mac/breast_cancer_sim_empty_bw_0.5.yaml -c${LOCAL_SNAKEMAKE_NCPUS} scores_plots_thorax

