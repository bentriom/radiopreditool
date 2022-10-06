
conda activate radiopreditool
RULES="multiple_scores_cox_lasso_radiomics_all_R \
multiple_scores_cox_lasso_radiomics_features_hclust_corr_R \
multiple_scores_rsf \
multiple_scores_rsf_features_hclust_corr \
multiple_scores_baseline_analysis_R
"
export LOCAL_SNAKEMAKE_NCPUS=6

snakemake -n --rerun-incomplete --use-conda --configfile config/local/pathol_cardiaque_grade3_drugs_iccc_other_bw_1.0.yaml -c${LOCAL_SNAKEMAKE_NCPUS} ${RULES}

