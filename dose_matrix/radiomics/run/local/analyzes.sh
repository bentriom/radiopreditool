
conda activate radiopreditool
RULES="multiple_scores_cox_lasso_radiomics_all_R \
multiple_scores_cox_lasso_radiomics_features_hclust_corr_R \
multiple_scores_rsf \
multiple_scores_rsf_features_hclust_corr \
multiple_scores_baseline_analysis_R
"
LASSO_RULES="cox_bootstrap_lasso_radiomics_whole_heart_all_R \
cox_bootstrap_lasso_radiomics_whole_heart_features_hclust_corr_R \
cox_bootstrap_lasso_radiomics_subparts_heart_all_R \
cox_bootstrap_lasso_radiomics_subparts_heart_features_hclust_corr_R"

export LOCAL_SNAKEMAKE_NCPUS=8

#snakemake --rerun-incomplete --use-conda --configfile config/local/pathol_cardiaque_grade3_drugs_iccc_other_bw_0.5.yaml -c${LOCAL_SNAKEMAKE_NCPUS} ${LASSO_RULES}
snakemake --keep-incomplete --use-conda --configfile config/local/pathol_cardiaque_grade3_drugs_iccc_other_bw_0.5.yaml -c${LOCAL_SNAKEMAKE_NCPUS} ${LASSO_RULES}

