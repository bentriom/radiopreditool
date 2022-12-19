#!/bin/bash

NJOBS=5
NTHREADS=1
TIME="24:00:00"
PARTITION="cpu_long"
MEMORY_PER_NODE="10G"
MODEL_NAME="pathol_cardiaque"
DRYRUN=""

## Rules
# VIMP rules
VIMP_RULES="rsf_vimp"
# Cox Lasso rules
COX_RULES_ALL="multiple_scores_baseline_analysis_R multiple_scores_cox_lasso_radiomics_all_R"
COX_RULES_FEATURES_HCLUST_CORR="multiple_scores_cox_lasso_radiomics_features_hclust_corr_R"
COX_RULES="${COX_RULES_ALL} ${COX_RULES_FEATURES_HCLUST_CORR}"
# RSF rules
RSF_RULES_ALL="multiple_scores_rsf"
RSF_RULES_FEATURES_HCLUST_CORR="multiple_scores_rsf_features_hclust_corr"
RSF_RULES="${RSF_RULES_ALL} ${RSF_RULES_FEATURES_HCLUST_CORR}"
# Bootstrap Cox Lasso rules
# COX_BLASSO_RULES_ALL="cox_bootstrap_lasso_radiomics_whole_heart_all_R \
# cox_bootstrap_lasso_radiomics_subparts_heart_all_R"
# COX_BLASSO_RULES_FEATURES_HCLUST_CORR="cox_bootstrap_lasso_radiomics_whole_heart_features_hclust_corr_R \
# cox_bootstrap_lasso_radiomics_subparts_heart_features_hclust_corr_R"
COX_BLASSO_RULES_ALL="multiple_scores_cox_bootstrap_lasso_radiomics_all_R"
COX_BLASSO_RULES_FEATURES_HCLUST_CORR="multiple_scores_cox_bootstrap_lasso_radiomics_features_hclust_corr_R"
COX_BLASSO_RULES="${COX_BLASSO_RULES_ALL} ${COX_BLASSO_RULES_FEATURES_HCLUST_CORR}"
# Gathering rules
RULES_ALL="${COX_RULES_ALL} ${RSF_RULES_ALL} ${COX_BLASSO_RULES_ALL}"
RULES_FEATURES_HCLUST_CORR="${COX_RULES_FEATURES_HCLUST_CORR} ${RSF_RULES_FEATURES_HCLUST_CORR} ${COX_BLASSO_RULES_FEATURES_HCLUST_CORR}"
RULES="${RULES_ALL} ${RULES_FEATURES_HCLUST_CORR}"
TARGET_RULES="$RULES"
# TARGET_RULES="$RULES_ALL"
# TARGET_RULES="${COX_BLASSO_RULES}"

ARGS="$@"
POSITIONAL=()

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --dryrun)
            DRYRUN="-n"
            TIME="01:00:00"
            PARTITION="cpu_short"
            shift # past argument
            ;;
        -j|--jobs)
            NJOBS="$2"
            shift # past argument
            shift # past value
            ;;
        -t|--cpus-per-task|--threads)
            NTHREADS="$2"
            shift # past argument
            shift # past value
            ;;
        -T|--time)
            if [ "$DRYRUN" == "" ]
            then
                TIME="$2"
            fi
            shift # past argument
            shift # past value
            ;;
        --model)
            MODEL_NAME="$2"
            shift # past argument
            shift # past value
            ;;
        --rules)
            TARGET_RULES="$2"
            shift # past argument
            shift # past value
            ;;
        -p|--partition)
            if [ "$DRYRUN" == "" ]
            then
                PARTITION="$2"
            fi
            shift # past argument
            shift # past value
            ;;
        -m|--mem)
            MEMORY_PER_NODE="$2"
            shift # past argument
            shift # past value
            ;;
        *)    # unknown option
            POSITIONAL+=("$1") # save it in an array for later
            shift # past argument
            ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

#declare -A LIST_CONFIG_FILE=( ["pathol_cardiaque"]="slurm_pathol_cardiaque.yaml" ["pathol_cardiaque_chimio"]="slurm_pathol_cardiaque_chimio.yaml" ["pathol_cardiaque_drugs"]="slurm_pathol_cardiaque_drugs.yaml" ["pathol_cardiaque_drugs_iccc_other"]="slurm_pathol_cardiaque_drugs_iccc_other.yaml" ["pathol_cardiaque_grade3_chimio"]="slurm_pathol_cardiaque_grade3_chimio.yaml" ["pathol_cardiaque_grade3_drugs"]="slurm_pathol_cardiaque_grade3_drugs.yaml" ["pathol_cardiaque_grade3_drugs_iccc_other"]="slurm_pathol_cardiaque_grade3_drugs_iccc_other.yaml" ["pathol_cardiaque_grade3_drugs_iccc_other_bw_0.1"]="slurm_pathol_cardiaque_grade3_drugs_iccc_other_bw_0.1.yaml" ["pathol_cardiaque_grade3_drugs_iccc_other_bw_0.5"]="slurm_pathol_cardiaque_grade3_drugs_iccc_other_bw_0.5.yaml" )

SNAKEMAKE_CONFIG_FILE="config/slurm/${MODEL_NAME}.yaml"
SNAKEMAKE_SBATCH="'sbatch --partition=cpu_long --mem=50G --ntasks=1 --cpus-per-task={threads} --time=16:00:00 --output=log/${MODEL_NAME}/%x-%j.out'"
#THREADS_ARGS="--set-threads rsf_features_hclust_corr_analysis=20 rsf_analysis=20"
THREADS_ARGS=""

ANALYZES_DIR="/workdir/bentrioum/results/radiopreditool/radiomics/analyzes/${MODEL_NAME}/"

COMMANDS_JOB="
module purge
module load anaconda3/2021.05/gcc-9.2.0
echo ${ARGS}
source activate radiopreditool
set -x
cd ~/opt/radiopreditool/dose_matrix/radiomics/
mkdir -p log/${MODEL_NAME}/
rm log/${MODEL_NAME}/*.out
echo $(date)
export OMP_PLACES=cores
echo ${MODEL_NAME}
snakemake --use-conda --configfile ${SNAKEMAKE_CONFIG_FILE} \
--cluster ${SNAKEMAKE_SBATCH} ${THREADS_ARGS} --jobs $NJOBS \
${DRYRUN} --rerun-incomplete \
${TARGET_RULES}
"

sbatch \
    --job-name=learning_${NJOBS}_${MODEL_NAME} --output=./out/learning_${MODEL_NAME}_${PARTITION}_${NJOBS} --time=$TIME \
    --ntasks=1 \
    --partition=$PARTITION \
    --mem=$MEMORY_PER_NODE --wrap="$COMMANDS_JOB"
 
