#!/bin/bash

NJOBS=45
NTHREADS=1
TIME="04:00:00"
PARTITION="cpu_med"
MEMORY_PER_NODE="50G"
MODEL_NAME="pathol_cardiaque"
POSITIONAL=()

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
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
            TIME="$2"
            shift # past argument
            shift # past value
            ;;
        --model)
            MODEL_NAME="$2"
            shift # past argument
            shift # past value
            ;;
        -p|--partition)
            PARTITION="$2"
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

declare -A LIST_CONFIG_FILE=( ["pathol_cardiaque"]="config/slurm/slurm_pathol_cardiaque.yaml" ["pathol_cardiaque_chimio"]="config/slurm/slurm_pathol_cardiaque_chimio.yaml" ["pathol_cardiaque_drugs"]="config/slurm/slurm_pathol_cardiaque_drugs.yaml" )

SNAKEMAKE_CONFIG_FILE=${LIST_CONFIG_FILE[${MODEL_NAME}]}
SNAKEMAKE_SBATCH="'sbatch --partition=cpu_long --mem=175G --ntasks=1 --cpus-per-task={threads} --time=16:00:00 --output=log/${MODEL_NAME}/%x-%j.out'"
THREADS_ARGS="--set-threads list_newdosi_checks=40 --set-resources list_newdosi_checks:partition=mem list_newdosi_checks:mem=800G"
#TARGET_RULES="/workdir/bentrioum/results/radiopreditool/radiomics/metadata/report_checks.txt"
TARGET_RULES="entropy_analysis"

COMMANDS_JOB="
module purge
module load anaconda3/2021.05/gcc-9.2.0
source activate radiopreditool
set -x
cd ~/opt/radiopreditool/dose_matrix/radiomics/
mkdir -p log/${MODEL_NAME}/
rm log/${MODEL_NAME}/*.out
echo $(date)
export OMP_PLACES=cores
snakemake --use-conda --configfile ${SNAKEMAKE_CONFIG_FILE} \
--cluster ${SNAKEMAKE_SBATCH} ${THREADS_ARGS} --jobs $NJOBS \
${TARGET_RULES}"

sbatch \
    --job-name=check_data_${NJOBS} --output=./out/check_data_${PARTITION}_${NJOBS} --time=$TIME \
    --ntasks=1 \
    --partition=$PARTITION \
    --mem=$MEMORY_PER_NODE --wrap="$COMMANDS_JOB"

