#!/bin/bash

NJOBS=200
NTHREADS=1
TIME="20:00:00"
PARTITION="cpu_long"
MEMORY_PER_NODE="175G"
MODEL_NAME="pathol_cardiaque_grade3_drugs_iccc_other_bw_0.5"
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

declare -A LIST_CONFIG_FILE=( ["pathol_cardiaque"]="pathol_cardiaque.yaml" ["pathol_cardiaque_chimio"]="pathol_cardiaque_chimio.yaml" ["pathol_cardiaque_drugs"]="pathol_cardiaque_drugs.yaml" ["pathol_cardiaque_grade3_chimio"]="pathol_cardiaque_grade3_chimio.yaml" ["pathol_cardiaque_grade3_drugs_iccc_other_bw_0.1"]="pathol_cardiaque_grade3_drugs_iccc_other_bw_0.1.yaml" ["pathol_cardiaque_grade3_drugs_iccc_other_bw_0.5"]="pathol_cardiaque_grade3_drugs_iccc_other_bw_0.5.yaml" )

SNAKEMAKE_CONFIG_FILE="config/slurm/${LIST_CONFIG_FILE[${MODEL_NAME}]}"
SNAKEMAKE_NBATCHES=50
SNAKEMAKE_SBATCH="\"sbatch --partition=cpu_med --mem=15G --ntasks=1 --time=00:30:00 --output=/dev/null\""

COMMANDS_JOB="
module purge
module load anaconda3/2021.05/gcc-9.2.0
source activate radiopreditool
cd ~/opt/radiopreditool/dose_matrix/radiomics/
export OMP_PLACES=cores
echo $(date)
set -x
"
for i in $(seq 1 $SNAKEMAKE_NBATCHES);
do
COMMANDS_JOB="${COMMANDS_JOB}
echo \"Batch $i/${SNAKEMAKE_NBATCHES}\"
echo $(date)
snakemake --configfile ${SNAKEMAKE_CONFIG_FILE} --cluster ${SNAKEMAKE_SBATCH} --jobs ${NJOBS} --batch gather_radiomics=$i/${SNAKEMAKE_NBATCHES} \
gather_radiomics
"
done


sbatch \
    --job-name=radiomics_batch_${NJOBS} --output=./out/extract_radiomics_batch_snake_submits_${NJOBS} --time=$TIME \
    --ntasks=1 \
    --partition=$PARTITION \
    --mem=$MEMORY_PER_NODE --wrap="$COMMANDS_JOB"
 
