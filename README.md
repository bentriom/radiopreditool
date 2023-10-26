# RadioPrediTool

This repository contains the analyses of the late radio-induced pathologies of the French Childhood Cancer Survivor 
Study (FCCSS) developped during my post-doc.

 ## viz/
 This directory contains a script that make visualization of the FCCSS clinical dataset (aggregated database) 
 using Plotly/Dash.

 ## dose\_matrix/radiomics

This directory contains a Snakemake pipeline that gather:
- the conversion of CSV newdosi files to images
- the extraction of dosiomics
- the predictive models based on dosiomics

### Run

The tasks are run from a Snakemake pipeline. The file `workflow/envs/radiopreditool_env.yaml` describes a python 
environment that contains the dependencies to run the pipeline. It can be installed and activated through:

```
conda env create -f workflow/envs/radiopreditool_env.yaml
conda activate radiopreditool
```

Examples of bash scripts that run the pipeline can be found in `run/`. Scripts related to the mesocentre can be found in 
`run/slurm/`, whereas the other directories are for local runs.

### Configuration file

The extraction/analyzes are based on a configuration file, some examples are locaed in the directory config.
This file set important configuration variables for a proper execution of the Snakemake pipeline. Here is a description 
of the variables:

- `RESULTS_DIR`: the directory where the results of the pipeline will be saved
- `MODEL_NAME`: the user-defined model name of the analysis. The predictive analyses will be stored in 
`[RESULTS_DIR]/analyzes/[MODEL_NAME]`.
- `RADIOMICS_NAME`: the user-defined name of directory where the extracted dosiomics will be stored. Extracted dosiomics 
will be saved in the directory `[RESULTS_DIR]/extraction/[RADIOMICS_NAME]`.
- `DOSES_DATASET_DIR`: The parent directory of the dataset containing the CSV newdosi files.
- `DOSES_DATASET_SUBDIRS`: the subdirectories of the dataset. Each directory corresponds to an institution.
- `RADIOMICS_PARAMS_FILE`: a file that contains the settings for pyradiomics.
- `LABELS_T_ORGANS_FILE`: a file that associates each numeric "T" label to a suborgan.
- `NAME_SUPER_T_FUNC`: the name of the function in `workflow/scripts/radiopreditool_utils.py` to use when creating 
a super set of "T" labels. Example: each label in 320:324 is associated to the super T label 1320.
- `LABELS_SUPER_T_VOI`: The super "T" labels where dosiomics should be extracted from.
- `LABELS_SUPER_T_VOI`: The T labels where dosiomics should be extracted from.
- `FCCSS_CLINICAL_DATASET`: The FCCSS aggregated clinical database ("base assemblée")
- `FCCSS_CLINICAL_VARIABLES`: the clinical variables that should be added to the covariates of the predictive models.
- `EVENT_COL`: the name of the variable that represents the event column in the FCCSS aggregated clinical dataset.
- `DATE_EVENT_COL`: the name of the variable that represents the date of event column (if any) 
in the FCCSS aggregated clinical dataset.
- `NB_ESTIM_SCORE_MODELS`: the number of folds in the predictive models' cross-validation.


