
suppressPackageStartupMessages(library("stringr", quietly = TRUE))
suppressPackageStartupMessages(library("parallel", quietly = TRUE))

# Get clinical variables from all features
get.clinical_features <- function(columns, event_col, duration_col) {
    regex_non_clinical <- paste("^((X[0-9]{3,4}_)|(dv_)|(",event_col,")|(",duration_col,")|(ctr)|(numcent)|(has_radiomics))", sep = "")
    idx_non_clinical_vars <- grep(regex_non_clinical, columns)
    if (length(idx_non_clinical_vars) > 0) {
        return (columns[-idx_non_clinical_vars])
    }
    return (columns)
}

# Eliminate specific gray level image features
filter.gl <- function(features) {
    regex_removed <- "^X[0-9]{3,4}_original_((glcm_Sum(Average|Squares))|(shape_))"
    idx_removed <- grep(regex_removed, features)
    if (length(idx_removed) > 0) {
        return (features[-idx_removed])
    }
    return (features)
}

# Get available ncpus
get.ncpus <- function() {
    slurm_ntasks <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
    local_ntasks <- as.numeric(Sys.getenv("LOCAL_SNAKEMAKE_NCPUS"))
    if (is.na(slurm_ntasks)) {
        if (is.na(local_ntasks)) {
            ncpus <- parallel::detectCores()
        } else {
            ncpus <- local_ntasks
        }
    } else {
        ncpus <- slurm_ntasks
    }
    ncpus
}

get.nworkers <- function() get.ncpus()-1

# Pretty label names
pretty.label <- function(label) {
    pattern_dosiomics = "X([0-9]{3,4})_[a-z]+_[a-z]+_(\\w+)"
    pattern_dosesvol = "dv_((D|V)[0-9]{1,3})_(1320)"
    pattern_iccc = "iccc_([0-9]|nan)"
    if (str_detect(label, pattern_dosiomics)) {
        matches = str_match(label, pattern_dosiomics)
        matches[3]
    } else if (str_detect(label, pattern_dosesvol)) {
        matches = str_match(label, pattern_dosesvol)
        paste(matches[4], matches[2])
    } else if (str_detect(label, pattern_iccc)) {
        pretty.iccc(label)
    } else { label }
}

pretty.labels <- function(labels) {
    sapply(labels, pretty.label)
}

# Automatically create a survival formula
get.surv.formula <- function(event_col, covariates, duration_col = "survival_time_years") {
    str.surv_formula <- paste("Surv(", duration_col, ",", event_col, ") ~ ", sep = '')
    for (var in covariates) {
        str.surv_formula <- paste(str.surv_formula, var, " + ", sep = '')
    }
    str.surv_formula <- substr(str.surv_formula, 1, nchar(str.surv_formula) - 2)
    as.formula(str.surv_formula)
}

get.ipcw.surv.formula <- function(event_col, covariates, duration_col = "survival_time_years") {
    clinical_vars <- get.clinical_features(covariates, event_col, duration_col)
    regex_removed <- "^iccc_"
    idx_removed <- grep(regex_removed, clinical_vars)
    if (length(idx_removed) > 0)
        get.surv.formula(event_col, clinical_vars[-idx_removed])
    else
        get.surv.formula(event_col, clinical_vars)
}

# Get the proportion of events in data
event_prop <- function(fccss.data, event_col) {
    return(sum(fccss.data[[event_col]] == 1) / nrow(fccss.data))
}

# Labels iccc
pretty.iccc <- function(label_iccc) {
    iccc_labels = list(
                       'iccc_10.d'= 'Gonadal carcinomas',
                       'iccc_11.b'= 'Thyroid carcinomas',
#                       'iccc_06.a'= 'Nephroblastoma and other nonepithelial renal tumors',
                       'iccc_06.a'= 'Nephroblastoma (06.a)',
                       'iccc_08.a'= 'Osteosarcomas',
#                       'iccc_04.a'= 'Neuroblastoma and ganglioneuroblastoma',
                       'iccc_04.a'= 'Neuroblastoma/ganglioneuroblastoma',
                       'iccc_09.a'= 'Rhabdomyosarcomas',
                       'iccc_10.c'= 'Malignant gonadal germ cell tumors',
#                       'iccc_02.b'= 'Non-Hodgkin lymphomas (except Burkitt lymphoma)',
                       'iccc_02.b'= 'Non-Hodgkin lymphomas (02.b)',
                       'iccc_02.e'= 'Unspecified lymphomas',
                       'iccc_05.a'= 'Retinoblastoma',
                       'iccc_03.b'= 'Astrocytomas',
                       'iccc_09.e'= 'Unspecified soft tissue sarcomas',
                       'iccc_11.d'= 'Malignant melanomas',
#                       'iccc_09.b'= 'Fibrosarcomas, peripheral nerve sheath tumors, and other fibrous neoplasms',
                       'iccc_09.b'= 'Soft-tissue tumors (09.b)',
#                       'iccc_03.c'= 'Intracranial and intraspinal embryonal tumors',
                       'iccc_03.c'= 'Intracranial/intraspinal embryonal tumors',
                       'iccc_12.b'= 'Other unspecified malignant tumors',
                       'iccc_09.d'= 'Other specified soft tissue sarcomas',
#                       'iccc_10.a'= 'Intracranial and intraspinal germ cell tumors',
                       'iccc_10.a'= 'Intracranial/intraspinal germ cell tumors',
#                       'iccc_03.f'= 'Unspecified intracranial and intraspinal neoplasms',
                       'iccc_03.f'= 'Unspecified intracranial/intraspinal neoplasms',
                       'iccc_11.e'= 'Skin carcinomas',
                       'iccc_08.c'= 'Ewing tumor and related sarcomas of bone',
                       'iccc_06.b'= 'Renal carcinomas',
                       'iccc_02.a'= 'Hodgkin lymphomas',
#                       'iccc_03.a'= 'Ependymomas and choroid plexus tumor',
                       'iccc_03.a'= 'Ependymomas/choroid plexus tumor',
                       'iccc_03.e'= 'Other specified intracranial/intraspinal neoplasms',
#                       'iccc_03.e'= 'Other specified intracranial and intraspinal neoplasms',
                       'iccc_11.f'= 'Other and unspecified carcinomas',
                       'iccc_11.a'= 'Adrenocortical carcinomas',
                       'iccc_10.e'= 'Other and unspecified malignant gonadal tumors',
                       'iccc_02.d'= 'Miscellaneous lymphoreticular neoplasms',
                       'iccc_08.b'= 'Chondrosarcomas',
                       'iccc_08.d'= 'Other specified malignant bone tumors',
                       'iccc_03.d'= 'Other gliomas',
#                       'iccc_10.b'= 'Malignant extracranial and extragonadal germ cell tumors',
                       'iccc_10.b'= 'Malignant extracranial/extragonadal germ cell tumors',
                       'iccc_07.b'= 'Hepatic carcinomas',
                       'iccc_11.c'= 'Nasopharyngeal carcinomas',
                       'iccc_07.a'= 'Hepatoblastoma',
                       'iccc_04.b'= 'Other peripheral nervous cell tumors',
                       'iccc_12.a'= 'Other specified malignant tumors',
                       'iccc_02.c'= 'Burkitt lymphoma',
                       'iccc_08.e'= 'Unspecified malignant bone tumors',
                       'iccc_09.c'= 'Kaposi sarcoma',
                       'iccc_01.a'= 'Lymphoid leukemias',
                       'iccc_00.0'= 'Unknown',
                       'iccc_nan'= 'Unknown'
    )
    iccc_labels[[label_iccc]]
}

