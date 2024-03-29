
suppressPackageStartupMessages(library("stringr", quietly = TRUE))
suppressPackageStartupMessages(library("parallel", quietly = TRUE))

# Get analyzes dir from the snakemake's config file
get.analyzes_dir_from_config <- function(config) {
    paste0(config$RESULTS_DIR, "analyzes/", config$MODEL_NAME, "/")
}

# Get clinical variables from all features
get.clinical_features <- function(columns, event_col, duration_col) {
    regex_non_clinical <- paste0("^((X[0-9]{3,5}_)|(dv_)|(",event_col,")|(",duration_col,")|",
                                 "(ctr)|(numcent)|(has_radiomics))")
    idx_non_clinical_vars <- grep(regex_non_clinical, columns)
    if (length(idx_non_clinical_vars) > 0) {
        return (columns[-idx_non_clinical_vars])
    }
    return (columns)
}

# Eliminate specific gray level image features
filter.gl <- function(features) {
    regex_removed <- "^X[0-9]{3,5}_original_((glcm_Sum(Average|Squares))|(shape_))"
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

get.nworkers <- function() get.ncpus()

get.na.coefs <- function(object) {
  if (is(object, "coxph")) {
    coefs_coxph <- coef(object)
    return(names(coefs_coxph)[is.na(coefs_coxph)])
  }
  else if (is(object, "selection.coxnet") | is(object, "bootstrap.coxnet")) {
    coefs_coxph <- coef(object$coxph.fit)
    return(names(coefs_coxph)[is.na(coefs_coxph)])
  }
  else if (is(object, "rfsrc")) {
    return(NULL)
  }
}

# Pretty label names
pretty.label <- function(label) {
    pattern_dosiomics = "X([0-9]{3,5})_[a-z]+_([a-z]+)_(\\w+)"
    pattern_dosesvol = "dv_((D|V)[0-9]{1,3})_([0-9]{3,5})"
    pattern_iccc = "iccc_([0-9]|nan)"
    if (stringr::str_detect(label, pattern_dosiomics)) {
        matches = str_match(label, pattern_dosiomics)
        type_dosio <- matches[3]
        if (type_dosio == "firstorder") type_dosio <- "" 
        pretty_label <- paste0(pretty.mask(matches[2]), " ", type_dosio, " ", matches[4], " dose")
    } else if (stringr::str_detect(label, pattern_dosesvol)) {
        matches = str_match(label, pattern_dosesvol)
        pretty_label <- paste(pretty.mask(matches[4]), matches[2])
    } else if (stringr::str_detect(label, pattern_iccc)) {
        pretty_label <- pretty.iccc(label)
    } else { pretty_label <- pretty.clinical.var(label) }
    stringr::str_trunc(pretty_label, 35)
}

pretty.labels <- function(labels) {
    sapply(labels, pretty.label)
}

# Automatically create a survival formula
get.surv.formula <- function(event_col, covariates, duration_col = "survival_time_years") {
    str.surv_formula <- paste("Surv(", duration_col, ",", event_col, ") ~ ", sep = '')
    if (length(covariates) > 0) {
      for (var in covariates) {
          str.surv_formula <- paste(str.surv_formula, var, " + ", sep = '')
      }
      str.surv_formula <- substr(str.surv_formula, 1, nchar(str.surv_formula) - 2)
    } else {
      str.surv_formula <- paste0(str.surv_formula, "1")
    }
    as.formula(str.surv_formula)
}

get.ipcw.surv.formula <- function(event_col, covariates, duration_col = "survival_time_years") {
    clinical_vars <- get.clinical_features(covariates, event_col, duration_col)
    regex_removed <- "^(iccc|do)_"
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

# Labels clinical variables
pretty.clinical.var <- function(label_var) {
  var_labels = list(
                    'ANTHRA' = 'Anthracyclines',
                    'ALKYL' = 'Alkylating agents',
                    'VINCA' = 'Vinca alkaloids'
                    )
  if (label_var %in% names(var_labels)) var_labels[[label_var]]
  else label_var
}
# Labels subdivisions (masks of the dose distribution)
pretty.mask <- function(label_mask) {
  mask_labels = list(
                     '320' = 'Myocardium',
                     '321' = 'Right atrium',
                     '322' = 'Left atrium',
                     '323' = 'Right ventricle',
                     '324' = 'Left ventricle',
                     '1320' = 'Heart'
                     )
  mask_labels[[label_mask]]
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

# Preliminary screening of features (independant, identical, unimodal features)
preliminary_filter <- function(df_dataset, covariates, event_col, 
                               screening_method = "all", id_set = "", analyzes_dir = NULL) {
  # Get screened features
  if (screening_method == "features_hclust_corr") {
    file_features <- paste0(analyzes_dir, "screening/features_hclust_corr")
    if (id_set == "") file_features <- paste0(file_features, ".csv")
    else file_features <- paste0(file_features, "_", id_set, ".csv")
    screened_features <- as.character(read.csv(file_features)[,1])
    # Add "X" for R colname compatibility
    screened_features <- as.character(lapply(screened_features, 
                                      function(x) `if`(stringr::str_detect(substr(x, 1, 1), "[0-9]"), 
                                                       paste("X", x, sep = ""), x)))
    covariates <- covariates[covariates %in% screened_features]
  }
  # Check duplicated of columns with unique values
  filter_data <- !duplicated(as.list(df_dataset[covariates])) & 
                  unlist(lapply(df_dataset[covariates], 
                                function(col) { length(unique(col)) > 1 } ))
  filtered_covariates <- names(filter_data)[filter_data]
  # Filter iccc columns with chi2/fisher test on contingency table
  irrelevant_iccc_cols <- filter_dummies_iccc(df_dataset[, c(filtered_covariates, event_col)], event_col)
  irrelevant_drugs_cols <- filter_drugs_cumsums(df_dataset[, c(filtered_covariates, event_col)], event_col)
  filtered_covariates <- filtered_covariates[!(filtered_covariates %in% c(irrelevant_iccc_cols, irrelevant_drugs_cols))]
  filtered_covariates
}

# Perfoms chi2 test to only keep dummy primary cancers variables that are dependant with any event
# Returns the iccc columns to delete
filter_dummies_iccc <- function(df_dataset, event_col) {
  stopifnot(event_col %in% colnames(df_dataset))
  iccc_cols <- grep("^iccc_\\d", colnames(df_dataset), value = T)
  irrelevant_iccc_cols <- na.omit(unlist(sapply(iccc_cols, function(iccc) {
    table_iccc <- table(df_dataset[, c(iccc, event_col)])
    if (any(table_iccc <= 10)) pval <- fisher.test(table_iccc)$p.value
    else pval <- chisq.test(table_iccc)$p.value
    # Rejected test means the two variables are dependant
    if (pval > 0.01) return (iccc)
    return (NULL)
  })))
  irrelevant_iccc_cols
}

# Performs Wilcoxon test to only keep chemotherapy cumulative doses that are dependant with any events
# Returns the chemotherapy variables to delete
filter_drugs_cumsums <- function(df_dataset, event_col) {
  stopifnot(event_col %in% colnames(df_dataset))
  drugs_cols <- grep("do_(\\w+)_cumsum", colnames(df_dataset), value = T)
  mask_no_event <- (df_dataset[[event_col]] == 0)
  irrelevant_drugs_cols <- na.omit(unlist(sapply(drugs_cols, function(drug) {
    pval <- wilcox.test(df_dataset[mask_no_event, drug], df_dataset[!mask_no_event, drug])$p.value
    # Rejected test means the two variables are dependant
    if (pval > 0.01) return (drug)
    return (NULL)
  })))
  irrelevant_drugs_cols
}

