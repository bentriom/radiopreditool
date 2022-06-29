
suppressPackageStartupMessages(library("stringr", quietly = TRUE))

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
    regex_removed <- "X[0-9]{3,4}_original_glcm_Sum(Average|Squares)"
    idx_removed <- grep(regex_removed, features)
    if (length(idx_removed) > 0) {
        return (features[-idx_removed])
    }
    return (features)
}

# Pretty label names
pretty.label <- function(label) {
    pattern_dosiomics = "X([0-9]{3,4})_[a-z]+_[a-z]+_(\\w+)"
    pattern_dosesvol = "dv_((D|V)[0-9]{1,3})_([0-9]{3,4})"
    if (str_detect(label, pattern_dosiomics)) {
        matches = str_match(label, pattern_dosiomics)
        paste(matches[2], matches[3])
    } else if (str_detect(label, pattern_dosesvol)) {
        matches = str_match(label, pattern_dosesvol)
        paste(matches[4], matches[2])
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

