
options(show.error.locations = TRUE, error=traceback)

suppressPackageStartupMessages({
library("yaml", quietly = TRUE)
library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("randomForestSRC", quietly = TRUE)
library("pec", quietly = TRUE)
library("Hmisc", quietly = TRUE)
library("logger", quietly = TRUE)
library("parallel", quietly = TRUE)
})

source("workflow/scripts/utils_rsf.R")

rsf_learning <- function(file_trainset, file_testset, file_features, event_col, 
                         analyzes_dir, duration_col, suffix_model, subdivision_type) {
    dir.create(paste0(analyzes_dir, "rsf/"), showWarnings = FALSE)
    nworkers <- get.nworkers()
    options(rf.cores = nworkers, mc.cores = nworkers)
    rsf_logfile <- paste0(analyzes_dir, "rsf_", suffix_model, "_", subdivision_type,".log")
    if (file.exists(rsf_logfile)) { file.remove(rsf_logfile) }
    log_appender(appender_file(rsf_logfile, append = TRUE))
    log_info("Random Survival Forest learning")
    # Dataset
    df_trainset <- read.csv(file_trainset, header = TRUE)
    df_testset <- read.csv(file_testset, header = TRUE)
    # Select subset of features due to feature elimination
    features <- `if`(file_features == "all", colnames(df_trainset), as.character(read.csv(file_features)[,1]))
    # Add "X" for R colname compatibility
    features <- as.character(lapply(features, function(x) { `if`(str_detect(substr(x, 1, 1), "[0-9]"), paste0("X", x), x) }))
    df_trainset <- df_trainset[,features]
    df_testset <- df_testset[,features]
    clinical_vars <- get.clinical_features(colnames(df_trainset), event_col, duration_col)
    log_info(paste0("Trainset file:", file_trainset, "with", nrow(df_trainset), "samples"))

    if (subdivision_type == "32X") {
        # Model 32X radiomics firstorder covariates
        log_info("Model 32X radiomics firstorder")
        model_name <- paste0("32X_radiomics_firstorder_", suffix_model)
        cols_32X <- grep("^X32[0-9]{1}_original_firstorder_", colnames(df_trainset), value = TRUE)
        covariates_32X <- c(clinical_vars, cols_32X)
        rsf.obj <- model_rsf(df_trainset, df_testset, covariates_32X, event_col, duration_col, analyzes_dir, model_name, rsf_logfile)
        # plot_vimp(rsf.obj, analyzes_dir, model_name)
    
        # Model 32X radiomics full covariates
        log_info("Model 32X radiomics full")
        model_name <- paste0("32X_radiomics_full_", suffix_model)
        cols_32X <- filter.gl(grep("^X32[0-9]{1}_original_", colnames(df_trainset), value = TRUE))
        covariates_32X <- c(clinical_vars, cols_32X)
        rsf.obj <- model_rsf(df_trainset, df_testset, covariates_32X, event_col, duration_col, analyzes_dir, model_name, rsf_logfile)
        # plot_vimp(rsf.obj, analyzes_dir, model_name)
    } 
    else if (subdivision_type == "1320") {
        if (suffix_model == "all") {
            # Model 1320 doses volumes indicators covariates
            log_info("Model 1320 heart doses volumes")
            model_name <- "1320_dosesvol"
            cols_dosesvol <- grep("dv_\\w+_1320", colnames(df_trainset), value = TRUE)
            covariates_dv <- c(cols_dosesvol, clinical_vars)
            rsf.obj <- model_rsf(df_trainset, df_testset, covariates_dv, event_col, duration_col, analyzes_dir, model_name, rsf_logfile)
        }

        # Model 1320 radiomics firstorder covariates
        log_info("Model 1320 radiomics firstorder")
        model_name <- paste0("1320_radiomics_firstorder_", suffix_model)
        cols_1320 <- grep("^X1320_original_firstorder_", colnames(df_trainset), value = TRUE)
        covariates_1320 <- c(clinical_vars, cols_1320)
        rsf.obj <- model_rsf(df_trainset, df_testset, covariates_1320, event_col, duration_col, analyzes_dir, model_name, rsf_logfile)
        # plot_vimp(rsf.obj, analyzes_dir, model_name)
    
        # Model 1320 radiomics full covariates
        log_info("Model 1320 radiomics full")
        model_name <- paste0("1320_radiomics_full_", suffix_model)
        cols_1320 <- filter.gl(grep("^X1320_original_", colnames(df_trainset), value = TRUE))
        covariates_1320 <- c(clinical_vars, cols_1320)
        rsf.obj <- model_rsf(df_trainset, df_testset, covariates_1320, event_col, duration_col, analyzes_dir, model_name, rsf_logfile)
        # plot_vimp(rsf.obj, analyzes_dir, model_name)
        }
    else {  
        stop("Subdivision type of features unrecognized")
    }
}

# Script args
args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    config <- yaml.load_file(args[1])
    run_type <- args[2]
    subdivision_type <- args[3]
    analyzes_dir <- get.analyzes_dir_from_config(config)
    event_col <- config$EVENT_COL
    duration_col <- `if`(is.null(config$DURATION_COL), "survival_time_years", config$DURATION_COL)
    file_trainset = paste0(analyzes_dir, "datasets/trainset.csv.gz")
    file_testset = paste0(analyzes_dir, "datasets/testset.csv.gz")
    file_features <- "all"
    log_threshold(INFO)
    if (run_type == "rsf_radiomics_all") {
        rsf_learning(file_trainset, file_testset, file_features, event_col, 
                     analyzes_dir, duration_col, "all", subdivision_type)
    } else if (run_type == "rsf_radiomics_features_hclust_corr") {
        rsf_learning(file_trainset, file_testset, file_features, event_col, 
                     analyzes_dir, duration_col, "features_hclust_corr", subdivision_type)
    } else {
        stop(paste("Run type unrecognized:", run_type))
    }
} else {
    print("No arguments provided. Skipping.")
}

