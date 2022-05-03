
options(show.error.locations = TRUE, error=traceback)

source("workflow/scripts/utils_cox.R")

baseline_models_learning <- function(file_trainset, file_testset, event_col, analyzes_dir, duration_col) {
    dir.create(paste0(analyzes_dir, "coxph_R_plots/"), showWarnings = FALSE)
    dir.create(paste0(analyzes_dir, "coxph_R_results/"), showWarnings = FALSE)
    ntasks <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
    nworkers <- `if`(is.na(ntasks), parallel::detectCores(), ntasks)
    logfile <- paste0(analyzes_dir, "baseline_models_R.log")
    if (file.exists(logfile)) { file.remove(logfile) }
    log_appender(appender_file(logfile, append = TRUE))
    log_info("Baseline models learning R")
    # Dataset
    df_trainset <- read.csv(file_trainset, header = TRUE)
    df_testset <- read.csv(file_testset, header = TRUE)
    clinical_vars <- get.clinical_features(colnames(df_trainset), event_col, duration_col)
    log_info(paste("Trainset file:", file_trainset, "with", nrow(df_trainset), "samples"))
    log_info(paste("Testset file:", file_testset, "with", nrow(df_testset), "samples"))

    # Coxph mean dose of heart (1320)
    model_name <- "1320_mean"
    covariates <- c("X1320_original_firstorder_Mean", clinical_vars)
    log_info("Model heart mean dose (1320)")
    model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, logfile, penalty = "none") 

    # Coxph doses volumes indicators of heart (1320)
    model_name = "1320_dosesvol"
    cols_dosesvol <- grep("dv_\\w+_1320", colnames(df_trainset), value = TRUE)
    covariates = c(cols_dosesvol, clinical_vars)
    log_info("Model heart doses volumes (1320)")
    model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, logfile, penalty = "none")
    
    # Coxph doses volumes indicators of heart Lasso (1320)
    model_name = "1320_dosesvol_lasso"
    cols_dosesvol <- grep("dv_\\w+_1320", colnames(df_trainset), value = TRUE)
    covariates = c(cols_dosesvol, clinical_vars)
    log_info("Model heart doses volumes lasso (1320)")
    model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, logfile)
}

cox_radiomics_learning <- function(file_trainset, file_testset, file_features, event_col, analyzes_dir, duration_col, suffix_model) {
    dir.create(paste0(analyzes_dir, "coxph_R_plots/"), showWarnings = FALSE)
    dir.create(paste0(analyzes_dir, "coxph_R_results/"), showWarnings = FALSE)
    ntasks <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
    nworkers <- `if`(is.na(ntasks), parallel::detectCores(), ntasks)
    logfile <- paste0(analyzes_dir, "cox_lasso_radiomics_R_",suffix_model,".log")
    if (file.exists(logfile)) { file.remove(logfile) }
    log_appender(appender_file(logfile, append = TRUE))
    log_info("Cox lasso radiomics learning R")
    # Dataset
    df_trainset <- read.csv(file_trainset, header = TRUE)
    df_testset <- read.csv(file_testset, header = TRUE)
    features <- `if`(file_features == "all", colnames(df_trainset), as.character(read.csv(file_features)[,1]))
    # Add "X" for R colname compatibility
    features <- as.character(lapply(features, function(x) { `if`(str_detect(substr(x, 1, 1), "[0-9]"), paste("X", x, sep = ""), x) }))
    df_trainset <- df_trainset[,features]
    df_testset <- df_testset[,features]
    clinical_vars <- get.clinical_features(colnames(df_trainset), event_col, duration_col)
    log_info(paste("Trainset file:", file_trainset, "with", nrow(df_trainset), "samples"))
    log_info(paste("Testset file:", file_testset, "with", nrow(df_testset), "samples"))

    # Coxph Lasso radiomics firstorder 32X
    model_name = paste0("32X_radiomics_firstorder_lasso_", suffix_model)
    cols_32X_firstorder <- grep("^X32[0-9]{1}_original_firstorder_", colnames(df_trainset), value = TRUE)
    covariates = c(cols_32X_firstorder, clinical_vars)
    log_info("Model radiomics firstorder lasso (32X)")
    model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, logfile)
    
    # Coxph Lasso radiomics firstorder 1320
    model_name = paste0("1320_radiomics_firstorder_lasso_", suffix_model)
    cols_1320_firstorder <- grep("^X1320_original_firstorder_", colnames(df_trainset), value = TRUE)
    covariates = c(cols_1320_firstorder, clinical_vars)
    log_info("Model radiomics firstorder lasso (1320)")
    model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, logfile)

    # Coxph Lasso all radiomics 32X
    model_name = paste0("32X_radiomics_full_lasso_", suffix_model)
    cols_32X <- grep("^X32[0-9]{1}_", colnames(df_trainset), value = TRUE)
    covariates = c(cols_32X, clinical_vars)
    log_info("Model radiomics full lasso (32X)")
    model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, logfile)
 
    # Coxph Lasso all radiomics 1320
    model_name = paste0("1320_radiomics_full_lasso_", suffix_model)
    cols_1320 <- grep("^X1320_", colnames(df_trainset), value = TRUE)
    covariates = c(cols_1320, clinical_vars)
    log_info("Model radiomics full lasso (1320)")
    model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, logfile)
}

# Script args
args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    run_type <- args[1]
    analyzes_dir <- args[2]
    event_col <- args[3]
    duration_col <- args[4]
    file_trainset = paste0(analyzes_dir, "datasets/trainset.csv.gz")
    file_testset = paste0(analyzes_dir, "datasets/testset.csv.gz")
    log_threshold(INFO)
    if (run_type == "baseline_models") {
        baseline_models_learning(file_trainset, file_testset, event_col, analyzes_dir, duration_col)
    } else if (run_type == "cox_lasso_radiomics_all") {
        file_features <- "all"
        cox_radiomics_learning(file_trainset, file_testset, file_features, event_col, analyzes_dir, duration_col, "all")
    } else if (run_type == "cox_lasso_radiomics_features_hclust_corr") {
        file_features <- args[5]
        cox_radiomics_learning(file_trainset, file_testset, file_features, event_col, analyzes_dir, duration_col, "features_hclust_corr")
    } else {
        print("Run type unrecognized.")
    }
} else {
    print("No arguments provided. Skipping.")
}

