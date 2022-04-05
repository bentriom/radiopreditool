
options(show.error.locations = TRUE, error=traceback)

source("workflow/scripts/utils_cox.R")

baseline_models_learning <- function(file_trainset, file_testset, event_col, analyzes_dir, duration_col) {
    dir.create(paste0(analyzes_dir, "coxph_R_plots/"), showWarnings = FALSE)
    dir.create(paste0(analyzes_dir, "coxph_R_results/"), showWarnings = FALSE)
    ntasks <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
    nworkers <- `if`(is.na(ntasks), parallel::detectCores(), ntasks)
    options(rf.cores = nworkers, mc.cores = nworkers)
    logfile <- paste0(analyzes_dir, "baseline_models_R.log")
    if (file.exists(logfile)) { file.remove(logfile) }
    log_appender(appender_file(logfile, append = TRUE))
    log_info("Baseline models learning R")
    # Dataset
    df_trainset <- read.csv(file_trainset, header = TRUE)
    df_testset <- read.csv(file_testset, header = TRUE)
    clinical_vars <- get.clinical_features(colnames(df_trainset), event_col, duration_col)
    log_info(paste0("Trainset file:", file_trainset, "with", nrow(df_trainset), "samples"))
    log_info(paste0("Testset file:", file_testset, "with", nrow(df_testset), "samples"))

    # Coxph mean dose of heart (1320)
    model_name <- "1320_mean"
    covariates <- c("X1320_original_firstorder_Mean", clinical_vars)
    log_info("Model heart mean dose (1320)")
    #model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, logfile, penalty = "none") 

    # Coxph doses volumes indicators of heart (1320)
    model_name = "1320_dosesvol"
    cols_dosesvol <- grep("dv_\\w+_1320", colnames(df_trainset), value = TRUE)
    covariates = c(cols_dosesvol, clinical_vars)
    log_info("Model heart doses volumes (1320)")
    #model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, logfile, penalty = "none")
    
    # Coxph doses volumes indicators of heart Lasso (1320)
    model_name = "1320_dosesvol_lasso"
    cols_dosesvol <- grep("dv_\\w+_1320", colnames(df_trainset), value = TRUE)
    covariates = c(cols_dosesvol, clinical_vars)
    log_info("Model heart doses volumes lasso (1320)")
    model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, logfile)
}

# Script args
args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    file_trainset = args[1]
    file_testset = args[2]
    file_features <- args[3]
    event_col <- args[4]
    analyzes_dir <- args[5]
    if (length(args) == 6) {
        duration_col <- args[6]
    } else {
        duration_col <- "survival_time_years"
    }

    log_threshold(INFO)
    baseline_models_learning(file_trainset, file_testset, event_col, analyzes_dir, duration_col)
} else {
    print("No arguments provided. Skipping.")
}

