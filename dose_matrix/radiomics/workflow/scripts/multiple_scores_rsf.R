
options(show.error.locations = TRUE, error=traceback)

library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("randomForestSRC", quietly = TRUE)
library("pec", quietly = TRUE)
library("Hmisc", quietly = TRUE)
library("logger", quietly = TRUE)
library("parallel", quietly = TRUE)

source("workflow/scripts/utils_rsf.R")

multiple_scores_rsf <- function(nb_estim, file_features, event_col, analyzes_dir, duration_col, suffix_model) {
    dir.create(paste(analyzes_dir, "rsf_results/", sep = ""), showWarnings = FALSE)
    ntasks <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
    nworkers <- `if`(is.na(ntasks), parallel::detectCores(), ntasks)
    options(rf.cores = 1, mc.cores = 1)
    rsf_logfile <- paste(analyzes_dir, "multiple_scores_rsf_", suffix_model, ".log", sep = "")
    if (file.exists(rsf_logfile)) { file.remove(rsf_logfile) }
    log_appender(appender_file(rsf_logfile, append = TRUE))
    log_info("Multiple scores")
    # Dataset
    df_trainset0 <- read.csv(paste(analyzes_dir, "datasets/trainset_0.csv.gz", sep = ""), header = TRUE)
    # Select subset of features due to feature elimination
    features <- `if`(file_features == "all", colnames(df_trainset0), as.character(read.csv(file_features)[,1]))
    # Add "X" for R colname compatibility
    features <- as.character(lapply(features, function(x) { `if`(str_detect(substr(x, 1, 1), "[0-9]"), paste("X", x, sep = ""), x) }))
    clinical_vars <- get.clinical_features(colnames(df_trainset0), event_col, duration_col)
    index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")

    # Model 32X radiomics covariates
    log_info("Model 32X")
    model_name <- paste("model_32X_", suffix_model, sep = "")
    cols_32X <- grep("^X32[0-9]{1}_", features, value = TRUE)
    covariates_32X <- c(clinical_vars, cols_32X)
    results <- mclapply(0:(nb_estim-1), function (i) { refit.best.rsf.id(i, covariates_32X, event_col, duration_col, analyzes_dir, model_name = model_name) }, mc.cores = nworkers) 
    results <- as.data.frame(results)
    df_results <- data.frame(Mean = apply(results, 1, mean), Std = apply(results, 1, sd)) 
    rownames(df_results) <- index_results
    filename_results <- paste(analyzes_dir, "rsf_results/", nb_estim, "_runs_test_metrics_", model_name, ".csv", sep = "")
    write.csv(df_results, file = filename_results, row.names = TRUE)

    # Model 1320 radiomics covariates
    log_info("Model 1320")
    model_name <- paste("model_1320_", suffix_model, sep = "")
    cols_1320 <- grep("^X1320_", features, value = TRUE)
    covariates_1320 <- c(clinical_vars, cols_1320)
    results <- mclapply(0:(nb_estim-1), function (i) { refit.best.rsf.id(i, covariates_1320, event_col, duration_col, analyzes_dir, model_name = model_name) }, mc.cores = nworkers)
    results <- as.data.frame(results)
    df_results <- data.frame(Mean = apply(results, 1, mean), Std = apply(results, 1, sd)) 
    rownames(df_results) <- index_results
    filename_results <- paste(analyzes_dir, "rsf_results/", nb_estim, "_runs_test_metrics_", model_name, ".csv", sep = "")
    write.csv(df_results, file = filename_results, row.names = TRUE)
}

# Script args
args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    nb_estim <- args[1]
    file_features <- args[2]
    event_col <- args[3]
    analyzes_dir <- args[4]
    suffix_model <- args[5]
    if (length(args) == 6) {
        duration_col <- args[6]
    } else {
        duration_col <- "survival_time_years"
    }

    log_threshold(INFO)
    multiple_scores_rsf(nb_estim, file_features, event_col, analyzes_dir, duration_col, suffix_model)
} else {
    print("No arguments provided. Skipping.")
}


