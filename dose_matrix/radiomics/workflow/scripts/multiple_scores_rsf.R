
suppressPackageStartupMessages({library("yaml", quietly = TRUE)})
options(show.error.locations = TRUE, error=traceback)

source("workflow/scripts/utils_rsf.R")

multiple_scores_rsf <- function(nb_estim, file_features, event_col, analyzes_dir, duration_col, suffix_model) {
    dir.create(paste0(analyzes_dir, "rsf_results/"), showWarnings = FALSE)
    nworkers <- get.nworkers()
    options(rf.cores = 1, mc.cores = 1)
    rsf_logfile <- paste0(analyzes_dir, "multiple_scores_rsf_", suffix_model, ".log")
    if (file.exists(rsf_logfile)) { file.remove(rsf_logfile) }
    log_appender(appender_file(rsf_logfile, append = TRUE))
    log_info("Multiple scores")
    # Dataset
    df_trainset0 <- read.csv(paste0(analyzes_dir, "datasets/trainset_0.csv.gz"), header = TRUE)
    # Select subset of features due to feature elimination
    features <- `if`(file_features == "all", colnames(df_trainset0), as.character(read.csv(file_features)[,1]))
    # Add "X" for R colname compatibility
    features <- as.character(lapply(features, function(x) { `if`(str_detect(substr(x, 1, 1), "[0-9]"), paste0("X", x), x) }))
    clinical_vars <- get.clinical_features(colnames(df_trainset0), event_col, duration_col)
    index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
       
    if (suffix_model == "all") { 
        # Model 1320 doses volumes indicators covariates
        log_info("Model 1320 heart doses volumes")
        model_name <- "1320_dosesvol"
        cols_dosesvol <- grep("dv_\\w+_1320", features, value = TRUE)
        covariates_dv <- c(cols_dosesvol, clinical_vars)
        results <- mclapply(0:(nb_estim-1), function (i) { model_rsf.id(i, covariates_dv, event_col, duration_col, analyzes_dir, model_name, rsf_logfile) }, mc.cores = nworkers) 
        results <- as.data.frame(results)
        df_results <- data.frame(Mean = apply(results, 1, mean), Std = apply(results, 1, sd)) 
        rownames(df_results) <- index_results
        filename_results <- paste0(analyzes_dir, "rsf_results/", model_name, "/", nb_estim, "_runs_test_metrics.csv")
        write.csv(df_results, file = filename_results, row.names = TRUE)
    }

    # Model 32X radiomics firstorder covariates
    log_info("Model 32X radiomics firstorder")
    model_name <- paste0("32X_radiomics_firstorder_", suffix_model)
    cols_32X <- grep("^X32[0-9]{1}_original_firstorder_", features, value = TRUE)
    covariates_32X <- c(clinical_vars, cols_32X)
    results <- mclapply(0:(nb_estim-1), function (i) { model_rsf.id(i, covariates_32X, event_col, duration_col, analyzes_dir, model_name, rsf_logfile) }, mc.cores = nworkers) 
    results <- as.data.frame(results)
    df_results <- data.frame(Mean = apply(results, 1, mean), Std = apply(results, 1, sd)) 
    rownames(df_results) <- index_results
    filename_results <- paste0(analyzes_dir, "rsf_results/", model_name, "/", nb_estim, "_runs_test_metrics.csv")
    write.csv(df_results, file = filename_results, row.names = TRUE)

    # Model 1320 radiomics firstorder covariates
    log_info("Model 1320 radiomics firstorder")
    model_name <- paste0("1320_radiomics_firstorder_", suffix_model)
    cols_1320 <- grep("^X1320_original_firstorder_", features, value = TRUE)
    covariates_1320 <- c(clinical_vars, cols_1320)
    results <- mclapply(0:(nb_estim-1), function (i) { model_rsf.id(i, covariates_1320, event_col, duration_col, analyzes_dir, model_name, rsf_logfile) }, mc.cores = nworkers)
    results <- as.data.frame(results)
    df_results <- data.frame(Mean = apply(results, 1, mean), Std = apply(results, 1, sd)) 
    rownames(df_results) <- index_results
    filename_results <- paste0(analyzes_dir, "rsf_results/", model_name, "/", nb_estim, "_runs_test_metrics.csv")
    write.csv(df_results, file = filename_results, row.names = TRUE)

    # Model 32X all radiomics covariates
    log_info("Model 32X radiomics full")
    model_name <- paste0("32X_radiomics_full_", suffix_model)
    cols_32X <- filter.gl(grep("^X32[0-9]{1}_original_", features, value = TRUE))
    covariates_32X <- c(clinical_vars, cols_32X)
    results <- mclapply(0:(nb_estim-1), function (i) { model_rsf.id(i, covariates_32X, event_col, duration_col, analyzes_dir, model_name, rsf_logfile) }, mc.cores = nworkers) 
    results <- as.data.frame(results)
    df_results <- data.frame(Mean = apply(results, 1, mean), Std = apply(results, 1, sd)) 
    rownames(df_results) <- index_results
    filename_results <- paste0(analyzes_dir, "rsf_results/", model_name, "/", nb_estim, "_runs_test_metrics.csv")
    write.csv(df_results, file = filename_results, row.names = TRUE)

    # Model 1320 all radiomics covariates
    log_info("Model 1320 radiomics full")
    model_name <- paste0("1320_radiomics_full_", suffix_model)
    cols_1320 <- filter.gl(grep("^X1320_original_", features, value = TRUE))
    covariates_1320 <- c(clinical_vars, cols_1320)
    results <- mclapply(0:(nb_estim-1), function (i) { model_rsf.id(i, covariates_1320, event_col, duration_col, analyzes_dir, model_name, rsf_logfile) }, mc.cores = nworkers)
    results <- as.data.frame(results)
    df_results <- data.frame(Mean = apply(results, 1, mean), Std = apply(results, 1, sd)) 
    rownames(df_results) <- index_results
    filename_results <- paste0(analyzes_dir, "rsf_results/", model_name, "/", nb_estim, "_runs_test_metrics.csv")
    write.csv(df_results, file = filename_results, row.names = TRUE)
}

# Script args
args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    config <- yaml.load_file(args[1])
    run_type <- args[2]
    analyzes_dir <- get.analyzes_dir_from_config(config)
    event_col <- config$EVENT_COL
    duration_col <- `if`(is.null(config$DURATION_COL), "survival_time_years", config$DURATION_COL)
    file_trainset = paste0(analyzes_dir, "datasets/trainset.csv.gz")
    file_testset = paste0(analyzes_dir, "datasets/testset.csv.gz")
    file_features <- "all"
    log_threshold(INFO)
    if (run_type == "rsf_radiomics_all") {
        multiple_scores_rsf(nb_estim, file_features, event_col, analyzes_dir, duration_col, "all")
    } else if (run_type == "rsf_radiomics_features_hclust_corr") {
        multiple_scores_rsf(nb_estim, file_features, event_col, analyzes_dir, duration_col, "features_hclust_corr")
    } else {
        stop(paste("Run type unrecognized:", run_type))
    }
} else {
    print("No arguments provided. Skipping.")
}

