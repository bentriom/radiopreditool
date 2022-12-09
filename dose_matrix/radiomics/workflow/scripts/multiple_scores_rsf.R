
suppressPackageStartupMessages({
    library("yaml", quietly = TRUE)
    library("hms", quietly = TRUE)
})
options(show.error.locations = TRUE, error=traceback)

source("workflow/scripts/utils_rsf.R")

multiple_scores_rsf <- function(nb_estim, screening_method, event_col, analyzes_dir, duration_col) {
    stopifnot({
      nb_estim > 0
      screening_method %in% c("all", "features_hclust_corr")
    })
    dir.create(paste0(analyzes_dir, "rsf/"), showWarnings = FALSE)
    nworkers <- get.nworkers()
    options(rf.cores = 1, mc.cores = 1)
    rsf_logfile <- paste0(analyzes_dir, "multiple_scores_rsf_", screening_method, ".log")
    if (file.exists(rsf_logfile)) { file.remove(rsf_logfile) }
    log_appender(appender_file(rsf_logfile, append = TRUE))
    log_info("Multiple scores Random Survival Forests")
    start_time = Sys.time()
    # Dataset
    df_trainset0 <- read.csv(paste0(analyzes_dir, "datasets/trainset_0.csv.gz"), header = TRUE)
    features <- colnames(df_trainset0)
    # 
    # # Select subset of features due to feature elimination
    # features <- `if`(file_features == "all", colnames(df_trainset0), as.character(read.csv(file_features)[,1]))
    # # Add "X" for R colname compatibility
    # features <- as.character(lapply(features, function(x) { `if`(str_detect(substr(x, 1, 1), "[0-9]"), 
    #                                                              paste0("X", x), x) }))
    clinical_vars <- get.clinical_features(colnames(df_trainset0), event_col, duration_col)
    index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
    parallel.method <- `if`(Sys.getenv("SLURM_NTASKS") == "", "mclapply", "rslurm")
       
    if (screening_method == "all") { 
        # Model 1320 doses volumes indicators covariates
        log_info("Model 1320 heart doses volumes")
        model_name <- "1320_dosesvol"
        cols_dosesvol <- grep("dv_\\w+_1320", features, value = TRUE)
        covariates <- c(cols_dosesvol, clinical_vars)
        parallel_multiple_scores_rsf(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                     model_name, rsf_logfile, 
                                     screening_method = screening_method, parallel.method = parallel.method)
    }

    # Model 32X radiomics firstorder covariates
    log_info("Model 32X radiomics firstorder")
    model_name <- paste0("32X_radiomics_firstorder_", screening_method)
    cols_32X <- grep("^X32[0-9]{1}_original_firstorder_", features, value = TRUE)
    covariates <- c(clinical_vars, cols_32X)
    parallel_multiple_scores_rsf(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, rsf_logfile,
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Model 1320 radiomics firstorder covariates
    log_info("Model 1320 radiomics firstorder")
    model_name <- paste0("1320_radiomics_firstorder_", screening_method)
    cols_1320 <- grep("^X1320_original_firstorder_", features, value = TRUE)
    covariates <- c(clinical_vars, cols_1320)
    parallel_multiple_scores_rsf(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, rsf_logfile,
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Model 32X all radiomics covariates
    log_info("Model 32X radiomics full")
    model_name <- paste0("32X_radiomics_full_", screening_method)
    cols_32X <- filter.gl(grep("^X32[0-9]{1}_original_", features, value = TRUE))
    covariates <- c(clinical_vars, cols_32X)
    parallel_multiple_scores_rsf(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, rsf_logfile,
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Model 1320 all radiomics covariates
    log_info("Model 1320 radiomics full")
    model_name <- paste0("1320_radiomics_full_", screening_method)
    cols_1320 <- filter.gl(grep("^X1320_original_", features, value = TRUE))
    covariates <- c(clinical_vars, cols_1320)
    parallel_multiple_scores_rsf(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, rsf_logfile, 
                                 screening_method = screening_method, parallel.method = parallel.method)
    
    log_info("Done. Time:")
    log_info(format(Sys.time() - start_time))
}

# Script args
args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    config <- yaml.load_file(args[1])
    run_type <- args[2]
    analyzes_dir <- get.analyzes_dir_from_config(config)
    event_col <- config$EVENT_COL
    duration_col <- `if`(is.null(config$DURATION_COL), "survival_time_years", config$DURATION_COL)
    nb_estim <- as.numeric(config$NB_ESTIM_SCORE_MODELS)
    log_threshold(INFO)
    if (run_type == "rsf_radiomics_all") {
        multiple_scores_rsf(nb_estim, "all", event_col, analyzes_dir, duration_col)
    } else if (run_type == "rsf_radiomics_features_hclust_corr") {
        file_features <- paste0(analyzes_dir, "screening/features_hclust_corr.csv")
        multiple_scores_rsf(nb_estim, "features_hclust_corr", event_col, analyzes_dir, duration_col)
    } else {
        stop(paste("Run type unrecognized:", run_type))
    }
} else {
    print("No arguments provided. Skipping.")
}

