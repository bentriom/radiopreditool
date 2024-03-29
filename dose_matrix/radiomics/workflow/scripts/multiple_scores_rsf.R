
suppressPackageStartupMessages({
  library("yaml", quietly = TRUE)
  library("hms", quietly = TRUE)
})
options(show.error.locations = TRUE, error=traceback)

source("workflow/scripts/utils_rsf.R")

multiple_scores_rsf <- function(nb_estim, screening_method, event_col, analyzes_dir, duration_col, subdivision_type) {
  stopifnot({
    nb_estim > 0
    screening_method %in% c("all", "features_hclust_corr")
  })
  dir.create(paste0(analyzes_dir, "rsf/"), showWarnings = FALSE)
  nworkers <- get.nworkers()
  options(rf.cores = 1, mc.cores = 1)
  rsf_logfile <- paste0(analyzes_dir, "multiple_scores_rsf_", subdivision_type, "_", screening_method, ".log")
  if (file.exists(rsf_logfile)) { file.remove(rsf_logfile) }
  log_appender(appender_file(rsf_logfile, append = TRUE))
  log_info("Multiple scores Random Survival Forests")
  start_time = Sys.time()
  # Dataset
  df_trainset0 <- read.csv(paste0(analyzes_dir, "datasets/trainset_0.csv.gz"), header = TRUE)
  features <- colnames(df_trainset0)
  clinical_vars <- get.clinical_features(colnames(df_trainset0), event_col, duration_col)
  log_info("Clinical variables:")
  log_info(paste(clinical_vars, collapse = " "))
  index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
  parallel.method <- `if`(Sys.getenv("SLURM_NTASKS") == "", "mclapply", "rslurm")

  if (subdivision_type == "heart") {
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
  } else if (subdivision_type == "breasts") {
    if (screening_method == "all") {
      # Model two breasts doses volumes indicators covariates
      log_info("Model breasts heart doses volumes")
      model_name <- "breasts_dosesvol"
      cols_dosesvol <- grep("dv_\\w+_(2413|3413)", features, value = TRUE)
      covariates <- c(cols_dosesvol, clinical_vars)
      parallel_multiple_scores_rsf(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                   model_name, rsf_logfile, 
                                   screening_method = screening_method, parallel.method = parallel.method)
    }
 
    # Model two breasts (2413, 3413) all radiomics covariates
    model_name <- paste0("breasts_radiomics_full_", screening_method)
    cols_breasts <- filter.gl(grep("^X(2413|3413)_", features, value = TRUE))
    covariates <- c(cols_breasts, clinical_vars)
    log_info("Model breasts (2413, 3413) radiomics full")
    parallel_multiple_scores_rsf(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, rsf_logfile, 
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else if (subdivision_type == "thorax") {
    # Model thorax all radiomics covariates
    model_name <- paste0("thorax_radiomics_full_", screening_method)
    cols_thorax <- filter.gl(grep("^X(309|310|1320|1702|2413|3413|1601)_", features, value = TRUE))
    covariates <- c(cols_thorax, clinical_vars)
    log_info("Model thorax radiomics full")
    parallel_multiple_scores_rsf(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, rsf_logfile, 
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else if (subdivision_type == "whole_body") {
    # Model whole body all radiomics covariates
    model_name <- paste0("whole_body_radiomics_full_", screening_method)
    cols_whole_body <- filter.gl(grep("^X10000_", features, value = TRUE))
    covariates <- c(cols_whole_body, clinical_vars)
    log_info("Model whole body radiomics full")
    parallel_multiple_scores_rsf(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, rsf_logfile, 
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else {
    stop(paste("Subdivision type of features unrecognized:", subdivision_type))
  }

  log_info("Done. Time:")
  log_info(format(Sys.time() - start_time))
}

# Script args
args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
  # Sys.sleep(20)
  config <- yaml.load_file(args[1])
  run_type <- args[2]
  subdivision_type <- args[3]
  analyzes_dir <- get.analyzes_dir_from_config(config)
  event_col <- config$EVENT_COL
  duration_col <- `if`(is.null(config$DURATION_COL), "survival_time_years", config$DURATION_COL)
  nb_estim <- as.numeric(config$NB_ESTIM_SCORE_MODELS)
  log_threshold(INFO)
  if (run_type == "rsf_radiomics_all") {
    multiple_scores_rsf(nb_estim, "all", event_col, analyzes_dir, duration_col, subdivision_type)
  } else if (run_type == "rsf_radiomics_features_hclust_corr") {
    multiple_scores_rsf(nb_estim, "features_hclust_corr", event_col, analyzes_dir, duration_col, subdivision_type)
  } else {
    stop(paste("Run type unrecognized:", run_type))
  }
} else {
  print("No arguments provided. Skipping.")
}

