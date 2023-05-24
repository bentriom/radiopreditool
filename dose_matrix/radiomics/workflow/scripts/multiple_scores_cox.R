
suppressPackageStartupMessages({
  library("yaml", quietly = TRUE)
  library("hms", quietly = TRUE)
})
options(show.error.locations = TRUE, error=traceback)

source("workflow/scripts/utils_cox.R")

multiple_scores_baseline_models <- function(nb_estim, event_col, analyzes_dir, duration_col, subdivision_type) {
  dir.create(paste0(analyzes_dir, "coxph_R/"), showWarnings = FALSE)
  nworkers <- get.nworkers()
  cox_logfile <- paste0(analyzes_dir, "multiple_scores_baseline_models_", subdivision_type, "_R.log")
  if (file.exists(cox_logfile)) { file.remove(cox_logfile) }
  log_appender(appender_file(cox_logfile, append = TRUE))
  log_info(paste0("Multiple scores baseline models learning R (",nworkers," workers)"))
  start_time = Sys.time()
  # Dataset
  df_trainset0 <- read.csv(paste0(analyzes_dir, "datasets/trainset_0.csv.gz"), header = TRUE)
  clinical_vars <- get.clinical_features(colnames(df_trainset0), event_col, duration_col)
  index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
  parallel.method <- `if`(Sys.getenv("SLURM_NTASKS") == "", "mclapply", "rslurm")

  if (subdivision_type == "heart") {
    # Coxph mean dose of heart (1320)
    model_name <- "1320_mean"
    covariates <- c("X1320_original_firstorder_Mean", clinical_vars)
    log_info("Multiple scores heart mean dose (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "none", parallel.method = parallel.method)

    # Coxph doses volumes indicators of heart (1320)
    model_name = "1320_dosesvol"
    cols_dosesvol <- grep("dv_\\w+_1320", colnames(df_trainset0), value = TRUE)
    covariates = c(cols_dosesvol, clinical_vars)
    log_info("Multiple scores heart doses volumes (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "none", parallel.method = parallel.method)

    # Coxph doses volumes indicators of heart Lasso (1320)
    model_name = "1320_dosesvol_lasso"
    cols_dosesvol <- grep("dv_\\w+_1320", colnames(df_trainset0), value = TRUE)
    covariates = c(cols_dosesvol, clinical_vars)
    log_info("Multiple scores heart doses volumes lasso (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", parallel.method = parallel.method)
  } else if (subdivision_type == "breasts") {
    # Coxph mean dose of breasts
    model_name <- "breasts_mean"
    covariates <- c("X2413_original_firstorder_Mean", "X3413_original_firstorder_Mean", clinical_vars)
    log_info("Multiple scores breasts mean dose (2413, 3413)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "none", parallel.method = parallel.method)
    
    # Coxph mean dose of breasts Lasso
    model_name <- "breasts_mean_lasso"
    covariates <- c("X2413_original_firstorder_Mean", "X3413_original_firstorder_Mean", clinical_vars)
    log_info("Multiple scores breasts mean dose (2413, 3413)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", parallel.method = parallel.method)

    # Coxph doses volumes indicators of breasts
    # model_name = "breasts_dosesvol"
    # cols_dosesvol <- grep("dv_\\w+_(2413|3413)", colnames(df_trainset0), value = TRUE)
    # covariates = c(cols_dosesvol, clinical_vars)
    # log_info("Multiple scores breasts doses volumes (2413, 3413)")
    # parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
    #                              model_name, cox_logfile, penalty = "none", parallel.method = parallel.method)

    # Coxph doses volumes indicators of breasts Lasso
    model_name = "breasts_dosesvol_lasso"
    cols_dosesvol <- grep("dv_\\w+_(2413|3413)", colnames(df_trainset0), value = TRUE)
    covariates = c(cols_dosesvol, clinical_vars)
    log_info("Multiple scores breasts doses volumes lasso (2413, 3413)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", parallel.method = parallel.method)
  } else {
    stop(paste("Subdivision type of features unrecognized:", subdivision_type))
  }

  log_info("Done. Time:")
  log_info(format(Sys.time() - start_time))
}

multiple_scores_cox_radiomics <- function(nb_estim, screening_method, event_col, 
                                          analyzes_dir, duration_col, subdivision_type) {
  stopifnot({
    nb_estim > 0
    screening_method %in% c("all", "features_hclust_corr")
  })
  dir.create(paste0(analyzes_dir, "coxph_R/"), showWarnings = FALSE)
  nworkers <- get.nworkers()
  cox_logfile <- paste0(analyzes_dir, "multiple_scores_cox_lasso_radiomics_", 
                        subdivision_type, "_", screening_method, "_R.log")
  if (file.exists(cox_logfile)) { file.remove(cox_logfile) }
  log_appender(appender_file(cox_logfile, append = TRUE))
  log_info(paste0("Multiple scores cox lasso radiomics learning R (",nworkers," workers)"))
  start_time = Sys.time()
  # Dataset
  df_dataset <- read.csv(paste0(analyzes_dir, "datasets/dataset.csv.gz"), header = TRUE)
  features <- colnames(df_dataset)
  clinical_vars <- get.clinical_features(features, event_col, duration_col)
  index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
  parallel.method <- `if`(Sys.getenv("SLURM_NTASKS") == "", "mclapply", "rslurm")

  if (subdivision_type == "heart") {
    # Coxph Lasso radiomics firstorder 32X
    model_name = paste0("32X_radiomics_firstorder_lasso_", screening_method)
    cols_32X <- grep("^X32[0-9]{1}_original_firstorder_", features, value = TRUE)
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics firstorder lasso (32X)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", 
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Coxph Lasso radiomics firstorder 1320
    model_name = paste0("1320_radiomics_firstorder_lasso_", screening_method)
    cols_1320 <- grep("^X1320_original_firstorder_", features, value = TRUE)
    covariates = c(cols_1320, clinical_vars)
    log_info("Multiple scores radiomics firstorder lasso (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", 
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Coxph Lasso all radiomics 32X
    model_name = paste0("32X_radiomics_full_lasso_", screening_method)
    cols_32X <- filter.gl(grep("^X32[0-9]{1}_original_", features, value = TRUE))
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics full lasso (32X)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", 
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Coxph Lasso all radiomics 1320
    model_name = paste0("1320_radiomics_full_lasso_", screening_method)
    cols_1320 <- filter.gl(grep("^X1320_original_", features, value = TRUE))
    covariates = c(cols_1320, clinical_vars)
    log_info("Multiple scores radiomics full lasso (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso",
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else if (subdivision_type == "breasts") {
    # Coxph Lasso all radiomics of the two breasts (2413, 3413)
    model_name <- paste0("breasts_radiomics_full_lasso_", screening_method)
    cols_breasts <- filter.gl(grep("^X(2413|3413)_", features, value = TRUE))
    covariates <- c(cols_breasts, clinical_vars)
    log_info("Multiple scores radiomics full lasso (breasts: 2413, 3413)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso",
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else if (subdivision_type == "thorax") {
    # Coxph Lasso all radiomics of the thorax
    model_name <- paste0("thorax_radiomics_full_lasso_", screening_method)
    cols_thorax <- filter.gl(grep("^X(309|310|1320|1702|2413|3413|1601)_", features, value = TRUE))
    covariates <- c(cols_thorax, clinical_vars)
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso",
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else {
    stop(paste("Subdivision type of features unrecognized:", subdivision_type))
  }

  log_info("Done")
  log_info(format(Sys.time() - start_time))
}

multiple_scores_cox_sis_radiomics <- function(nb_estim, screening_method, event_col, 
                                              analyzes_dir, duration_col, subdivision_type) {
  stopifnot({
    nb_estim > 0
    screening_method %in% c("all", "features_hclust_corr")
  })
  dir.create(paste0(analyzes_dir, "coxph_R/"), showWarnings = FALSE)
  nworkers <- get.nworkers()
  cox_logfile <- paste0(analyzes_dir, "multiple_scores_cox_sis_radiomics_",
                        subdivision_type, "_", screening_method, "_R.log")
  if (file.exists(cox_logfile)) { file.remove(cox_logfile) }
  log_appender(appender_file(cox_logfile, append = TRUE))
  log_info(paste0("Multiple scores cox (I)SIS radiomics learning R (",nworkers," workers)"))
  start_time = Sys.time()
  # Dataset
  df_dataset <- read.csv(paste0(analyzes_dir, "datasets/dataset.csv.gz"), header = TRUE)
  features <- colnames(df_dataset)
  clinical_vars <- get.clinical_features(features, event_col, duration_col)
  index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
  parallel.method <- `if`(Sys.getenv("SLURM_NTASKS") == "", "mclapply", "rslurm")

  if (subdivision_type == "heart") {
    # Coxph SIS selection radiomics 32X
    model_name = paste0("32X_radiomics_full_sis_", screening_method)
    cols_32X <- filter.gl(grep("^X32[0-9]{1}_original_", features, value = TRUE))
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics full sis (32X)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "sis", 
                                 screening_method = screening_method, parallel.method = parallel.method)
    
    # Coxph SIS selection radiomics 32X marrow
    model_name = paste0("32X_marrow_radiomics_full_sis_", screening_method)
    cols_32X <- filter.gl(grep("^X(32[0-9]{1}|1601)_original_", features, value = TRUE))
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics full sis (32X, 1601)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "sis", 
                                 screening_method = screening_method, parallel.method = parallel.method)
    
    # Coxph SIS selection radiomics 1320 marrow
    model_name = paste0("1320_marrow_radiomics_full_sis_", screening_method)
    cols_32X <- filter.gl(grep("^X(1320|1601)_original_", features, value = TRUE))
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics full sis (1320, 1601)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "sis", 
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else if (subdivision_type == "breasts") {
    # Coxph SIS selection all radiomics of the two breasts (2413, 3413)
    model_name <- paste0("breasts_radiomics_full_sis_", screening_method)
    cols_breasts <- filter.gl(grep("^X(2413|3413)_", features, value = TRUE))
    covariates <- c(cols_breasts, clinical_vars)
    log_info("Multiple scores radiomics full sis (breasts: 2413, 3413)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "sis",
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else if (subdivision_type == "thorax") {
    # Coxph SIS selection all radiomics of the thorax
    model_name <- paste0("thorax_radiomics_full_sis_", screening_method)
    cols_thorax <- filter.gl(grep("^X(309|310|1320|1702|2413|3413|1601)_", features, value = TRUE))
    covariates <- c(cols_thorax, clinical_vars)
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "sis",
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else {
    stop(paste("Subdivision type of features unrecognized:", subdivision_type))
  }

  log_info("Done")
  log_info(format(Sys.time() - start_time))
}

multiple_scores_cox_bootstrap_radiomics <- function(nb_estim, screening_method, event_col, 
                                                    analyzes_dir, duration_col, subdivision_type, n_boot) {
  stopifnot({
    nb_estim > 0
    screening_method %in% c("all", "features_hclust_corr")
  })
  dir.create(paste0(analyzes_dir, "coxph_R/"), showWarnings = FALSE)
  nworkers <- get.nworkers()
  cox_logfile <- paste0(analyzes_dir, "multiple_scores_cox_bootstrap_lasso_radiomics_",
                        subdivision_type, "_", screening_method, "_R.log")
  if (file.exists(cox_logfile)) { file.remove(cox_logfile) }
  log_appender(appender_file(cox_logfile, append = TRUE))
  log_info(paste0("Multiple scores cox bootstrap lasso radiomics learning R (",nworkers," workers)"))
  start_time = Sys.time()
  # Dataset
  df_dataset <- read.csv(paste0(analyzes_dir, "datasets/dataset.csv.gz"), header = TRUE)
  features <- colnames(df_dataset)
  clinical_vars <- get.clinical_features(features, event_col, duration_col)
  index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
  parallel.method <- `if`(Sys.getenv("SLURM_NTASKS") == "", "mclapply", "rslurm")

  if (subdivision_type == "heart") {
    # Coxph Bootstrap Lasso radiomics firstorder 32X
    model_name = paste0("32X_radiomics_firstorder_bootstrap_lasso_", screening_method)
    cols_32X <- grep("^X32[0-9]{1}_original_firstorder_", features, value = TRUE)
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics firstorder bootstrap lasso (32X)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir, model_name, cox_logfile, 
                                 penalty = "bootstrap_lasso", n_boot = n_boot,
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Coxph Bootstrap Lasso radiomics firstorder 1320
    model_name = paste0("1320_radiomics_firstorder_bootstrap_lasso_", screening_method)
    cols_1320 <- grep("^X1320_original_firstorder_", features, value = TRUE)
    covariates = c(cols_1320, clinical_vars)
    log_info("Multiple scores radiomics firstorder bootstrap lasso (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,model_name, cox_logfile, 
                                 penalty = "bootstrap_lasso", n_boot = n_boot,
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Coxph Bootstrap Lasso all radiomics 32X
    model_name = paste0("32X_radiomics_full_bootstrap_lasso_", screening_method)
    cols_32X <- filter.gl(grep("^X32[0-9]{1}_original_", features, value = TRUE))
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics full bootstrap lasso (32X)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir, model_name, cox_logfile, 
                                 penalty = "bootstrap_lasso", n_boot = n_boot,
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Coxph Bootstrap Lasso all radiomics 1320
    model_name = paste0("1320_radiomics_full_bootstrap_lasso_", screening_method)
    cols_1320 <- filter.gl(grep("^X1320_original_", features, value = TRUE))
    covariates = c(cols_1320, clinical_vars)
    log_info("Multiple scores radiomics full bootstrap lasso (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir, model_name, cox_logfile, 
                                 penalty = "bootstrap_lasso", n_boot = n_boot,
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else if (subdivision_type == "breasts") {
    # Coxph Bootstrap Lasso all radiomics of the two breasts (2413, 3413)
    model_name <- paste0("breasts_radiomics_full_bootstrap_lasso_", screening_method)
    cols_breasts <- filter.gl(grep("^X(2413|3413)_", features, value = TRUE))
    covariates <- c(cols_breasts, clinical_vars)
    log_info("Multiple scores radiomics full bootstrap lasso (breasts: 2413, 3413)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir, model_name, cox_logfile, 
                                 penalty = "bootstrap_lasso", n_boot = n_boot,
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else if (subdivision_type == "thorax") {
    # Coxph Bootstrap Lasso all radiomics of the thorax
    model_name <- paste0("thorax_radiomics_full_bootstrap_lasso_", screening_method)
    cols_thorax <- filter.gl(grep("^X(309|310|1320|1702|2413|3413|1601)_", features, value = TRUE))
    covariates <- c(cols_thorax, clinical_vars)
    log_info("Multiple scores radiomics full bootstrap lasso (thorax)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir, model_name, cox_logfile, 
                                 penalty = "bootstrap_lasso", n_boot = n_boot,
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else {
    stop(paste("Subdivision type of features unrecognized:", subdivision_type))
  }
  log_info("Done")
  log_info(format(Sys.time() - start_time))
}

multiple_scores_cox_radiomics <- function(nb_estim, screening_method, event_col, 
                                          analyzes_dir, duration_col, subdivision_type) {
  stopifnot({
    nb_estim > 0
    screening_method %in% c("all", "features_hclust_corr")
  })
  dir.create(paste0(analyzes_dir, "coxph_R/"), showWarnings = FALSE)
  nworkers <- get.nworkers()
  cox_logfile <- paste0(analyzes_dir, "multiple_scores_cox_lasso_radiomics_", 
                        subdivision_type, "_", screening_method, "_R.log")
  if (file.exists(cox_logfile)) { file.remove(cox_logfile) }
  log_appender(appender_file(cox_logfile, append = TRUE))
  log_info(paste0("Multiple scores cox lasso radiomics learning R (",nworkers," workers)"))
  start_time = Sys.time()
  # Dataset
  df_dataset <- read.csv(paste0(analyzes_dir, "datasets/dataset.csv.gz"), header = TRUE)
  features <- colnames(df_dataset)
  clinical_vars <- get.clinical_features(features, event_col, duration_col)
  index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
  parallel.method <- `if`(Sys.getenv("SLURM_NTASKS") == "", "mclapply", "rslurm")

  if (subdivision_type == "heart") {
    # Coxph Lasso radiomics firstorder 32X
    model_name = paste0("32X_radiomics_firstorder_lasso_", screening_method)
    cols_32X <- grep("^X32[0-9]{1}_original_firstorder_", features, value = TRUE)
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics firstorder lasso (32X)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", 
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Coxph Lasso radiomics firstorder 1320
    model_name = paste0("1320_radiomics_firstorder_lasso_", screening_method)
    cols_1320 <- grep("^X1320_original_firstorder_", features, value = TRUE)
    covariates = c(cols_1320, clinical_vars)
    log_info("Multiple scores radiomics firstorder lasso (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", 
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Coxph Lasso all radiomics 32X
    model_name = paste0("32X_radiomics_full_lasso_", screening_method)
    cols_32X <- filter.gl(grep("^X32[0-9]{1}_original_", features, value = TRUE))
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics full lasso (32X)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", 
                                 screening_method = screening_method, parallel.method = parallel.method)

    # Coxph Lasso all radiomics 1320
    model_name = paste0("1320_radiomics_full_lasso_", screening_method)
    cols_1320 <- filter.gl(grep("^X1320_original_", features, value = TRUE))
    covariates = c(cols_1320, clinical_vars)
    log_info("Multiple scores radiomics full lasso (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso",
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else if (subdivision_type == "breasts") {
    # Coxph Lasso all radiomics of the two breasts (2413, 3413)
    model_name <- paste0("breasts_radiomics_full_lasso_", screening_method)
    cols_breasts <- filter.gl(grep("^X(2413|3413)_", features, value = TRUE))
    covariates <- c(cols_breasts, clinical_vars)
    log_info("Multiple scores radiomics full lasso (breasts: 2413, 3413)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso",
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else if (subdivision_type == "thorax") {
    # Coxph Lasso all radiomics of the thorax
    model_name <- paste0("thorax_radiomics_full_lasso_", screening_method)
    cols_thorax <- filter.gl(grep("^X(309|310|1320|1702|2413|3413|1601)_", features, value = TRUE))
    covariates <- c(cols_thorax, clinical_vars)
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso",
                                 screening_method = screening_method, parallel.method = parallel.method)
  } else {
    stop(paste("Subdivision type of features unrecognized:", subdivision_type))
  }

  log_info("Done")
  log_info(format(Sys.time() - start_time))
}

# Script args
options(error = quote({
      dump.frames(to.file=T, dumpto='last.dump')
        load('last.dump.rda')
        print(last.dump)
        q()
}))
args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
  Sys.sleep(20)
  config <- yaml.load_file(args[1])
  run_type <- args[2]
  subdivision_type <- args[3]
  analyzes_dir <- get.analyzes_dir_from_config(config)
  event_col <- config$EVENT_COL
  duration_col <- `if`(is.null(config$DURATION_COL), "survival_time_years", config$DURATION_COL)
  nb_estim <- as.numeric(config$NB_ESTIM_SCORE_MODELS)
  n_boot <- `if`(is.null(config$N_BOOTSTRAP), 200, as.numeric(config$N_BOOTSTRAP))
  log_threshold(INFO)
  if (run_type == "baseline_models") {
    multiple_scores_baseline_models(nb_estim, event_col, analyzes_dir, duration_col, subdivision_type)
  } else if (run_type == "cox_lasso_radiomics_all") {
    multiple_scores_cox_radiomics(nb_estim, "all", event_col, analyzes_dir, duration_col, subdivision_type)
  } else if (run_type == "cox_bootstrap_lasso_radiomics_all") {
    multiple_scores_cox_bootstrap_radiomics(nb_estim, "all", event_col, analyzes_dir, duration_col, 
                                            subdivision_type, n_boot)
  } else if (run_type == "cox_sis_radiomics_all") {
    multiple_scores_cox_sis_radiomics(nb_estim, "all", event_col, analyzes_dir, duration_col, subdivision_type)
  } else if (run_type == "cox_lasso_radiomics_features_hclust_corr") {
    multiple_scores_cox_radiomics(nb_estim, "features_hclust_corr", event_col, analyzes_dir, 
                                  duration_col, subdivision_type)
  } else if (run_type == "cox_bootstrap_lasso_radiomics_features_hclust_corr") {
    multiple_scores_cox_bootstrap_radiomics(nb_estim, "features_hclust_corr", event_col,
                                            analyzes_dir, duration_col, subdivision_type, n_boot)
  } else if (run_type == "cox_sis_radiomics_features_hclust_corr") {
    multiple_scores_cox_sis_radiomics(nb_estim, "features_hclust_corr", event_col, analyzes_dir, 
                                      duration_col, subdivision_type)
  } else {
    stop(paste("Run type unrecognized:", run_type))
  }
} else {
  print("No arguments provided. Skipping.")
}

