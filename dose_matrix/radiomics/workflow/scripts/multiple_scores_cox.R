
suppressPackageStartupMessages({
    library("yaml", quietly = TRUE)
    library("hms", quietly = TRUE)
})
options(show.error.locations = TRUE, error=traceback)

source("workflow/scripts/utils_cox.R")

multiple_scores_baseline_models <- function(nb_estim, event_col, analyzes_dir, duration_col) {
    dir.create(paste0(analyzes_dir, "coxph_R/"), showWarnings = FALSE)
    nworkers <- get.nworkers()
    cox_logfile <- paste0(analyzes_dir, "multiple_scores_baseline_models_R.log")
    if (file.exists(cox_logfile)) { file.remove(cox_logfile) }
    log_appender(appender_file(cox_logfile, append = TRUE))
    log_info(paste0("Multiple scores baseline models learning R (",nworkers," workers)"))
    start_time = Sys.time()
    # Dataset
    df_trainset0 <- read.csv(paste0(analyzes_dir, "datasets/trainset_0.csv.gz"), header = TRUE)
    clinical_vars <- get.clinical_features(colnames(df_trainset0), event_col, duration_col)
    index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
    parallel.method <- `if`(Sys.getenv("SLURM_NTASKS") == "", "mclapply", "rslurm")
    
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

    log_info("Done. Time:")
    log_info(format(Sys.time() - start_time))
}

multiple_scores_cox_radiomics <- function(nb_estim, file_features, event_col, 
                                          analyzes_dir, duration_col, suffix_model) {
    dir.create(paste0(analyzes_dir, "coxph_R/"), showWarnings = FALSE)
    nworkers <- get.nworkers()
    cox_logfile <- paste0(analyzes_dir, "multiple_scores_cox_lasso_radiomics_R_",suffix_model,".log")
    if (file.exists(cox_logfile)) { file.remove(cox_logfile) }
    log_appender(appender_file(cox_logfile, append = TRUE))
    log_info(paste0("Multiple scores cox lasso radiomics learning R (",nworkers," workers)"))
    start_time = Sys.time()
    # Dataset
    df_trainset0 <- read.csv(paste0(analyzes_dir, "datasets/trainset_0.csv.gz"), header = TRUE)
    features <- `if`(file_features == "all", colnames(df_trainset0), as.character(read.csv(file_features)[,1]))
    # Add "X" for R colname compatibility
    features <- as.character(lapply(features, function(x) { `if`(str_detect(substr(x, 1, 1), "[0-9]"), 
                                                                 paste("X", x, sep = ""), x) }))
    df_trainset0 <- df_trainset0[,features]
    clinical_vars <- get.clinical_features(colnames(df_trainset0), event_col, duration_col)
    index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
    parallel.method <- `if`(Sys.getenv("SLURM_NTASKS") == "", "mclapply", "rslurm")

    # Coxph Lasso radiomics firstorder 32X
    model_name = paste0("32X_radiomics_firstorder_lasso_", suffix_model)
    cols_32X <- grep("^X32[0-9]{1}_original_firstorder_", colnames(df_trainset0), value = TRUE)
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics firstorder lasso (32X)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", parallel.method = parallel.method)
   
    # Coxph Lasso radiomics firstorder 1320
    model_name = paste0("1320_radiomics_firstorder_lasso_", suffix_model)
    cols_1320 <- grep("^X1320_original_firstorder_", colnames(df_trainset0), value = TRUE)
    covariates = c(cols_1320, clinical_vars)
    log_info("Multiple scores radiomics firstorder lasso (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", parallel.method = parallel.method)

    # Coxph Lasso all radiomics 32X
    model_name = paste0("32X_radiomics_full_lasso_", suffix_model)
    cols_32X <- filter.gl(grep("^X32[0-9]{1}_original_", colnames(df_trainset0), value = TRUE))
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics full lasso (32X)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", parallel.method = parallel.method)
   
    # Coxph Lasso all radiomics 1320
    model_name = paste0("1320_radiomics_full_lasso_", suffix_model)
    cols_1320 <- filter.gl(grep("^X1320_original_", colnames(df_trainset0), value = TRUE))
    covariates = c(cols_1320, clinical_vars)
    log_info("Multiple scores radiomics full lasso (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,
                                 model_name, cox_logfile, penalty = "lasso", parallel.method = parallel.method)

    log_info("Done")
    log_info(format(Sys.time() - start_time))
}

multiple_scores_cox_bootstrap_radiomics <- function(nb_estim, file_features, event_col, 
                                                    analyzes_dir, duration_col, suffix_model, n_boot) {
    dir.create(paste0(analyzes_dir, "coxph_R/"), showWarnings = FALSE)
    nworkers <- get.nworkers()
    cox_logfile <- paste0(analyzes_dir, "multiple_scores_cox_bootstrap_lasso_radiomics_R_",suffix_model,".log")
    if (file.exists(cox_logfile)) { file.remove(cox_logfile) }
    log_appender(appender_file(cox_logfile, append = TRUE))
    log_info(paste0("Multiple scores cox bootstrap lasso radiomics learning R (",nworkers," workers)"))
    start_time = Sys.time()
    # Dataset
    df_trainset0 <- read.csv(paste0(analyzes_dir, "datasets/trainset_0.csv.gz"), header = TRUE)
    features <- `if`(file_features == "all", colnames(df_trainset0), as.character(read.csv(file_features)[,1]))
    # Add "X" for R colname compatibility
    features <- as.character(lapply(features, function(x) { `if`(str_detect(substr(x, 1, 1), "[0-9]"), 
                                                                 paste("X", x, sep = ""), x) }))
    df_trainset0 <- df_trainset0[,features]
    clinical_vars <- get.clinical_features(colnames(df_trainset0), event_col, duration_col)
    index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
    parallel.method <- `if`(Sys.getenv("SLURM_NTASKS") == "", "mclapply", "rslurm")

    # Coxph Bootstrap Lasso radiomics firstorder 32X
    model_name = paste0("32X_radiomics_firstorder_bootstrap_lasso_", suffix_model)
    cols_32X <- grep("^X32[0-9]{1}_original_firstorder_", colnames(df_trainset0), value = TRUE)
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics firstorder bootstrap lasso (32X)")
    print("Before the call of parallel_multiple_scores_cox(), I print n_boot")
    print(paste("n_boot:", n_boot))
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir, model_name, cox_logfile, 
                                 penalty = "bootstrap_lasso", parallel.method = parallel.method, n_boot = n_boot)
   
    # Coxph Bootstrap Lasso radiomics firstorder 1320
    model_name = paste0("1320_radiomics_firstorder_bootstrap_lasso_", suffix_model)
    cols_1320 <- grep("^X1320_original_firstorder_", colnames(df_trainset0), value = TRUE)
    covariates = c(cols_1320, clinical_vars)
    log_info("Multiple scores radiomics firstorder bootstrap lasso (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir,model_name, cox_logfile, 
                                 penalty = "bootstrap_lasso", parallel.method = parallel.method, n_boot = n_boot)

    # Coxph Bootstrap Lasso all radiomics 32X
    model_name = paste0("32X_radiomics_full_bootstrap_lasso_", suffix_model)
    cols_32X <- filter.gl(grep("^X32[0-9]{1}_original_", colnames(df_trainset0), value = TRUE))
    covariates = c(cols_32X, clinical_vars)
    log_info("Multiple scores radiomics full bootstrap lasso (32X)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir, model_name, cox_logfile, 
                                 penalty = "bootstrap_lasso", parallel.method = parallel.method, n_boot = n_boot)
   
    # Coxph Bootstrap Lasso all radiomics 1320
    model_name = paste0("1320_radiomics_full_bootstrap_lasso_", suffix_model)
    cols_1320 <- filter.gl(grep("^X1320_original_", colnames(df_trainset0), value = TRUE))
    covariates = c(cols_1320, clinical_vars)
    log_info("Multiple scores radiomics full bootstrap lasso (1320)")
    parallel_multiple_scores_cox(nb_estim, covariates, event_col, duration_col, analyzes_dir, model_name, cox_logfile, 
                                 penalty = "bootstrap_lasso", parallel.method = parallel.method, n_boot = n_boot)

    log_info("Done")
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
    n_boot <- `if`(is.null(config$N_BOOTSTRAP), 200, as.numeric(config$N_BOOTSTRAP))
    print("In the main, I print n_boot")
    print(paste("n_boot:", n_boot))
    file_trainset = paste0(analyzes_dir, "datasets/trainset.csv.gz")
    file_testset = paste0(analyzes_dir, "datasets/testset.csv.gz")
    file_features <- "all"
    log_threshold(INFO)
    if (run_type == "baseline_models") {
        multiple_scores_baseline_models(nb_estim, event_col, analyzes_dir, duration_col)
    } else if (run_type == "cox_lasso_radiomics_all") {
        multiple_scores_cox_radiomics(nb_estim, file_features, event_col, analyzes_dir, duration_col, "all")
    } else if (run_type == "cox_bootstrap_lasso_radiomics_all") {
        print("Before the call of multiple_scores_cox_bootstrap_radiomics(), I print n_boot")
        print(paste("n_boot:", n_boot))
        multiple_scores_cox_bootstrap_radiomics(nb_estim, file_features, event_col, analyzes_dir, 
                                                duration_col, "all", n_boot)
    } else if (run_type == "cox_lasso_radiomics_features_hclust_corr") {
        file_features <- paste0(analyzes_dir, "screening/features_hclust_corr.csv")
        multiple_scores_cox_radiomics(nb_estim, file_features, event_col, 
                                      analyzes_dir, duration_col, "features_hclust_corr")
    } else if (run_type == "cox_bootstrap_lasso_radiomics_features_hclust_corr") {
        file_features <- paste0(analyzes_dir, "screening/features_hclust_corr.csv")
        multiple_scores_cox_bootstrap_radiomics(nb_estim, file_features, event_col, 
                                                analyzes_dir, duration_col, "features_hclust_corr", n_boot)
    } else {
        stop(paste("Run type unrecognized:", run_type))
    }
} else {
    print("No arguments provided. Skipping.")
}

