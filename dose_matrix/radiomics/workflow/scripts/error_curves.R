
options(show.error.locations = TRUE, error=traceback)
require("patchwork")

source("workflow/scripts/utils_cox.R")
source("workflow/scripts/utils_rsf.R")

# Undo the parallelism in the model's call: we already run bootstrap samples in parallel
undo_parallelism <- function(model_call) {
  if (model_call[[1]] == "selection.coxnet")
    model_call$cv.parallel = F
  if (model_call[[1]] == "cv.glmnet")
    model_call$parallel = F
  model_call
}

# Task of error estimation for one bootstrap sample
sample.estim.error <- function(data, indices, list_models, ipcw.formula, times) {
  boot_train <- data[indices, ]
  boot_test <- data[-indices, ]
  duration_col <- all.vars(ipcw.formula[[2]])[1]
  event_col <- all.vars(ipcw.formula[[2]])[2]
  surv_y_boot_test <- survival::Surv(boot_test[[duration_col]], boot_test[[event_col]])
  idx_surv <- length(times)
  lapply(list_models, function(model) {
    # Create a model call with proper data and filtered covariates
    boot_call <- undo_parallelism(rlang::duplicate(model$call))
    boot_call$data <- boot_train
    filtered_covariates <- preliminary_filter(boot_train, model$full_covariates, event_col, 
                                              model$screening_method, "", model$analyzes_dir)
    filtered_formula_model <- get.surv.formula(event_col, filtered_covariates, duration_col = duration_col)
    boot_call$formula <- filtered_formula_model
    boot_model <- eval(boot_call)
    na_coefs <- get.na.coefs(boot_model)
    while (length(na_coefs) > 0) {
      warning(paste0(paste("NA value:", na_coefs, collapse = ""), 
                           ". Discarding these variables and re-fitting ", typeof(boot_model), "."))
      filtered_covariates <- filtered_covariates[!(filtered_covariates %in% na_coefs)]
      filtered_formula_model <- get.surv.formula(event_col, filtered_covariates, duration_col = duration_col)
      boot_call$formula <- filtered_formula_model
      boot_model <- eval(boot_call)
      na_coefs <- get.na.coefs(boot_model)
    }
    # Error estimation
    survprob.boot <- predictSurvProb(boot_model, newdata = data.table::as.data.table(boot_test), times = times)
    harrell.cindex <- Hmisc::rcorr.cens(survprob.boot[,idx_surv], S = surv_y_boot_test)[["C Index"]]
    ipcw.cindexes <- pec::cindex(list("Boot model" = as.matrix(survprob.boot)),
                                 formula = ipcw.formula, data = boot_test, 
                                 cens.model = "cox", eval.times = times)$AppCindex[["Boot model"]]
    perror <- pec::pec(list("Boot model" = survprob.boot), formula = ipcw.formula, data = boot_test, 
                       cens.model = "cox", times = times, start = times[1], exact = F, reference = F)
    bs.times <- perror$AppErr[["Boot model"]]
    ibs <- pec::crps(perror)[1][["Boot model"]]
    list("Harrell's C-index" = harrell.cindex, "IPCW C-index tau" = ipcw.cindexes, 
         "Brier score tau" = bs.times, "IBS" = ibs)
  })
}

# Task for a slurm job: several sample.estim.error calls
slurm_job_boot_estim_error <- function(nb_estim, data, list_models, ipcw.formula, times) {
  listResBoot <- foreach(i = 1:nb_estim, .errorhandling = "pass") %do% {
    idx_boot <- sample(nrow(data), replace = T)
    sample.estim.error(data, idx_boot, list_models, ipcw.formula, times)
  }
  # Converts the result to have one list of models, where each element is a model
  # that contains the metrics of all bootstrap samples
  resBoot <- do.call(function (...) Map(cbind, ...), listResBoot)
  return(resBoot)
}

# Bootstrap error estimation
bootstrap.estim.error <- function(data, ipcw.formula, list_models, analyzes_dir, cens.model = "cox", 
                                  B = 100, times = seq(1, 60, by = 1), logfile = NULL, error_estim_per_job = 10,
                                  boot.parallel = "foreach", boot.ncpus = min(get.nworkers(), B)) {
  stopifnot(boot.parallel %in% c("foreach", "rslurm"))
  if (!is.null(logfile)) log_appender(appender_file(logfile, append = TRUE))
  else log_appender(appender_stdout)
  dir.create(paste0(analyzes_dir, "error_curves/"), showWarnings = FALSE)
  log_info(paste("Boot parallel method:", boot.parallel))
  log_info("Setting rfsrc parallel options to 1")
  old_rf.cores <- getOption("rf.cores")
  old_mc.cores <- getOption("mc.cores")
  options(rf.cores = 1, mc.cores = 1)
  covariates <- all.vars(ipcw.formula[[3]])
  duration_col <- all.vars(ipcw.formula[[2]])[1]
  event_col <- all.vars(ipcw.formula[[2]])[2]
  functions_to_export <- c("sample.estim.error", "selection.coxnet", "predictSurvProb.selection.coxnet", 
                           "undo_parallelism", "predictSurvProb", "coxlasso_data", "get.coefs.cox", "get.na.coefs",
                           "bootstrap.coxnet", "predictSurvProb.bootstrap.coxnet", "slurm_job_boot_estim_error",
                           "get.surv.formula", "get.best.lambda", "preliminary_filter", "filter_dummies_iccc")
  if (boot.parallel == "foreach") {
    log_info(paste("Bootstrap error estimation: creating a cluster with", boot.ncpus, "workers"))
    cl <- parallel::makeCluster(boot.ncpus, outfile = "/dev/stdout")
    doParallel::registerDoParallel(cl)
    log_info(paste(colnames(showConnections()), collapse = " "))
    log_info(paste(apply(showConnections(), 1, 
                         function(row) { paste(paste(as.character(row), collapse = " ")) }), collapse = "\n"))
    listResBoot <- foreach(i = 1:B, .export = functions_to_export,
                           .packages = c("survival", "randomForestSRC", "logger", "prodlim")) %dopar% {
      idx_boot <- sample(nrow(data), replace = T)
      sample.estim.error(data, idx_boot, list_models, ipcw.formula, times)
    }
    # Converts the result to have one list of models, where each element is a model
    # that contains the metrics of all bootstrap samples
    resBoot <- do.call(function (...) Map(cbind, ...), listResBoot)
    parallel::stopCluster(cl)
  } else if (boot.parallel == "rslurm") {
    nb_estim_jobs <- rep(error_estim_per_job, B %/% error_estim_per_job)
    if (B %% error_estim_per_job > 0) nb_estim_jobs <- c(nb_estim_jobs, B %% error_estim_per_job)
    log_info(paste("Number of slurm jobs:", length(nb_estim_jobs)))
    log_info(do.call(paste, as.list(nb_estim_jobs)))
    sopt <- list(time = "02:00:00", "ntasks" = 1, "cpus-per-task" = 1, 
                 partition = "cpu_med", mem = "20G")
    sjob <- slurm_apply(function(nb_estim) slurm_job_boot_estim_error(nb_estim, data, list_models, ipcw.formula, times),
                        data.frame(nb_estim = nb_estim_jobs), 
                        nodes = length(nb_estim_jobs), cpus_per_node = 1, processes_per_node = 1, 
                        global_objects = functions_to_export, 
                        slurm_options = sopt)
    log_info("Jobs are submitted")
    listResJobs <- get_slurm_out(sjob, outtype = "raw", wait = T)
    # Converts the result to have one list of models, where each element is a model
    # that contains the metrics of all bootstrap samples
    resBoot <- do.call(function (...) Map(cbind, ...), listResJobs) 
    cleanup_files(sjob, wait = T)
    log_info("End of all submitted jobs")
  }
  log_info("Setting back rfsrc parallel options to old ones")
  options(rf.cores = old_rf.cores, mc.cores = old_mc.cores)
  # For each model, we convert the stored metrics from list to matrix
  resBoot <- Map(function(model_metrics) {
                 new_model_metrics <- apply(model_metrics, 1, function(list_metric) {
                                              matrix_metric <- do.call(rbind, list_metric)
                                              matrix_metric
                                            })
                 stopifnot({
                   nrow(new_model_metrics[["Harrell's C-index"]]) == B
                   nrow(new_model_metrics[["IPCW C-index tau"]]) == B
                   nrow(new_model_metrics[["Brier score tau"]]) == B
                   nrow(new_model_metrics[["IBS"]]) == B
                 })
                 new_model_metrics$pred.times <- times
                 new_model_metrics
                 }, resBoot)
  out <- list("bootstrap_errors" = resBoot)
  class(out) <- "bootstrap.estim.error"
  out
}

# Get color according to model
get_color_model <- function(pretty_model_name) {
  if (stringr::str_detect(pretty_model_name, "Cox Lasso")) return("blue")
  if (stringr::str_detect(pretty_model_name, "Cox Bootstrap Lasso")) return("purple")
  if (stringr::str_detect(pretty_model_name, "Cox mean dose")) return("green")
  if (stringr::str_detect(pretty_model_name, "Cox doses-volumes")) return("orange")
  if (stringr::str_detect(pretty_model_name, "RSF ")) return("red")
}

# Plot time-dependant error curves
plot_error_curves <- function(resBoot, analyzes_dir) {
  df_plot <- lapply(names(resBoot$bootstrap_errors), function (model_name) {
                    model_errors <- resBoot$bootstrap_errors[[model_name]]
                    matrix_ipcw_cindex <- model_errors[["IPCW C-index tau"]]
                    matrix_bs <- model_errors[["Brier score tau"]]
                    data.frame(mean_ipcw_cindex = colMeans(matrix_ipcw_cindex),
                               std_ipcw_cindex = apply(matrix_ipcw_cindex, 2, sd),
                               mean_bs = colMeans(matrix_bs),
                               std_bs = apply(matrix_bs, 2, sd),
                               times = model_errors$pred.times,
                               model_name = model_name,
                               pretty_model_name = resBoot$pretty_model_names[[model_name]])
                  })
  df_plot <- as.data.frame(do.call("rbind", df_plot))
  save_results_dir <- paste0(analyzes_dir, "plots/")
  colors_models <- as.character(lapply(resBoot$pretty_model_names, get_color_model))
  # Plot IPCW C-index curve
  cindex_plot <- ggplot(df_plot, aes(x = times, y = mean_ipcw_cindex, color = pretty_model_name)) + geom_line() + 
                 geom_ribbon(aes(ymin = mean_ipcw_cindex - std_ipcw_cindex, ymax = mean_ipcw_cindex + std_ipcw_cindex, 
                                 fill = pretty_model_name), linetype = 0, alpha = 0.2, show.legend = F) +
                 scale_color_manual(breaks = resBoot$pretty_model_names, values = colors_models) +
                 scale_fill_manual(breaks = resBoot$pretty_model_names, values = colors_models) +
                 ylim(0.4, NA) +
                 labs(x = "Time (years)", y = "Bootstrap mean of IPCW C-index", color = "Model name") +
                 theme(aspect.ratio = 1.5)
  ggsave(paste0(save_results_dir, "error_curve_ipcw_cindex.png"), plot = cindex_plot, device = "png", dpi = 480)
  # Plot brier score curve
  bs_plot <- ggplot(df_plot, aes(x = times, y = mean_bs, color = pretty_model_name)) + geom_line() + 
             geom_ribbon(aes(ymin = mean_bs - std_bs, ymax = mean_bs + std_bs, 
                             fill = pretty_model_name), linetype = 0, alpha = 0.2, show.legend = F) +
             scale_color_manual(breaks = resBoot$pretty_model_names, values = colors_models) +
             scale_fill_manual(breaks = resBoot$pretty_model_names, values = colors_models) +
             ylim(0, NA) +
             labs(x = "Time (years)", y = "Bootstrap mean of Brier score", color = "Model name") +
             theme(aspect.ratio = 1.5)
  ggsave(paste0(save_results_dir, "error_curve_brier_score.png"), plot = bs_plot, device = "png", dpi = 480)
  full_plots <- cindex_plot + bs_plot + patchwork::plot_layout(guides = "collect") & 
                theme(legend.position = "top", aspect.ratio = 2) & guides(color = guide_legend(nrow = 3, byrow = T))
  ggsave(paste0(save_results_dir, "error_curves.png"), plot = full_plots, device = "png", dpi = 480)
}

# Curve estimation error for several models
pec_estimation <- function(analyzes_dir, event_col, duration_col, n_boot = 100, logfile = NULL) {
    if (!is.null(logfile)) log_appender(appender_file(logfile, append = TRUE))
    else log_appender(appender_stdout)
    file_dataset = paste0(analyzes_dir, "datasets/dataset.csv.gz")
    nworkers <- get.nworkers()
    options(rf.cores = 1, mc.cores = 1)
    dir.create(paste0(analyzes_dir, "plots/"), showWarnings = FALSE)
    dir.create(paste0(analyzes_dir, "error_curves/"), showWarnings = FALSE)
    df_dataset <- read.csv(file_dataset, header = T)

    # # Feature elimination
    # file_features_hclust_corr <- paste0(analyzes_dir, "screening/features_hclust_corr.csv")
    # features_hclust_corr <- as.character(read.csv(file_features_hclust_corr)[,1])
    # features_hclust_corr <- as.character(lapply(features_hclust_corr, function(x) { 
    #                                             `if`(str_detect(substr(x, 1, 1), "[0-9]"), paste0("X", x), x) }))
    # Covariables models
    # cols_lasso_1320 - cols_lasso_1320[cols_lasso_1320 %in% features_hclust_corr] # specific to this model
    clinical_vars <- get.clinical_features(colnames(df_dataset), event_col, duration_col)
    cols_dv <- grep("dv_\\w+_1320", colnames(df_dataset), value = TRUE)
    cols_rsf_1320_firstorder <- filter.gl(grep("^X1320_original_firstorder", colnames(df_dataset), value = TRUE))
    cols_lasso_32X_full <- filter.gl(grep("^X32[0-9]_original_", colnames(df_dataset), value = TRUE))
    cols_boot_lasso_32X <- filter.gl(grep("^X32[0-9]_original_", colnames(df_dataset), value = TRUE))
    covariates_all <- unique(c(clinical_vars, "X1320_original_firstorder_Mean", 
                               cols_dv, cols_rsf_1320_firstorder, cols_lasso_32X_full, cols_boot_lasso_32X))
    df_dataset <- df_dataset[, c(event_col, duration_col, covariates_all)]
    df_dataset <- na.omit(df_dataset)
    log_info(paste("Dataset rows:", nrow(df_dataset)))

    # Model Cox mean dose
    model_name_coxmean <- "1320_mean"
    covariates_coxmean <- c("X1320_original_firstorder_Mean", clinical_vars)
    formula_model_coxmean <- get.surv.formula(event_col, covariates_coxmean, duration_col = duration_col)
    coxmean <- coxph(formula_model_coxmean, data = df_dataset, x = TRUE, y = TRUE)
    coxmean$call$formula <- formula_model_coxmean
    coxmean$full_covariates <- covariates_coxmean
    coxmean$screening_method <- "all"

    # Model Cox doses-volumes
    model_name_coxdv <- "1320_dosesvol"
    covariates_coxdv <- c(cols_dv, clinical_vars)
    formula_model_coxdv <- get.surv.formula(event_col, covariates_coxdv, duration_col = duration_col)
    coxdv <- coxph(formula_model_coxdv, data = df_dataset, x = T, y = T)
    coxdv$call$formula <- formula_model_coxdv
    coxdv$full_covariates <- covariates_coxdv
    coxdv$screening_method <- "all"

    # Model rsf 1320 radiomics firstorder with screening
    model_name_rsf <- paste0("1320_radiomics_firstorder_features_hclust_corr")
    covariates_rsf_1320 <- c(clinical_vars, cols_rsf_1320_firstorder)
    formula_model_rsf <- get.surv.formula(event_col, covariates_rsf_1320, duration_col = duration_col)
    id_max_cv <- which.max(read.csv(paste0(analyzes_dir, "rsf/", model_name_rsf,
                                           "/5_runs_full_test_metrics.csv"), row.names = 1)[2,])
    rsf_1320_params.best <<- read.csv(paste0(analyzes_dir, "rsf/", model_name_rsf, "/",id_max_cv-1,"/cv.csv"))[1,]
    rsf_1320 <- rfsrc(formula_model_rsf, data = df_dataset, ntree = rsf_1320_params.best$ntree, 
                      nodesize = rsf_1320_params.best$nodesize, nsplit = rsf_1320_params.best$nsplit)
    rsf_1320$full_covariates <- covariates_rsf_1320
    rsf_1320$screening_method <- "features_hclust_corr"

    # Model Cox Lasso 32X full no screening 
    model_name_lasso_32X <- "32X_radiomics_full_lasso_all"
    covariates_lasso_32X <- c(cols_lasso_32X_full, clinical_vars)
    formula_model_lasso_32X <- get.surv.formula(event_col, covariates_lasso_32X, duration_col = duration_col)
    id_max_cv <- which.max(read.csv(paste0(analyzes_dir, "coxph_R/", model_name_lasso_32X, 
                                           "/5_runs_full_test_metrics.csv"), row.names = 1)[2,])
    lasso_32X_list.lambda <- read.csv(paste0(analyzes_dir, "coxph_R/", model_name_lasso_32X, "/",
                                             id_max_cv-1,"/path_lambda.csv"))[["lambda"]]
    coxlasso_32X <- selection.coxnet(formula_model_lasso_32X, data = df_dataset, 
                                     list.lambda = lasso_32X_list.lambda, type.measure = "C")
    coxlasso_32X$full_covariates <- covariates_lasso_32X
    coxlasso_32X$screening_method <- "all"

    # Model Cox bootstrap Lasso 32X with screening
    model_name_boot_lasso_32X <- "32X_radiomics_full_bootstrap_lasso_features_hclust_corr"
    covariates_boot_lasso_32X <- c(cols_boot_lasso_32X, clinical_vars)
    formula_model_boot_lasso_32X <- get.surv.formula(event_col, covariates_boot_lasso_32X, duration_col = duration_col)
    id_max_cv <- which.max(read.csv(paste0(analyzes_dir, "coxph_R/", model_name_boot_lasso_32X, 
                                           "/5_runs_full_test_metrics.csv"), row.names = 1)[2,])
    selected_features <- read.csv(paste0(analyzes_dir, "coxph_R/", model_name_boot_lasso_32X, 
                                         "/",id_max_cv-1,"/final_selected_features.csv"))[["selected_features"]]
    bootstrap_selected_features <- read.csv(paste0(analyzes_dir, "coxph_R/", model_name_boot_lasso_32X, 
                                                   "/",id_max_cv-1,"/bootstrap_selected_features.csv"), header = T)
    coxbootlasso_32X <- bootstrap.coxnet(data = df_dataset, formula = formula_model_boot_lasso_32X, pred.times,
                                         best.lambda.method = "lambda.1se", selected_features = selected_features, 
                                         bootstrap_selected_features = bootstrap_selected_features)
    coxbootlasso_32X$full_covariates <- covariates_boot_lasso_32X
    coxbootlasso_32X$screening_method <- "features_hclust_corr"
   
    # Self-made pec estim
    formula_ipcw = get.ipcw.surv.formula(event_col, colnames(df_dataset), duration_col = duration_col)
    pred.times <- seq(1, 60, 1)
    pec_M = floor(0.7 * nrow(df_dataset))
    compared_models <- list("Cox mean dose" = coxmean,
                            "Cox doses-volumes" = coxdv,
                            "Cox Lasso heart's subparts dosiomics" = coxlasso_32X,
                            "Cox Bootstrap Lasso \nheart's subparts dosiomics with screening" = coxbootlasso_32X,
                            "RSF whole-heart first-order dosiomics" = rsf_1320
                            )
    pretty_model_names <- as.list(names(compared_models))
    names(compared_models) <- c(model_name_coxmean, model_name_coxdv, model_name_lasso_32X,
                                model_name_boot_lasso_32X, model_name_rsf)
    names(pretty_model_names) <- names(compared_models)
    
    # We eval the calls' arguments because the corresponding variables won't be available in the functions
    compared_models <- lapply(compared_models, function(model) {
            model$call$formula <- eval(model$call$formula)
            model$analyzes_dir <- analyzes_dir
            if (model$call[[1]] == "selection.coxnet")  {
              model$call$list.lambda <- eval(model$call$list.lambda)
              model$call$list.lambda <- eval(model$call$list.lambda)
            }
            if (model$call[[1]] == "bootstrap.coxnet") {
              model$call$pred.times <- eval(model$call$pred.times)
              model$call$selected_features <- eval(model$call$selected_features)
              model$call$bootstrap_selected_features <- eval(model$call$bootstrap_selected_features)
            }
            if (model$call[[1]] == "rfsrc") {
              model$call$ntree <- eval(model$call$ntree)
              model$call$nodesize <- eval(model$call$nodesize)
              model$call$nsplit <- eval(model$call$nsplit)
            }
            model
          })
    
    # Self-made bootstrap error estimation
    log_info(paste("n_boot =", n_boot))
    boot.parallel <- `if`(Sys.getenv("SLURM_NTASKS") == "", "foreach", "rslurm")
    boot_error <- bootstrap.estim.error(df_dataset, formula_ipcw, compared_models, analyzes_dir, 
                                        B = n_boot, times = pred.times, 
                                        logfile = NULL, boot.parallel = boot.parallel)
    boot_error$pretty_model_names <- pretty_model_names
    saveRDS(boot_error, paste0(analyzes_dir, "error_curves/boot_error.rds"))
    plot_error_curves(boot_error, analyzes_dir)
    boot_error
}

args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    analyzes_dir <- args[1]
    event_col <- args[2]
    duration_col <- `if`(length(args) == 3, args[3], "survival_time_years")
    log_threshold(INFO)
    pec_estimation(analyzes_dir, event_col, duration_col)
} else {
    print("No arguments provided. Skipping.")
}

