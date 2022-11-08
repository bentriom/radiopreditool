
options(show.error.locations = TRUE, error=traceback)

source("workflow/scripts/utils_cox.R")
source("workflow/scripts/utils_rsf.R")

# glmnet.pec <- function(formula, data, family = "cox", alpha = 1, type.measure = "C", 
#                        lambda = NULL, best.lambda = "lambda.min", ...) {
#     covariates <- all.vars(formula[[3]])
#     duration_col <- all.vars(formula[[2]])[1]
#     event_col <- all.vars(formula[[2]])[2]
#     if (!("data.table" %in% class(data))) {
#         data_lasso <- coxlasso_data(data, covariates, event_col, duration_col)
#         X <- data_lasso$X
#         surv_y <- data_lasso$surv_y
#     } else {
#         X <- as.matrix(data[, ..covariates])
#         surv_y <- Surv(as.numeric(unlist(data[, ..duration_col])), as.numeric(unlist(data[, ..event_col])))
#     }
#     fit <- glmnet(X, surv_y, family = "cox", alpha = 1, lambda = lambda, type.measure = "C", ...)
#     out <- list("glmnet.fit" = fit, "call" = match.call(), "X" = X, "surv_y" = surv_y, 
#                 "best.lambda" = best.lambda, "covariates" = covariates)
#     class(out) <- "glmnet.pec"
#     out
# }
# 
# predictSurvProb.glmnet.pec <- function(object, newdata, times) {
#     covariates <- object$covariates
#     newdata_model <- newdata[, ..covariates]
#     # saveRDS(newdata, file = "newdata.rds")
#     # saveRDS(object$covariates, file = "covariates.rds")
#     # saveRDS(newdata_model, file = "newdata_model.rds")
#     cvcoxlasso.survfit <- survfit(object$glmnet.fit, x = object$X, y = object$surv_y, 
#                                   newx = as.matrix(newdata_model), s = object$best.lambda)
#     t(summary(cvcoxlasso.survfit, times = times)$surv)
# }

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
    # boot_call <- undo_parallelism(rlang::duplicate(model$call))
    boot_call <- undo_parallelism(rlang::duplicate(model$call))
    boot_call$data <- boot_train
    boot_model <- eval(boot_call)
    survprob.boot <- predictSurvProb(boot_model, newdata = data.table::as.data.table(boot_test), times = times)
    harrell.cindex <- Hmisc::rcorr.cens(survprob.boot[,idx_surv], S = surv_y_boot_test)[["C Index"]]
    ipcw.cindexes <- pec::cindex(list("Boot model" = as.matrix(survprob.boot)),
                                 formula = ipcw.formula, data = boot_test, 
                                 cens.model = "cox", eval.times = times)$AppCindex[["Boot model"]]
    perror <- pec::pec(list("Boot model" = survprob.boot), formula = ipcw.formula, data = boot_test, 
                       cens.model = "cox", times = times, start = times[1], exact = F, reference = F)
    bs.times <- perror$AppErr[["Boot model"]]
    ibs <- crps(perror)[1][["Boot model"]]
    list("Harrell's C-index" = harrell.cindex, "IPCW C-index tau" = ipcw.cindexes, 
         "Brier score tau" = bs.times, "IBS" = ibs)
  })
}

# Task for a slurm job: several sample.estim.error calls
slurm_job_boot_estim_error <- function(nb_estim, data, list_models, ipcw.formula, times) {
  resBoot <- foreach(i = 1:nb_estim, .combine = 'list') %do% {
    idx_boot <- sample(nrow(data), replace = T)
    sample.estim.error(data, idx_boot, list_models, ipcw.formula, times)
  }
  # Converts the result to have one list of models, where each element is a model
  # that contains the metrics of all bootstrap samples
  resBoot <- do.call(function (...) Map(cbind, ...), resBoot)
  return(resBoot)
}

# Bootstrap error estimation
bootstrap.estim.error <- function(data, ipcw.formula, list_models, analyzes_dir, cens.model = "cox", 
                                  B = 100, times = seq(1, 60, by = 1), logfile = NULL, 
                                  boot.parallel = "foreach", boot.ncpus = min(get.nworkers(), B), 
                                  estim_per_job = 5) {
  stopifnot(boot.parallel %in% c("foreach", "rslurm"))
  if (!is.null(logfile)) log_appender(appender_file(logfile, append = TRUE))
  if (is.null(logfile)) log_appender(appender_stdout)
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
                           "undo_parallelism", "predictSurvProb", "coxlasso_data", "get.coefs.cox", 
                           "get.surv.formula", "get.best.lambda", "prodlim::Hist")
  if (boot.parallel == "foreach") {
    log_info(paste("Bootstrap error estimation: creating a cluster with", boot.ncpus, "workers"))
    cl <- parallel::makeCluster(boot.ncpus, outfile = "/dev/stdout")
    doParallel::registerDoParallel(cl)
    log_info(paste(colnames(showConnections()), collapse = " "))
    log_info(paste(apply(showConnections(), 1, 
                         function(row) { paste(paste(as.character(row), collapse = " ")) }), collapse = "\n"))
    listResBoot <- foreach(i = 1:B, .export = functions_to_export, 
                       .packages = c("survival", "randomForestSRC", "logger")) %do% {
      idx_boot <- sample(nrow(data), replace = T)
      sample.estim.error(data, idx_boot, list_models, ipcw.formula, times)
    }
    # Converts the result to have one list of models, where each element is a model
    # that contains the metrics of all bootstrap samples
    resBoot <- do.call(function (...) Map(cbind, ...), listResBoot)
    parallel::stopCluster(cl)
  } else if (boot.parallel == "rslurm") {
    estim_per_job <- 5
    nb_estim_jobs <- rep(estim_per_job, B %/% estim_per_job)
    if (B %% estim_per_job > 0) nb_estim_jobs <- c(nb_estim_jobs, B %% estim_per_job)
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
                                              stopifnot(nrow(matrix_metric) == B)
                                              matrix_metric
                                            })
                 new_model_metrics$pred.times <- times
                 new_model_metrics
                 }, resBoot)
  out <- list("bootstrap_errors" = resBoot)
  class(out) <- "bootstrap.estim.error"
  out
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
                               model_name = model_name)
                  })
  df_plot <- as.data.frame(do.call("rbind", df_plot))
  save_results_dir <- paste0(analyzes_dir, "error_curves/")
  # Plot IPCW C-index curve
  ggplot(df_plot, aes(x = times, y = mean_ipcw_cindex, color = model_name)) + geom_line() + 
  geom_ribbon(aes(ymin = mean_ipcw_cindex - std_ipcw_cindex, ymax = mean_ipcw_cindex + std_ipcw_cindex, 
                  fill = model_name), linetype = 0, alpha = 0.2, show.legend = F) +
  ylim(0, NA) +
  labs(x = "Time (years)", y = "Bootstrap mean of IPCW C-index", color = "Model name")
  ggsave(paste0(save_results_dir, "ipcw_cindex.png"), device = "png", dpi = 480)
  # Plot brier score curve
  ggplot(df_plot, aes(x = times, y = mean_bs, color = model_name)) + geom_line() + 
  geom_ribbon(aes(ymin = mean_bs - std_bs, ymax = mean_bs + std_bs, 
                  fill = model_name), linetype = 0, alpha = 0.2, show.legend = F) +
  ylim(0, NA) +
  labs(x = "Time (years)", y = "Bootstrap mean of Brier score", color = "Model name")
  ggsave(paste0(save_results_dir, "brier_score.png"), device = "png", dpi = 480)
}

# plot_ipcw_cindex_model <- function(model_errors) {
#   matrix_ipcw_cindex <- model_errors[["IPCW C-index tau"]]
#   df_plot_ipcw_cindex = data.frame(mean_score = colMeans(matrix_ipcw_cindex),
#                                    std_score = apply(matrix_ipcw_cindex, 2, sd),
#                                    times = model_errors$pred.times)
#   ggplot(df_plot_ipcw_cindex, aes(x = times, y = mean_score)) +
#   geom_ribbon(aes(ymin = mean_score - std_score, ymax = mean_score + std_score), fill = "blue", alpha = 0.5) +
#   geom_line(color = "red") + 
#   geom_point(data = df_plot_ipcw_cindex, aes(x = times, y = mean_score), color = "purple") +
#   ylim(0, NA) +
#   labs(x = "Time (years)", y = "Bootstrap mean of IPCW C-index")
# }
# 
# plot_brier_score_model <- function(model_errors) {
#   matrix_bs <- model_errors[["Brier score tau"]]
#   df_plot_bs = data.frame(mean_score = colMeans(matrix_bs), 
#                           std_score = apply(matrix_bs, 2, sd),
#                           times = model_errors$pred.times)
#   # Brier score curve
#   ggplot(df_plot_bs, aes(x = times, y = mean_score)) +
#   geom_ribbon(aes(ymin = mean_score - std_score, ymax = mean_score + std_score), alpha = 0.5) +
#   geom_line() + 
#   geom_point(data = df_plot_bs, aes(x = times, y = mean_score)) +
#   ylim(0, NA) +
#   labs(x = "Time (years)", y = "Bootstrap mean of Brier score")
# }

# Curve estimation error for several models
pec_estimation <- function(file_dataset, event_col, analyzes_dir, duration_col, B = 200) {
    nworkers <- get.nworkers()
    options(rf.cores = 1, mc.cores = 1)
    dir.create(paste0(analyzes_dir, "pec_plots/"), showWarnings = FALSE)
    dir.create(paste0(analyzes_dir, "pec_results/"), showWarnings = FALSE)
    df_dataset <- read.csv(file_dataset, header = T)

    # Feature elimination
    file_features_hclust_corr <- paste0(analyzes_dir, "features_hclust_corr.csv")
    features_hclust_corr <- as.character(read.csv(file_features_hclust_corr)[,1])
    features_hclust_corr <- as.character(lapply(features_hclust_corr, function(x) { 
                                                `if`(str_detect(substr(x, 1, 1), "[0-9]"), paste0("X", x), x) }))
    # Covariables models
    cols_rsf_1320 <- filter.gl(grep("^X1320_original_", colnames(df_dataset), value = TRUE))
    cols_lasso_1320 <- filter.gl(grep("^X1320_original_firstorder_", colnames(df_dataset), value = TRUE))
    cols_lasso_1320 <- cols_lasso_1320[cols_lasso_1320 %in% features_hclust_corr] # specific to this model
    cols_boot_lasso_32X <- filter.gl(grep("^X32[0-9]{1}_original_firstorder_", colnames(df_dataset), value = TRUE))
    cols_dv <- grep("dv_\\w+_1320", colnames(df_dataset), value = TRUE)
    clinical_vars <- get.clinical_features(colnames(df_dataset), event_col, duration_col)
    covariates_all <- unique(c(clinical_vars, "X1320_original_firstorder_Mean", 
                               cols_dv, cols_rsf_1320, cols_lasso_1320, cols_boot_lasso_32X))
    df_dataset <- df_dataset[, c(event_col, duration_col, covariates_all)]

    # Preprocessing data
    infos <- preprocess_data_cox(df_dataset, covariates_all, event_col, duration_col)
    data_ex <- infos$data[sample(nrow(infos$data), 2000),]
    # After the preprocessing, some variables may be deleted: updating models' covariates
    cols_rsf_1320 <- cols_rsf_1320[cols_rsf_1320 %in% colnames(infos$data)]
    cols_lasso_1320 <- cols_lasso_1320[cols_lasso_1320 %in% colnames(infos$data)]
    cols_boot_lasso_32X <- cols_boot_lasso_32X[cols_boot_lasso_32X %in% colnames(infos$data)]
    cols_dv <- cols_dv[cols_dv %in% colnames(infos$data)]
    clinical_vars <- clinical_vars[clinical_vars %in% colnames(infos$data)]

    # Model rsf 1320 radiomics full covariates
    suffix_model <- "all"
    model_name_rsf <- paste0("1320_radiomics_full_", suffix_model)
    covariates_rsf_1320 <- c(clinical_vars, cols_rsf_1320)
    formula_model_rsf <- get.surv.formula(event_col, covariates_rsf_1320, duration_col = duration_col)
    rsf_1320_params.best <<- read.csv(paste(analyzes_dir, "rsf/", model_name_rsf, "/cv.csv", sep = ""))[1,]
    rsf_1320 <- rfsrc(formula_model_rsf, data = data_ex, ntree = rsf_1320_params.best$ntree, 
                      nodesize = rsf_1320_params.best$nodesize, nsplit = rsf_1320_params.best$nsplit)

    # Model mean cox lasso
    model_name_coxlassomean <- "1320_mean"
    covariates_coxlassomean <- c("X1320_original_firstorder_Mean", clinical_vars)
    formula_model_coxlassomean <- get.surv.formula(event_col, covariates_coxlassomean, duration_col = duration_col)
    coxlassomean <- selection.coxnet(formula_model_coxlassomean, data = data_ex)
    coxlassomean$call$formula <- formula_model_coxlassomean

    # Model mean cox
    # model_name_coxmean <- "1320_mean"
    # covariates_coxmean <- c("X1320_original_firstorder_Mean", clinical_vars)
    # formula_model_coxmean <- get.surv.formula(event_col, covariates_coxmean, duration_col = duration_col)
    # coxmean <- coxph(formula_model_coxmean, data = data_ex, x = TRUE, y = TRUE)
    # # coxmean <- rms::cph(formula_model_coxmean, data = data_ex, x = TRUE, y = TRUE, surv = TRUE)
    # coxmean$call$formula <- formula_model_coxmean

    # Model lasso dosesvolumes
    model_name_lassodv <- "1320_dosesvol_lasso"
    covariates_lassodv <- c(cols_dv, clinical_vars)
    formula_model_lassodv <- get.surv.formula(event_col, covariates_lassodv, duration_col = duration_col)
    lassodv_list.lambda <- read.csv(paste0(analyzes_dir, "coxph_R/", 
                                            model_name_lassodv, "/path_lambda.csv"))[["lambda"]]
    coxlassodv <- selection.coxnet(formula_model_lassodv, data = data_ex, 
                                   list.lambda = lassodv_list.lambda, type.measure = "C")
    
    # Model 1320 lasso features hclust corr
    model_name_lasso_1320 <- "1320_radiomics_full_lasso_features_hclust_corr"
    covariates_lasso_1320 <- c(cols_lasso_1320, clinical_vars)
    formula_model_lasso_1320 <- get.surv.formula(event_col, covariates_lasso_1320, duration_col = duration_col)
    lasso_1320_list.lambda <- read.csv(paste0(analyzes_dir, "coxph_R/", 
                                             model_name_lasso_1320, "/path_lambda.csv"))[["lambda"]]
    coxlasso_1320 <- selection.coxnet(formula_model_lasso_1320, data = data_ex, 
                                     list.lambda = lasso_1320_list.lambda, type.measure = "C")


    # Model 32X firstorder bootstrap lasso all
    model_name_boot_lasso_32X <- "32X_radiomics_firstorder_bootstrap_lasso_all"
    covariates_boot_lasso_32X <- c(cols_boot_lasso_32X, clinical_vars)
    formula_model_boot_lasso_32X <- get.surv.formula(event_col, covariates_boot_lasso_32X, duration_col = duration_col)
    selected_features <- read.csv(paste0(analyzes_dir, "coxph_R/", model_name_boot_lasso_32X, 
                                         "/final_selected_features.csv"))[["selected_features"]]
    bootstrap_selected_features <- read.csv(paste0(analyzes_dir, "coxph_R/", model_name_boot_lasso_32X, 
                                                   "/bootstrap_selected_features.csv"), header = T)
    coxbootlasso_32X <- bootstrap.coxnet(data = data_ex, formula = formula_model_boot_lasso_32X, pred.times,
                                         best.lambda.method = "lambda.1se", selected_features = selected_features, 
                                         bootstrap_selected_features = bootstrap_selected_features)

    # Self-made pec estim
    formula_ipcw = get.ipcw.surv.formula(event_col, colnames(infos$data), duration_col = duration_col)
    pred.times <- seq(1, 60, 1)
    pec_M = floor(0.7 * nrow(infos$data))
    pec_B = B
    compared_models <- list("Cox mean dose lasso" = coxlassomean,
                            #"Cox Lasso doses-volumes" = coxlassodv,
                            "Cox Lasso screened whole heart dosiomics" = coxlasso_1320,
                            "Cox Bootstrap Lasso heart's subparts first-order dosiomics" = coxbootlasso_32X
                            #"RSF whole-heart dosiomics" = rsf_1320
                            )
    pretty_model_names <- rlang::duplicate(names(compared_models))
    # names(compared_models) <- c(model_name_coxlassomean, model_name_lassodv, model_name_lasso_1320, 
    #                             model_name_boot_lasso_32X, model_name_rsf)
    names(compared_models) <- c(model_name_coxlassomean, model_name_lasso_1320, model_name_boot_lasso_32X)
    # We eval the calls' arguments because the corresponding variables won't be available in the functions
    compared_models <- lapply(compared_models, function(model) {
            model$call$formula <- eval(model$call$formula)
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
    boot.parallel <- `if`(Sys.getenv("SLURM_NTASKS") == "", "foreach", "rslurm")
    boot_error <- bootstrap.estim.error(infos$data, formula_ipcw, compared_models, analyzes_dir, 
                                        B = B, times = pred.times, 
                                        logfile = NULL, boot.parallel = boot.parallel)
    boot_error 
    # PEC
    # formula_ipcw = get.ipcw.surv.formula(event_col, colnames(infos$data), duration_col = duration_col)
    # pred.times <- seq(1, 60, 1)
    # pec_M = floor(0.7 * nrow(infos$data))
    # pec_B = B
    # cl <- parallel::makeCluster(nworkers)
    # doParallel::registerDoParallel(cl)
    # print(paste("PEC, B =", pec_B))
    # compared_models <- list("Cox mean dose" = coxmean, 
    #                    "Cox Lasso doses-volumes" = coxlassodv, 
    #                    "Cox Lasso heart's subparts screened dosiomics" = coxlasso_32X, 
    #                    "RSF whole-heart screened first-order dosiomics" = rsf.best
    #                    )
    # fitpec <- pec(compared_models, 
    #               data = infos$data, formula = formula_ipcw,
    #               times = pred.times, start = pred.times[1], 
    #               exact = F, splitMethod = "BootCv", reference = F, 
    #               B = pec_B, M = pec_M, keep.index = T, keep.matrix = T)
    # print("End pec")
    # saveRDS(fitpec, file = paste0(analyzes_dir, "pec_results/fit_pec_", pec_B, ".rds"))
    # png(paste0(analyzes_dir, "pec_plots/bootcv_", pec_B, ".png"), width = 800, height = 600)
    # plot(fitpec, what = "BootCvErr", xlim = c(0, 60),
    #      axis1.at = seq(0, 60, 5), axis1.label = seq(0, 60, 5))
    # dev.off()
    # parallel::stopCluster(cl)
    # print(paste("Cindex, B =", pec_B))
    # print(class(infos$data))
    # print(dim(infos$data))
    # fitcindex <- pec::cindex(compared_models, 
    #                          data = infos$data, formula = formula_ipcw,
    #                          times = pred.times, start = pred.times[1], 
    #                          exact = F, splitMethod = "bootcv", reference = F, 
    #                          B = pec_B, M = pec_M, keep.index = T, keep.matrix = T)
    # print("End cindex")
    # saveRDS(fitcindex, file = paste0(analyzes_dir, "pec_results/fit_cindex_", pec_B, ".rds"))
    # png(paste0(analyzes_dir, "pec_plots/cindex_", pec_B, ".png"), width = 800, height = 600)
    # plot(fitcindex, what = "BootCvErr", xlim = c(0, 60),
    #      axis1.at = seq(0, 60, 5), axis1.label = seq(0, 60, 5))
    # dev.off()
}

args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    analyzes_dir <- args[1]
    event_col <- args[2]
    duration_col <- `if`(length(args) == 3, args[3], "survival_time_years")
    file_dataset = paste0(analyzes_dir, "datasets/dataset.csv.gz")
    log_threshold(INFO)
    pec_estimation(file_dataset, event_col, analyzes_dir, duration_col)
} else {
    print("No arguments provided. Skipping.")
}

