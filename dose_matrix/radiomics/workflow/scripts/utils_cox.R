
suppressPackageStartupMessages({
library("caret", quietly = T)
library("boot", quietly = T)
library("survival", quietly = T)
library("glmnet", quietly = T)
library("pec", quietly = T)
library("Hmisc", quietly = T)
library("logger", quietly = T)
library("parallel", quietly = T)
library("doParallel", quietly = T)
library("ggplot2", quietly = T)
library("reshape2", quietly = T)
library("randomForestSRC", quietly = T)
library("rslurm", quietly = T)
})
# Not available in conda
# if(!require("bolasso")) {
#     install.packages("bolasso", repos = "https://cran.irsn.fr/")
#}

source("workflow/scripts/utils_radiopreditool.R")

# Select the best lambda after CV coxnet for a glmnet::cv.glmnet object
loglik_ratio_best_lambda <- function(cox_object, cv.params) {
    lambda.ref <- cox_object$lambda.min
    deviance.params <- data.frame(penalty = cox_object$lambda, deviance = deviance(cox_object$glmnet.fit))
    cv.params.merge <- merge(cv.params, deviance.params, by = "penalty")
    rownames(cv.params.merge) <- cv.params.merge$penalty
    deviance.ref <- cv.params.merge[as.character(lambda.ref), "deviance"]
    nonzero.ref <- cv.params.merge[as.character(lambda.ref), "non_zero_coefs"]
    cv.params.unique <- cv.params.merge[order(-cv.params.merge$non_zero_coefs, cv.params.merge$penalty), ]
    cv.params.unique <- cv.params.unique[!duplicated(cv.params.unique$non_zero_coefs), ]
    cv.params.unique <- cv.params.unique[(cv.params.unique$penalty > lambda.ref) &
                                         (cv.params.unique$non_zero_coefs < nonzero.ref), ]
    logger::log_info("Best lambda selection")
    logger::log_info(paste("Lambda ref:", lambda.ref, nonzero.ref))
    logger::log_info(paste("Nzeros of lambda ref:", nonzero.ref, " - Deviance ref:", deviance.ref))
    # write.csv(cv.params, file = "test_cv_params.csv")
    # write.csv(cv.params.unique, file = "test_cv_params_unique.csv")
    lambda.new <- lambda.ref
    if (nrow(cv.params.unique) < 1) return (lambda.ref)
    for (i in 1:nrow(cv.params.unique)) {
        deviance.new <- cv.params.unique[i, "deviance"]
        nonzero.new <- cv.params.unique[i, "non_zero_coefs"]
        loglik.ratio <- deviance.new - deviance.ref
        degrees <- nonzero.ref - nonzero.new
        pvalue <- 1 - pchisq(loglik.ratio, degrees)
        logger::log_info(paste("- compared to", cv.params.unique[i, "penalty"], nonzero.new))
        if (is.na(pvalue) | is.null(pvalue)) {
            log_warn(paste("- lambda ref, nzeros:", lambda.ref, nonzero.ref))
            log_warn(paste("- lambda new to test, nzeros:", cv.params.unique[i, 'penalty'], nonzero.new))
            log_warn(paste("- degrees:", degrees))
            log_warn(paste("- deviance ref:", deviance.ref))
            log_warn(paste("- deviance new:", deviance.new))
            log_warn(paste("- loglik:", loglik.ratio))
            log_warn(paste("- pvalue:", pvalue))
        }
        if (pvalue < 0.01) break
        # deviance.ref <- cv.params.unique[i, "deviance"]
        # nonzero.ref <- cv.params.unique[i, "non_zero_coefs"]
        lambda.new <- cv.params.unique[i, "penalty"]
    }
    lambda.new
}

# Get the best lambda of a cv.glmnet object according to a specified method (glmnet, home-made..)
get.best.lambda <- function(object, method, cv.params) {
    if (is(object, "cv.glmnet")) {
        if (method == "lambda.1se") return(object[["lambda.1se"]])
        if (method == "lambda.min") return(object[["lambda.min"]])
        if (method == "loglik_ratio") return(loglik_ratio_best_lambda(object, cv.params))
    }
    return(NULL)
}

# Simple data preprocessing for cox models
preprocess_data_cox <- function(df_dataset, covariates, event_col, duration_col) {
    stopifnot({
        !(event_col %in% covariates)
        !(duration_col %in% covariates)
    })
    ## Preprocessing sets
    filtered_covariates <- preliminary_filter(df_dataset, covariates, event_col)
    df_model <- df_dataset[,c(event_col, duration_col, filtered_covariates)]
    df_model <- na.omit(df_model)
    # Z normalisation
    df_model <- normalize_data(df_model, filtered_covariates, event_col, duration_col)$train
    formula_model <- get.surv.formula(event_col, filtered_covariates, duration_col = duration_col)
    return (list("data" = df_model, "covariates" = filtered_covariates, "formula_model" = formula_model)) 
}

# Z-normalization of data
normalize_data <- function(df_train, covariates, event_col, duration_col, df_test = NULL) {
    continuous_vars <- covariates
    discrete_vars <- NULL
    regex_non_continuous <- paste0("^((Sexe)|(iccc)|(has_radiomics)|(categ_age_at_diagnosis)|", 
                                   "(chimiotherapie)|(ALKYL)|(ANTHRA)|(radiotherapie_1K))")
    idx_non_continuous <- grep(regex_non_continuous, covariates)
    if (length(idx_non_continuous) > 0) {
        continuous_vars <- covariates[-idx_non_continuous]
        discrete_vars <- covariates[idx_non_continuous]
    }
    means_train <- as.numeric(lapply(df_train[continuous_vars], mean))
    stds_train <- as.numeric(lapply(df_train[continuous_vars], sd))
    df_train[, continuous_vars] <- scale(df_train[continuous_vars], center = means_train, scale = stds_train)
    if (!is.null(discrete_vars)) {
        means_unique_train <- as.numeric(lapply(df_train[discrete_vars], function(x) { mean(unique(x)) } ))
        scales_unique_train <- as.numeric(lapply(df_train[discrete_vars], function(x) { max(x) - min(x) } ))
        df_train[, discrete_vars] <- scale(df_train[,discrete_vars], 
                                           center = means_unique_train, scale = scales_unique_train)
    }
    if (!is.null(df_test)) {
        df_test[, continuous_vars] <- scale(df_test[continuous_vars], center = means_train, scale = stds_train)
        if (!is.null(discrete_vars))
            df_test[, discrete_vars] <- scale(df_test[,discrete_vars], 
                                              center = means_unique_train, scale = scales_unique_train)
    }
    list("train" = df_train, "test" = df_test) 
}

# From dataframe to glmnet objects
coxlasso_data <- function(df, covariates, event_col, duration_col) {
    X <- as.matrix(df[covariates])
    surv_y <- survival::Surv(df[[duration_col]], df[[event_col]])
    list("X" = X, "surv_y" = surv_y)
}

# S3 method for selection.coxnet
predictSurvProb.selection.coxnet <- function(object, newdata, times, ...) {
    pec::predictSurvProb(object$coxph.fit, newdata = newdata, times = times, ...)
}

# S3 method for bootstrap.coxnet
predictSurvProb.bootstrap.coxnet <- function(object, newdata, times, ...) {
    pec::predictSurvProb(object$coxph.fit, newdata = newdata, times = times, ...)
}

# Select the best lambda from cv.glmnet / glmnet object
# To be implemented : method is a function
get.coefs.cox <- function(object, lambda) {
    # It means it's a loaded object in my case : get the last lambda
    if (is(object, "glmnet")) return(glmnet::coef.glmnet(object, s = tail(object$lambda, 1))[,1])
    # Best lambda method allowed by glmnet
    if (is(object, "cv.glmnet")) return(glmnet::coef.glmnet(object, s = lambda)[,1])
}

# Lasso selection + coxph model
selection.coxnet <- function(formula, data, alpha = 1, nfolds = 5, list.lambda = NULL,
                             best.lambda.method = "lambda.1se", cv.parallel = T, 
                             type.measure = "C", logfile = NULL, run_type = "error_estimation") {
    stopifnot(run_type %in% c("selection", "error_estimation"))
    if (!is.null(logfile)) logger::log_appender(logger::appender_file(logfile, append = T))
    else logger::log_appender(logger::appender_stdout)
    covariates <- all.vars(formula[[3]])
    duration_col <- all.vars(formula[[2]])[1]
    event_col <- all.vars(formula[[2]])[2]
    lasso_data <- coxlasso_data(data, covariates, event_col, duration_col)
    coxnet_X <- lasso_data$X 
    coxnet_surv_y <- lasso_data$surv_y
    # Lasso selection
    # If we have to estimate the lambda parameter
    cv.params <- NULL
    if (is.null(list.lambda)) {
        if (cv.parallel) { 
            logger::log_info(paste("CV lasso: creating a cluster with", nfolds, "workers"))
            cl <- parallel::makeCluster(nfolds)
            doParallel::registerDoParallel(cl)
            logger::log_info(paste(colnames(showConnections()), collapse = " "))
            logger::log_info(paste(apply(showConnections(), 1, 
                                   function(row) { paste(paste(as.character(row), collapse = " ")) }), collapse = "\n"))
        }
        coxnet_model <- glmnet::cv.glmnet(coxnet_X, coxnet_surv_y, family = "cox", alpha = alpha,  
                                          nfolds = nfolds, parallel = cv.parallel, type.measure = "C")
        if (cv.parallel) parallel::stopCluster(cl)
        cv.params <- data.frame(non_zero_coefs = as.numeric(coxnet_model$nzero), 
                                penalty = coxnet_model$lambda, 
                                mean_score = coxnet_model$cvm, 
                                std_score = as.numeric(coxnet_model$cvsd))
        cv.params <- cv.params[order(cv.params$mean_score, decreasing = T), ]
    } 
    # We know the best lamba (and the regularization path)
    else {
      coxnet_model <- glmnet::glmnet(coxnet_X, coxnet_surv_y, family = "cox", alpha = alpha, lambda = list.lambda,  
                                     nfolds = nfolds, parallel = cv.parallel, type.measure = "C")
    }
    best.lambda <- get.best.lambda(coxnet_model, best.lambda.method, cv.params)
    best.coefs.cox <- get.coefs.cox(coxnet_model, best.lambda)
    nonnull.covariates <- names(best.coefs.cox[abs(best.coefs.cox) > 0])
    print(paste("Number of covariates", length(covariates)))
    print(paste("Number of non null covariates", length(nonnull.covariates)))
    print(paste("Best lambda:", best.lambda))
    print(paste("Lambda.1se coxnet:", coxnet_model$lambda.1se))
    print(paste("Lambda.min coxnet:", coxnet_model$lambda.min))
    # Coxph on selected features
    formula_nonnull <- get.surv.formula(event_col, nonnull.covariates, duration_col = duration_col)
    if (run_type == "error_estimation") 
        coxmodel <- survival::coxph(formula_nonnull, data = data, x = T, y = T, control = coxph.control(iter.max = 1000))
    else
        coxmodel <- NULL
    out <- list("call" = match.call(), "selected_features" = nonnull.covariates, "coxnet.fit" = coxnet_model, 
                "coxnet.cv.params" = cv.params, "best.lambda.method" = best.lambda.method, "best.lambda" = best.lambda, 
                "coxph.fit" = coxmodel, "coxph.formula" = formula_nonnull, "surv_y" = coxnet_surv_y)
    class(out) <- "selection.coxnet"
    out
}

# Returns selected variables names by a bootstrap Coxnet based on a threshold
select.bolasso.features <- function(bootstrap_selected_features, threshold = 1) {
    freq_features <- colSums(bootstrap_selected_features) / nrow(bootstrap_selected_features)
    colnames(bootstrap_selected_features)[freq_features >= threshold]
}

# Task of Coxnet selection for one bootstrap sample
sample.selection.coxnet <- function(data, indices, lasso_data_full, formula, alpha, 
                                    best.lambda.method, nfolds, type.measure, pred.times, run_type) {
    stopifnot(run_type %in% c("selection", "error_estimation"))
    bstrap_data <- data[indices, ]
    select_coxmodel <- selection.coxnet(formula, bstrap_data, best.lambda.method = best.lambda.method,
                                        nfolds = nfolds, cv.parallel = F, run_type = run_type)
    covariates <- all.vars(formula[[3]])
    ind_selected_features <- rep(0, length(covariates))
    ind_selected_features[covariates %in% select_coxmodel$selected_features] <- 1
    if (run_type == "error_estimation") {
      idx_surv <- length(pred.times)
      cox.survprob.boot <- predictSurvProb.selection.coxnet(select_coxmodel, times = pred.times, 
                                                            newdata = data.table::as.data.table(bstrap_data))
      cox.survprob.all <- predictSurvProb.selection.coxnet(select_coxmodel,times = pred.times, 
                                                           newdata = data.table::as.data.table(data))
      cindex.boot <- Hmisc::rcorr.cens(cox.survprob.boot[,idx_surv], S = select_coxmodel$surv_y)[["C Index"]]
      cindex.orig <- Hmisc::rcorr.cens(cox.survprob.all[,idx_surv], S = lasso_data_full$surv_y)[["C Index"]]
      c(cindex.boot, cindex.orig, ind_selected_features)
    } else if(run_type == "selection") {
      c(NA, NA, ind_selected_features)
    }
}

# Task for a slurm job in bootstrap lasso: several sample.selection.coxnet calls
slurm_job_boot_coxnet <- function(nb_coxnet, data, lasso_data_full, formula, alpha, 
                                  best.lambda.method, nfolds, type.measure, pred.times, run_type) {
  resBoot <- foreach(i = 1:nb_coxnet, .combine = 'rbind') %do% {
    idx_boot <- sample(nrow(data), replace = T)
    sample.selection.coxnet(data, idx_boot, lasso_data_full = lasso_data_full, formula = formula, alpha = alpha, 
                            best.lambda.method = best.lambda.method, nfolds = nfolds, 
                            type.measure = type.measure, pred.times = pred.times, run_type = run_type)
  }
  return(resBoot)
}

# Bootstrap error selection/estimation for the coxnet model
bootstrap.coxnet <- function(data, formula, pred.times, B = 100, alpha = 1, run_type = "selection",
                             best.lambda.method = "lambda.1se", nfolds = 5, boot.parallel = "foreach", 
                             boot.ncpus = get.nworkers(), coxnet_per_job = 5, 
                             type.measure = "C", bolasso.threshold = 0.9, selected_features = NULL,
                             bootstrap_selected_features = NULL, logfile = NULL) {
    stopifnot(boot.parallel %in% c("boot.multicore", "foreach", "rslurm"))
    stopifnot(run_type %in% c("selection", "error_estimation"))
    if (!is.null(logfile)) log_appender(appender_file(logfile, append = T))
    else log_appender(appender_stdout)
    covariates <- all.vars(formula[[3]])
    duration_col <- all.vars(formula[[2]])[1]
    event_col <- all.vars(formula[[2]])[2]
    lasso_data_full <- coxlasso_data(data, covariates, event_col, duration_col)
    log_info(paste("Bootstrap lasso, best lambda method:", best.lambda.method))
    log_info(paste("Bootstrap Lasso threshold:", bolasso.threshold))
    log_info(paste("Bootstrap lasso, run type:", run_type))
    if (is.null(selected_features)) {
        log_info(paste("Bootstrap lasso: parallel method is", boot.parallel))
        functions_to_export <- c("slurm_job_boot_coxnet", "sample.selection.coxnet", "selection.coxnet", "loglik_ratio_best_lambda",
                                 "predictSurvProb.selection.coxnet", "coxlasso_data", "preliminary_filter", "filter_dummies_iccc",
                                 "get.coefs.cox", "get.surv.formula", "get.best.lambda")
        # Launch bootstrap
        if (boot.parallel == "boot.multicore") {
          resBoot <- boot(data, sample.selection.coxnet, R = B, parallel = "multicore", ncpus = boot.ncpus, 
                          lasso_data_full = lasso_data_full, formula = formula, alpha = alpha, 
                          nfolds = nfolds, best.lambda.method = best.lambda.method, 
                          type.measure = type.measure, pred.times = pred.times)$t
        } else if (boot.parallel == "foreach") { 
          log_info(paste("Bootstrap lasso: creating a cluster with", boot.ncpus, "workers"))
          cl <- parallel::makeCluster(boot.ncpus)
          doParallel::registerDoParallel(cl)
          log_info(paste(colnames(showConnections()), collapse = " "))
          log_info(paste(apply(showConnections(), 1, 
                               function(row) { paste(paste(as.character(row), collapse = " ")) }), collapse = "\n"))
          # survival::Surv doesn't work in foreach.. must export survival
          resBoot <- foreach(i = 1:B, .combine = 'rbind', .export = functions_to_export,
                             .packages = c("survival")) %dopar% {
            idx_boot <- sample(nrow(data), replace = T)
            sample.selection.coxnet(data, idx_boot, lasso_data_full, formula, alpha, 
                                    best.lambda.method, nfolds, type.measure, pred.times)
          }
          if (nrow(resBoot[,-c(1,2)]) != B) stop(paste("Bootstrap matrix nrow:", nrow(resBoot[,-c(1,2)]), "- B:", B))
          parallel::stopCluster(cl)
        } else if (boot.parallel == "rslurm") {
          nb_coxnet_jobs <- rep(coxnet_per_job, B %/% coxnet_per_job)
          if (B %% coxnet_per_job > 0) nb_coxnet_jobs <- c(nb_coxnet_jobs, B %% coxnet_per_job)
          log_info(paste("Number of slurm jobs:", length(nb_coxnet_jobs)))
          log_info(do.call(paste, as.list(nb_coxnet_jobs)))
          sopt <- list(time = "03:30:00", "ntasks" = 1, "cpus-per-task" = 1, 
                       partition = "cpu_med", mem = "20G")
          job_uuid <- stringr::str_split(uuid::UUIDgenerate(), "-")[[1]][1]
          sjob <- slurm_apply(function(nb_coxnet) slurm_job_boot_coxnet(nb_coxnet, data, 
                              lasso_data_full = lasso_data_full, formula = formula, alpha = alpha, 
                              best.lambda.method = best.lambda.method, nfolds = nfolds, 
                              type.measure = type.measure, pred.times = pred.times, run_type = run_type),
                              data.frame(nb_coxnet = nb_coxnet_jobs), 
                              nodes = length(nb_coxnet_jobs), cpus_per_node = 1, processes_per_node = 1, 
                              global_objects = functions_to_export, 
                              jobname = paste0("bootstrap_lasso_", B, "_", job_uuid),
                              slurm_options = sopt)
          log_info("Jobs are submitted")
          listResJobs <- get_slurm_out(sjob, outtype = "raw", wait = T)
          resBoot <- do.call("rbind", listResJobs)
          if (nrow(resBoot[,-c(1,2)]) != B) stop(paste("Bootstrap matrix nrow:", nrow(resBoot[,-c(1,2)]), "- B:", B))
          cleanup_files(sjob, wait = T)
          log_info("End of all submitted jobs")
        }
        if (run_type == "error_estimation") {
          # Computation of the adjusted C-index over the whole dataset
          select_coxmodel_app <- selection.coxnet(formula, data, alpha = alpha, nfolds = nfolds, 
                                                  best.lambda.method = best.lambda.method, cv.parallel = F)
          idx_surv <- length(pred.times)
          cox.survprob.all <- predictSurvProb(select_coxmodel_app, newdata = data.table::as.data.table(data), 
                                              times = pred.times)
          cindex.app <- rcorr.cens(cox.survprob.all[,idx_surv], S = lasso_data_full$surv_y)[["C Index"]]
          optimism = mean(resBoot[,1] - resBoot[,2])
          cindex.adjust <- cindex.app - optimism
        }
        # Select the features based on the bootstrap
        bootstrap_selected_features <- resBoot[,-c(1,2)]
        colnames(bootstrap_selected_features) <- covariates
        selected_features <- select.bolasso.features(bootstrap_selected_features, bolasso.threshold)
    }
    # We only take the selected features that are in the formula
    else {
      selected_features <- selected_features[selected_features %in% covariates]
    }
    log_info(paste("Selected features:", do.call(paste, as.list(selected_features))))
    formula_selected <- get.surv.formula(event_col, selected_features, duration_col = duration_col)
    coxmodel <- coxph(formula_selected, data = data, x = T, y = T, control = coxph.control(iter.max = 1000))
    # bootstrap.coxnet model
    if (is.null(selected_features)) {
        out <- list("cindex_adjusted" = cindex.adjust, "optimism" = optimism, "cindex_app" = cindex.app,  
                    "cindex_bootstrap" = resBoot[,1], "cindex_orig" = resBoot[,2],
                    "bootstrap_selected_features" = bootstrap_selected_features, 
                    "selected_features" = selected_features, 
                    "coxph.fit" = coxmodel, "call" = match.call())
    } else {
        out <- list("coxph.fit" = coxmodel, "selected_features" = selected_features, 
                    "bootstrap_selected_features" = bootstrap_selected_features, "call" = match.call())
    }
    class(out) <- "bootstrap.coxnet"
    out
}

# Train a Cox model with a specified labeled dataset
model_cox.id <- function(id_set, covariates, event_col, duration_col, 
                         analyzes_dir, model_name, coxlasso_logfile, screening_method = "all", n_boot = 200,
                         load_results = F, save_results = T, do_plot = F, save_rds = F, penalty = "lasso") {
    df_trainset <- read.csv(paste0(analyzes_dir, "datasets/trainset_", id_set, ".csv.gz"), header = T)
    df_testset <- read.csv(paste0(analyzes_dir, "datasets/testset_", id_set, ".csv.gz"), header = T)
    if (!is.null(coxlasso_logfile)) log_appender(appender_file(coxlasso_logfile, append = T))
    else log_appender(appender_stdout)
    print("Accessible variables")
    print(ls())
    print("Now i'm going to print n_boot and penalty")
    print(paste("penalty:", as.character(penalty)))
    print(paste("nboot:", as.character(n_boot)))
    print("n_boot is printed")
    model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, 
              model_name, coxlasso_logfile, screening_method = screening_method, penalty = penalty, n_boot = n_boot,
              do_plot = do_plot, load_results = load_results, save_results = save_results, save_rds = save_rds,
              run_multiple = T, level = INFO, id_set = id_set)
}

# Train a Cox model
model_cox <- function(df_trainset, df_testset, covariates, event_col, duration_col, 
                      analyzes_dir, model_name, coxlasso_logfile,
                      screening_method = "all", penalty = "lasso", cv_nfolds = 5, n_boot = 200,
                      do_plot = T, load_results = F, save_results = T, save_rds = T, run_multiple = F,
                      level = INFO, id_set = "") {
    stopifnot(screening_method %in% c("all", "features_hclust_corr"))
    stopifnot(penalty %in% c("none", "lasso", "bootstrap_lasso"))
    log_threshold(level)
    if (!is.null(coxlasso_logfile)) log_appender(appender_file(coxlasso_logfile, append = T))
    else log_appender(appender_stdout)
    save_results_dir <- paste0(analyzes_dir, "coxph_R/", model_name, "/")
    dir.create(save_results_dir, showWarnings = F)
    if (id_set != "") save_results_dir <- paste0(save_results_dir, id_set, "/")
    dir.create(save_results_dir, showWarnings = F)
    ## Preprocessing sets
    filtered_covariates <- preliminary_filter(df_trainset, covariates, event_col, screening_method, id_set, analyzes_dir)
    df_model_train <- df_trainset[,c(event_col, duration_col, filtered_covariates)]
    df_model_test <- df_testset[,c(event_col, duration_col, filtered_covariates)]
    df_model_train <- na.omit(df_model_train)
    df_model_test <- na.omit(df_model_test)
    # Z normalisation
    norm_data <- normalize_data(df_model_train, filtered_covariates, event_col, duration_col, df_test = df_model_test)
    df_model_train <- norm_data$train; df_model_test <- norm_data$test
    formula_model <- get.surv.formula(event_col, filtered_covariates, duration_col = duration_col)
    log_info(paste0("(", id_set, ") ", "Model name: ", model_name))
    log_info(paste0("(", id_set, ") ", "Covariates (", length(filtered_covariates),"):"))
    log_info(paste0("(", id_set, ") ", "Screening method: ", screening_method))
    if (!run_multiple) {
        log_info(paste0(filtered_covariates, collapse = ", "))
        log_info("NAs are omitted")
        log_info(paste("Trained:", nrow(df_model_train), "samples"))
        log_info(paste("Testset: ", nrow(df_model_test), " samples"))
        log_info(paste("Number of bootstrap samples (for bootstrap model only):", n_boot))
    }
    final.time <- floor(min(max(df_model_train[[duration_col]]), max(df_model_test[[duration_col]]), 60))
    pred.times <- seq(1, final.time, by = 1)
    idx_surv <- length(pred.times)
    ## Model and predictions
    coxlasso_data_train <- coxlasso_data(df_model_train, filtered_covariates, event_col, duration_col)
    surv_y_train <- coxlasso_data_train$surv_y
    coxlasso_data_test <- coxlasso_data(df_model_test, filtered_covariates, event_col, duration_col)
    surv_y_test <- coxlasso_data_test$surv_y   
    print(paste("n_boot:", n_boot))
    if (penalty == "none") {
        coxmodel <- coxph(formula_model, data = df_model_train, x = T, y = T, 
                          control = coxph.control(iter.max = 1000))
    } else if (penalty == "lasso") {
        best.lambda.method <- "lambda.1se"
        if (load_results) {
            best.lambda <- read.csv(paste0(save_results_dir, "best_params.csv"))[1, "penalty"]
            list.lambda <- read.csv(paste0(save_results_dir, "path_lambda.csv"))[["lambda"]]
            coxmodel <- selection.coxnet(formula_model, df_model_train, alpha = 1, 
                                         list.lambda = list.lambda, type.measure = "C", logfile = coxlasso_logfile)
        } else {
            coxmodel <- selection.coxnet(formula_model, df_model_train, 
                                         alpha = 1, nfolds = cv_nfolds, best.lambda.method = best.lambda.method,
                                         cv.parallel = F, type.measure = "C", logfile = coxlasso_logfile)
            cv.params <- coxmodel$coxnet.cv.params
            best.lambda <- coxmodel$best.lambda
            if (save_results) {
                mat.coefs <- t(as.matrix(coef(cox_object$glmnet.fit)))
                rownames(mat.coefs) <- cox_object$glmnet.fit$lambda
                write.csv(mat.coefs, file = paste0(save_results_dir, "mat_coefs.csv"), row.names = T)
                write.csv(cv.params, file = paste0(save_results_dir, "cv.csv"), row.names = F)
                write.csv(data.frame(penalty = best.lambda, l1_ratio = 1.0), row.names = F, 
                          file = paste0(save_results_dir, "best_params.csv"))
                write.csv(data.frame(lambda = coxmodel$coxnet.fit$lambda[coxmodel$coxnet.fit$lambda >= best.lambda]), 
                          row.names = F, file = paste0(save_results_dir, "path_lambda.csv"))
            }
        }
        if (!run_multiple) log_info(paste("Best lambda:", best.lambda))
    } else if (penalty == "bootstrap_lasso") {
        best.lambda.method <- "loglik_ratio"
        if (load_results) {
            selected_features <- read.csv(paste0(save_results_dir, "final_selected_features.csv"))[["selected_features"]]
            bootstrap_selected_features <- read.csv(paste0(save_results_dir, "bootstrap_selected_features.csv"), 
                                                    header = T)
            coxmodel <- bootstrap.coxnet(df_model_train, formula_model, pred.times, B = n_boot, 
                                         best.lambda.method = best.lambda.method, selected_features = selected_features, 
                                         bootstrap_selected_features = bootstrap_selected_features, 
                                         logfile = coxlasso_logfile)
        } else {
            boot.parallel <- `if`(Sys.getenv("SLURM_NTASKS") == "", "foreach", "rslurm")
            bolasso.threshold <- 0.8
            coxmodel <- bootstrap.coxnet(df_model_train, formula_model, pred.times, B = n_boot,
                                         boot.parallel = boot.parallel, bolasso.threshold = bolasso.threshold,
                                         best.lambda.method = best.lambda.method, logfile = coxlasso_logfile)
            if (save_results) {
                write.csv(coxmodel$bootstrap_selected_features, row.names = F, 
                          file = paste0(save_results_dir, "bootstrap_selected_features.csv"))
                write.csv(data.frame(selected_features = coxmodel$selected_features), row.names = F,
                          file = paste0(save_results_dir, "final_selected_features.csv"))
            }
        }
    }

    coxlasso.survprob.train <- predictSurvProb(coxmodel, newdata = data.table::as.data.table(df_model_train), 
                                               times = pred.times)
    coxlasso.survprob.test <- predictSurvProb(coxmodel, newdata = data.table::as.data.table(df_model_test), 
                                              times = pred.times)
    formula_ipcw <- get.ipcw.surv.formula(event_col, filtered_covariates)
    # C-index ipcw (censored free, marginal = KM)
    coxlasso.cindex.ipcw.train <- pec::cindex(list("Best coxlasso" = as.matrix(coxlasso.survprob.train[,idx_surv])), 
                                              formula = formula_ipcw, 
                                              data = df_model_train, cens.model = "cox")$AppCindex[["Best coxlasso"]]
    coxlasso.cindex.ipcw.test <- pec::cindex(list("Best coxlasso" = as.matrix(coxlasso.survprob.test[,idx_surv])), 
                                             formula = formula_ipcw, 
                                             data = df_model_test, cens.model = "cox")$AppCindex[["Best coxlasso"]]
    # Harrell's C-index
    coxlasso.cindex.harrell.train <- rcorr.cens(coxlasso.survprob.train[,idx_surv], S = surv_y_train)[["C Index"]]
    coxlasso.cindex.harrell.test <- rcorr.cens(coxlasso.survprob.test[,idx_surv], S = surv_y_test)[["C Index"]]
    # IBS
    coxlasso.perror.train <- pec(object= list("train" = coxlasso.survprob.train), 
                                 formula = formula_ipcw, data = df_model_train, 
                                 cens.model = "cox",
                                 times = pred.times, start = pred.times[1], 
                                 exact = F, reference = F)
    coxlasso.perror.test <- pec(object= list("test" = coxlasso.survprob.test), 
                                formula = formula_ipcw, data = df_model_test, 
                                cens.model = "cox", 
                                times = pred.times, start = pred.times[1], 
                                exact = F, reference = F)
    coxlasso.bs.final.train <- tail(coxlasso.perror.train$AppErr$train, 1)
    coxlasso.bs.final.test <- tail(coxlasso.perror.test$AppErr$test, 1)
    coxlasso.ibs.train <- crps(coxlasso.perror.train)[1]
    coxlasso.ibs.test <- crps(coxlasso.perror.test)[1]
    if (!run_multiple) {
        log_info(paste(as.character(formula_ipcw)[c(2,1,3)], collapse = " "))
        log_info(paste0("Harrell's C-index on trainset: ", coxlasso.cindex.harrell.train))
        log_info(paste0("Harrell's C-index on testset: ", coxlasso.cindex.harrell.test))
        log_info(paste0("IPCW C-index on trainset: ", coxlasso.cindex.ipcw.train))
        log_info(paste0("IPCW C-index on testset: ", coxlasso.cindex.ipcw.test))
        log_info(paste0("BS at 60 on trainset: ", coxlasso.bs.final.train))
        log_info(paste0("BS at 60 on testset: ", coxlasso.bs.final.test))
        log_info(paste0("IBS on trainset: ", coxlasso.ibs.train))
        log_info(paste0("IBS on testset: ", coxlasso.ibs.test))
    }
    results_train <- c(coxlasso.cindex.harrell.train, coxlasso.cindex.ipcw.train, 
                       coxlasso.bs.final.train, coxlasso.ibs.train)
    results_test <- c(coxlasso.cindex.harrell.test, coxlasso.cindex.ipcw.test,
                      coxlasso.bs.final.test, coxlasso.ibs.test)
    log_info(paste0("(", id_set, ") ", "Train: ", results_train[1], " & ", results_train[2], 
                    " & ", results_train[3], " & ", results_train[4]))
    log_info(paste0("(", id_set, ") ", "Test: ", results_test[1], " & ", results_test[2], 
                    " & ", results_test[3], " & ", results_test[4]))
    df_results <- data.frame(Train = results_train, Test = results_test)
    rownames(df_results) <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
    if (save_results) {
        # Save coefs from final Cox PH fit
        if (is(coxmodel, "selection.coxnet") | is(coxmodel, "bootstrap.coxnet")) coxph_fit <- coxmodel$coxph.fit
        else coxph_fit <- coxmodel
        best_coefs <- coef(coxph_fit)
        df_coefs = data.frame(labels = pretty.labels(names(best_coefs)), raw_labels = names(best_coefs),
                              coefs = best_coefs)
        df_coefs <- df_coefs[order(-abs(df_coefs$coefs)), ]
        write.csv(df_coefs, file = paste0(save_results_dir, "best_coefs.csv"), row.names = F)
        write.csv(df_results, file = paste0(save_results_dir, "metrics.csv"), row.names = T)
        if (save_rds) saveRDS(coxmodel, file = paste0(save_results_dir, "model.rds"))
    }
    if (do_plot) {
        plot_cox_coefs(save_results_dir)
        if (penalty == "lasso") plot_cox_lambda_path(save_results_dir)
        if (penalty == "bootstrap_lasso") plot_bootstrap_cox(save_results_dir, n_boot)
    }
    log_threshold(INFO)
    results_test
}

# Run multiple scores estimation for a Cox model with presaved train / test sets in parallel
parallel_multiple_scores_cox <- function(nb_estim, covariates, event_col, duration_col, analyzes_dir, model_name, 
                                         logfile, penalty = "lasso", parallel.method = "mclapply", 
                                         screening_method = "all", n_boot = 200) {
  stopifnot(parallel.method %in% c("mclapply", "rslurm"))
  stopifnot(penalty %in% c("none", "lasso", "bootstrap_lasso"))
  if (!is.null(logfile)) log_appender(appender_file(logfile, append = T))
  else log_appender(appender_stdout)
  index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
  # ggsave() doesn't work with mclapply... what a language
  if (parallel.method == "mclapply") {
    nworkers <- get.nworkers()
    results <- lapply(0:(nb_estim-1), function (i)  model_cox.id(i, covariates, event_col, duration_col, 
                                      analyzes_dir, model_name, logfile, n_boot = n_boot, 
                                      load_results = F, save_results = T, do_plot = F, save_rds = F, penalty = penalty))
    results <- as.data.frame(results)
    stopifnot(ncol(results) == nb_estim)
  } else if (parallel.method == "rslurm") {
    functions_to_export <- c("model_cox.id", "model_cox", "loglik_ratio_best_lambda", "get.best.lambda", "get.coefs.cox", 
                             "preprocess_data_cox", "normalize_data", "coxlasso_data", "preliminary_filter", "filter_dummies_iccc",
                             "get.clinical_features","predictSurvProb.bootstrap.coxnet", "predictSurvProb.selection.coxnet",
                             "selection.coxnet", "select.bolasso.features", "sample.selection.coxnet", "slurm_job_boot_coxnet",
                             "plot_cox", "plot_bootstrap", "pretty.labels", "pretty.label", "pretty.iccc",
                             "bootstrap.coxnet", "get.surv.formula", "get.ipcw.surv.formula")
    nb_max_slurm_jobs <- 40
    log_info(paste("Maximum number of slurm jobs:", nb_max_slurm_jobs))
    sopt <- list(time = "03:30:00", "ntasks" = 1, "cpus-per-task" = 1, 
                 partition = "cpu_med", mem = "20G")
    job_uuid <- stringr::str_split(uuid::UUIDgenerate(), "-")[[1]][1]
    print("Before the call of slurm_apply(), I print n_boot")
    print(paste("n_boot:", n_boot))
    sjob <- slurm_apply(function (i)  model_cox.id(i, covariates, event_col, duration_col,
                        analyzes_dir, model_name, logfile, n_boot = n_boot, screening_method = screening_method,
                        load_results = F, save_results = T, do_plot = T, save_rds = F, penalty = penalty),
                        data.frame(i = 0:(nb_estim-1)), 
                        nodes = nb_max_slurm_jobs, cpus_per_node = 1, processes_per_node = 1, 
                        global_objects = functions_to_export,
                        jobname = paste0("multiple_scores_cox_", nb_estim, "_runs_", job_uuid),
                        slurm_options = sopt)
    log_info("Jobs are submitted")
    list_results <- get_slurm_out(sjob, outtype = "raw", wait = T)
    results <- t(do.call("rbind", list_results))
    stopifnot(ncol(results) == nb_estim)
    cleanup_files(sjob, wait = T)
    log_info("End of all submitted jobs")
  }
  filename_results <- paste0(analyzes_dir, "coxph_R/", model_name, "/", nb_estim, "_runs_full_test_metrics.csv")
  rownames(results) <- index_results
  write.csv(results, file = filename_results, row.names = T)
  summary_results <- data.frame(Mean = apply(results, 1, mean), Std = apply(results, 1, sd)) 
  rownames(summary_results) <- index_results
  filename_summary_results <- paste0(analyzes_dir, "coxph_R/", model_name, "/", nb_estim, "_runs_test_metrics.csv")
  write.csv(summary_results, file = filename_summary_results, row.names = T)
}

# Plots of cox models : coefficients
plot_cox_coefs <- function(save_results_dir) {
  df_coefs <- read.csv(paste0(save_results_dir, "best_coefs.csv"))
  df_coefs <- df_coefs[abs(df_coefs$coefs) > 0, ]
  df_coefs <- df_coefs[order(-abs(df_coefs$coefs)), ]
  ggplot(df_coefs, aes(x = reorder(labels, abs(coefs)), y = coefs)) +
  geom_bar(stat = "identity") + coord_flip() 
  ggsave(paste0(save_results_dir, "coefs.png"), device = "png", dpi = 480)
}

# Plots of cox models: regularization path with CV error for lambda estimation
plot_cox_lambda_path <- function(save_results_dir) {
  # Mean errors of cross-validation
  cv.params <- read.csv(paste0(save_results_dir, "cv.csv"))
  cv.params.unique <- cv.params[order(-cv.params$non_zero_coefs, cv.params$penalty), ]
  cv.params.unique <- cv.params.unique[!duplicated(cv.params.unique$non_zero_coefs), ]
  mask_even = which((1:nrow(cv.params.unique)) %% 2 == 0)
  ggplot(cv.params, aes(x = penalty, y = mean_score)) +
  geom_ribbon(aes(ymin = mean_score - std_score, ymax = mean_score + std_score), fill = "blue", alpha = 0.5) +
  geom_line(color = "blue") + geom_point(color = " red") +
  geom_vline(xintercept = cv.params[1, "penalty"], color = "orange") +
  geom_vline(xintercept = best.params[1, "penalty"], color = "darkgreen", linetype = "dotdash") +
  geom_point(data = cv.params.unique, aes(x = penalty, y = mean_score), color = "purple") +
  geom_text(data = cv.params.unique[mask_even,], aes(x = penalty, y = mean_score, label = non_zero_coefs), 
            size = 3, vjust = -1) +
  geom_text(data = cv.params.unique[-mask_even,], aes(x = penalty, y = mean_score, label = non_zero_coefs), 
            size = 3, vjust = 2) +
  scale_x_log10() +
  labs(x = "Penalty (log10)", y = "Mean score (1 - Cindex)")
  ggsave(paste0(save_results_dir, "cv_mean_error.png"), device = "png", dpi = 480)
  # Regularization path
  mat.coefs <- read.csv(paste0(save_results_dir, "mat_coefs.csv"))
  df.mat.coefs <- melt(mat.coefs)
  colnames(df.mat.coefs) <- c("lambda", "varname", "coef")
  df.mat.coefs[, "varname"] <- pretty.labels(as.character(df.mat.coefs[["varname"]]))
  first.lambda <- rownames(mat.coefs)[nrow(mat.coefs)]
  ggplot(df.mat.coefs, aes(x = lambda, y = coef, color = varname)) + geom_line() + 
  xlab("Penalty (log10)") + ylab("Coefficient") + theme(legend.position = "none") +
  geom_text(data = subset(df.mat.coefs, lambda == first.lambda & varname %in% names.nonnull.coefs), 
            aes(x = lambda, y = coef, label = pretty.labels(as.character(varname))), size = 3, hjust = 1) +
  coord_cartesian(clip = 'off') +
  theme(plot.margin = unit(c(1,1,1,1), "lines"), axis.title.y = element_text(margin = margin(0,2,0,0, "cm"))) + 
  geom_vline(xintercept = cv.params[1, "penalty"], color = "orange") +
  geom_vline(xintercept = best.params[1, "penalty"], color = "darkgreen", linetype = "dotdash") +
  scale_x_log10()
  ggsave(paste0(save_results_dir, "regularization_path.png"), device = "png", dpi = 480)
}

# Plot bootstrap selected coefficients + error statistics
plot_bootstrap_cox <- function(save_results_dir, B) {
    bootstrap_selected_features <- read.csv(paste0(save_results_dir, "bootstrap_selected_features.csv"))
    freq_selection <- colSums(bootstrap_selected_features[, colSums(bootstrap_selected_features) > 0])
    df.coefs = data.frame(labels = pretty.labels(names(freq_selection)), coefs = freq_selection)
    df.coefs = df.coefs[order(-df.coefs$coefs), ]
    if (nrow(df.coefs) > 30) df.coefs <- df.coefs[1:30, ]
    ggplot(df.coefs, aes(x = reorder(labels, coefs), y = coefs)) + geom_bar(stat = "identity") + 
    ggtitle("Number of times the features are selecteds (30 best)") + coord_flip()
    ggsave(paste0(save_results_dir, "freq_selected_features.png"), device = "png", dpi = 480)
}

