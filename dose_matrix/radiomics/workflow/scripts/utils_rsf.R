
suppressPackageStartupMessages({
library("stringr", quietly = TRUE)
library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("randomForestSRC", quietly = TRUE)
library("pec", quietly = TRUE)
library("logger", quietly = TRUE)
library("Hmisc", quietly = TRUE)
library("parallel", quietly = TRUE)
library("doParallel", quietly = TRUE)
library("rslurm", quietly = TRUE)
})

source("workflow/scripts/utils_radiopreditool.R")

# Learning a model

model_rsf.id <- function(id_set, covariates, event_col, duration_col, analyzes_dir, model_name, rsf_logfile) {
    df_trainset <- read.csv(paste0(analyzes_dir, "datasets/trainset_", id_set, ".csv.gz"), header = TRUE)
    df_testset <- read.csv(paste0(analyzes_dir, "datasets/testset_", id_set, ".csv.gz"), header = TRUE)
    log_appender(appender_file(rsf_logfile, append = TRUE))
    model_rsf(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, rsf_logfile, 
              save_results = FALSE, load_results = TRUE, level = INFO, id_set = id_set)
}

model_rsf <- function(df_trainset, df_testset, covariates, event_col, duration_col, 
                      analyzes_dir, model_name, rsf_logfile, id_set = "", cv_nfolds = 5,  
                      save_results = TRUE, load_results = FALSE, level = INFO,  
                      ntrees = c(100, 300, 1000), nodesizes = c(15, 50), nsplits = c(700)) {
    log_threshold(level)
    log_appender(appender_file(rsf_logfile, append = TRUE))
    run_parallel <- load_results & !save_results
    save_results_dir <- paste0(analyzes_dir, "rsf/", model_name, "/")
    dir.create(save_results_dir, showWarnings = FALSE)
    ## Preprocessing sets
    filter_train <- !duplicated(as.list(df_trainset[covariates])) & 
                    unlist(lapply(df_trainset[covariates], 
                                  function(col) { length(unique(col)) > 1 } ))
    filtered_covariates <- names(filter_train)[filter_train]
    df_model_train <- df_trainset[,c(event_col, duration_col, filtered_covariates)]
    df_model_test <- df_testset[,c(event_col, duration_col, filtered_covariates)]
    df_model_train <- na.omit(df_model_train)
    df_model_test <- na.omit(df_model_test)
    log_info(paste0("(", id_set, ") ", "Model name: ", model_name))
    log_info(paste0("(", id_set, ") ", "Covariates (", length(filtered_covariates),"):"))
    if (!run_parallel) {
        log_info(paste0(filtered_covariates, collapse = ", "))
        log_info(paste0("Trained:", nrow(df_model_train), "samples"))
        log_info(paste0("Testset: ", nrow(df_model_test), " samples"))
        log_info("NAs are omitted")
    }
    formula_model <- get.surv.formula(event_col, filtered_covariates, duration_col = duration_col)
    final.time <- floor(min(max(df_model_train[[duration_col]]), max(df_model_test[[duration_col]]), 60))
    pred.times <- seq(1, final.time, by = 1)
    if (load_results) {
        params.best <- read.csv(paste0(save_results_dir, "cv.csv"))[1,]
    } else {
        params.df <- create.params.df(ntrees, nodesizes, nsplits)
        # test.params.df <- data.frame(ntrees = c(5), nodesizes = c(50), nsplits = c(10))
        parallel.method <- `if`(Sys.getenv("SLURM_NTASKS") == "", "rfsrc", "rslurm")
        cv.params <- cv.rsf(formula_model, df_model_train, params.df, event_col, rsf_logfile, nfolds = cv_nfolds, 
                            parallel.method = parallel.method, pred.times = pred.times, error.metric = "cindex")
    }
    if (save_results) {
        write.csv(cv.params, file = paste0(save_results_dir, "cv.csv"), row.names = FALSE)
        params.best <- cv.params[1,]
    }
    if (!run_parallel) {
        log_info("Best params:")
        log_info(toString(names(params.best)))
        log_info(toString(params.best))
    } 
    rsf.best <- rfsrc(formula_model, data = df_model_train, 
                      ntree = params.best$ntree, nodesize = params.best$nodesize, nsplit = params.best$nsplit)
    # Predictions
    rsf.survprob.train <- predictSurvProb(rsf.best, newdata = df_model_train, times = pred.times)
    rsf.survprob.oob <- predictSurvProbOOB(rsf.best, times = pred.times)
    rsf.survprob.test <- predictSurvProb(rsf.best, newdata = df_model_test, times = pred.times)
    rsf.pred.test <- predict(rsf.best, newdata = df_model_test)
    formula_ipcw <- get.ipcw.surv.formula(event_col, filtered_covariates)
    # C-index ipcw (cox + ipcw on clinical vars)
    rsf.cindex.ipcw.train <- pec::cindex(list("Best rsf" = rsf.best), formula = formula_ipcw, 
                                         data = df_model_train, cens.model = "cox")$AppCindex[["Best rsf"]]
    rsf.cindex.ipcw.oob <- pec::cindex(list("Best rsf" = rsf.best), formula = formula_ipcw, 
                                       data = df_model_train, cens.model = "cox", 
                                       method = "OutOfBagCindex")$AppCindex[["Best rsf"]]
    rsf.cindex.ipcw.test <- pec::cindex(list("Best rsf" = rsf.best), formula = formula_ipcw, 
                                        data = df_model_test, cens.model = "cox")$AppCindex[["Best rsf"]]
    # Harrell's C-index
    # 1 - rcorr.cens because the mortality (risk) is given instead of survival probability
    rsf.cindex.harrell.train <- 1-rcorr.cens(rsf.best$predicted, S = Surv(df_model_train[[duration_col]], 
                                                                          df_model_train[[event_col]]))[["C Index"]]
    rsf.cindex.harrell.oob <- 1-rcorr.cens(rsf.best$predicted.oob, S = Surv(df_model_train[[duration_col]], 
                                                                            df_model_train[[event_col]]))[["C Index"]]
    rsf.cindex.harrell.test <- 1-rcorr.cens(rsf.pred.test$predicted, S = Surv(df_model_test[[duration_col]], 
                                                                              df_model_test[[event_col]]))[["C Index"]]
    # Cindex rfsrc
    rsf.err.oob <- get.cindex(rsf.best$yvar[[duration_col]], rsf.best$yvar[[event_col]], rsf.best$predicted.oob)
    rsf.err.train <- get.cindex(rsf.best$yvar[[duration_col]], rsf.best$yvar[[event_col]], rsf.best$predicted)
    rsf.err.test <- get.cindex(rsf.pred.test$yvar[[duration_col]], rsf.pred.test$yvar[[event_col]], rsf.pred.test$predicted)
    if (!run_parallel) {
        log_info(paste0("Harrell's C-index on trainset: ", rsf.cindex.harrell.train))
        log_info(paste0("Harrell's C-index OOB trainset: ", rsf.cindex.harrell.oob))
        log_info(paste0("Harrell's C-index on testset: ", rsf.cindex.harrell.test))
        log_info(paste0("rfsrc C-index on trainset: ", 1-rsf.err.train))
        log_info(paste0("rfsrc C-index OOB trainset: ", 1-rsf.err.oob))
        log_info(paste0("rfsrc C-index on testset: ", 1-rsf.err.test))
        log_info(paste0("IPCW C-index on trainset: ", rsf.cindex.ipcw.train))
        log_info(paste0("IPCW C-index OOB trainset: ", rsf.cindex.ipcw.oob))
        log_info(paste0("IPCW C-index on testset: ", rsf.cindex.ipcw.test))
    }
    # IBS
    # Z normalisation for Breslow estimator of pec
    # means_train <- as.numeric(lapply(df_model_train[filtered_covariates], mean))
    # stds_train <- as.numeric(lapply(df_model_train[filtered_covariates], sd))
    # df_model_train_norm <- data.frame(df_model_train)
    # df_model_train_norm[, filtered_covariates] <- scale(df_model_train[filtered_covariates], center = means_train, scale = stds_train)
    rsf.perror.train <- pec(object = list("train"=rsf.survprob.train, "oob"=rsf.survprob.oob), 
                            formula = formula_ipcw, data = df_model_train, 
                            cens.model = "cox", 
                            times = pred.times, start = pred.times[1], 
                            exact = FALSE, reference = FALSE)
    rsf.perror.test <- pec(object= list("test"=rsf.survprob.test), 
                           formula = formula_ipcw, data = df_model_test, 
                           cens.model = "cox", 
                           times = pred.times, start = pred.times[1], 
                           exact = FALSE, reference = FALSE)
    rsf.bs.final.train <- tail(rsf.perror.train$AppErr$train, 1)
    rsf.bs.final.oob <- tail(rsf.perror.train$AppErr$oob, 1)
    rsf.bs.final.test <- tail(rsf.perror.test$AppErr$test, 1)
    rsf.ibs.train <- crps(rsf.perror.train)[1]
    rsf.ibs.oob <- crps(rsf.perror.train)[2]
    rsf.ibs.test <- crps(rsf.perror.test)[1]
    if (!run_parallel) {
        log_info(paste0("BS at 60 on trainset: ", rsf.bs.final.train))
        log_info(paste0("BS OOB at 60 on trainset: ", rsf.bs.final.oob))
        log_info(paste0("BS at 60 on testset: ", rsf.bs.final.test))
        log_info(paste0("IBS on trainset: ", rsf.ibs.train))
        log_info(paste0("IBS OOB on trainset: ", rsf.ibs.oob))
        log_info(paste0("IBS on testset: ", rsf.ibs.test))
    }
    results_train <- c(rsf.cindex.harrell.train, rsf.cindex.ipcw.train, 
                       rsf.bs.final.train, rsf.ibs.train)
    results_test <- c(rsf.cindex.harrell.test, rsf.cindex.ipcw.test, 
                      rsf.bs.final.test, rsf.ibs.test)
    log_info(paste0("(", id_set, ") ", "Train:", results_train[1], "&", results_train[2], "&", results_train[3], "&", results_train[4]))
    log_info(paste0("(", id_set, ") ", "Test:", results_test[1], "&", results_test[2], "&", results_test[3], "&", results_test[4]))
    df_results <- data.frame(Train = results_train, Test = results_test)
    rownames(df_results) <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
    if (save_results) {
        write.csv(df_results, file = paste0(save_results_dir, "metrics.csv"), row.names = TRUE)
        saveRDS(rsf.best, file = paste0(save_results_dir, "model.rds"))
    }
    log_threshold(INFO)
    results_test
}

# Run multiple scores estimation for a RSF model with presaved train / test sets in parallel
parallel_multiple_scores_rsf <- function(nb_estim, covariates, event_col, duration_col, analyzes_dir, 
                                         model_name, logfile, parallel.method = "mclapply") {
  stopifnot(parallel.method %in% c("mclapply", "rslurm"))
  index_results <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
  if (parallel.method == "mclapply") {
    nworkers <- get.nworkers()
    results <- mclapply(0:(nb_estim-1), function (i)  model_rsf.id(i, covariates, event_col, duration_col, 
                                        analyzes_dir, model_name, logfile), mc.cores = nworkers)
    results <- as.data.frame(results)
  } else if (parallel.method == "rslurm") {
    nb_max_slurm_jobs <- 40
    log_info(paste("Maximum number of slurm jobs:", nb_max_slurm_jobs))
    sopt <- list(time = "02:00:00", "ntasks" = 1, "cpus-per-task" = 1, 
                 partition = "cpu_med", mem = "20G")
    sjob <- slurm_apply(function (i)  model_rsf.id(i, covariates, event_col, duration_col, 
                        analyzes_dir, model_name, logfile, penalty = "none"), 
                        data.frame(i = 0:(nb_estim-1)), 
                        nodes = nb_max_slurm_jobs, cpus_per_node = 1, processes_per_node = 1, 
                        global_objects = c("model_rsf.id", "model_rsf", 
                                           "get.surv.formula", "get.ipcw.surv.formula", "bootstrap.undersampling", 
                                           "predictSurvProbOOB", "create.params.df", "cv.rsf", "get.param.cv.error"), 
                        slurm_options = sopt)
    log_info("Jobs are submitted")
    list_results <- get_slurm_out(sjob, outtype = "raw", wait = T)
    results <- do.call("rbind", list_results)
    cleanup_files(sjob, wait = T)
    log_info("End of all submitted jobs")
  }
  df_results <- data.frame(Mean = apply(results, 1, mean), Std = apply(results, 1, sd)) 
  rownames(df_results) <- index_results
  filename_results <- paste0(analyzes_dir, "coxph_R/", model_name, "/", nb_estim, "_runs_test_metrics.csv")
  write.csv(df_results, file = filename_results, row.names = TRUE)
}

# Plot of features importances with VIMP method
plot_vimp <- function(rsf.obj, analyzes_dir, model_name) {
    new_labels <- pretty.labels(rsf.obj$xvar.names)
    subs.rsf.obj <- subsample(rsf.obj, B = 100)
    save_results_dir <- paste0(analyzes_dir, "rsf/", model_name, "/")
    png(paste0(save_results_dir, "rsf_vimp.png"), width = 1250, height = 1600, res = 70)
    par(oma = c(0.5, 10, 0.5, 0.5))
    par(cex.axis = 2.0, cex.lab = 2.0, cex.main = 2.0, mar = c(6.0,17,1,1), mgp = c(4, 1, 0))
    pmax = 30
    p = length(new_labels)
    xlab <- `if`(p < pmax, paste0("Variable Importance (x 100) -", p, "features"), 
                 paste0("Variable Importance (x 100) -", pmax, "best features"))
    new.plot.subsample.rfsrc(subs.rsf.obj, xlab = xlab, cex = 1.25, ylab = new_labels, pmax = pmax)
    dev.off()
}

# Brier score computation for OOB samples
predictSurvProbOOB <- function(object, times, ...){
    ptemp <- predict(object, importance="none",...)$survival.oob
    pos <- prodlim::sindex(jump.times=object$time.interest,eval.times=times)
    p <- cbind(1,ptemp)[,pos+1,drop=FALSE]
    if (NROW(p) != dim(ptemp)[1] || NCOL(p) != length(times))
        stop(paste0("\nPrediction matrix has wrong dimensions:\nRequested newdata x times: ",
                   NROW(newdata)," x ",length(times),"\nProvided prediction matrix: ",NROW(p)," x ",NCOL(p),"\n\n"))
    p
}

# Create a data frame with the hyperparameters to test in CV
create.params.df <- function(ntrees, nodesizes, nsplits) {
    params.df <- data.frame(ntree=integer(), nodesize=integer(), nsplit=integer())
    for (ntree in ntrees) {
        for (nodesize in nodesizes) {
            for (nsplit in nsplits) {
                params.df <- rbind(params.df, data.frame(ntree=ntree, nodesize=nodesize, nsplit=nsplit))
            }
        }
    }
    params.df
}

# Cross-validation for estimation of RSF's hyperparameters
cv.rsf <- function(formula, data, params.df, event_col, rsf_logfile, 
                   duration_col = "survival_time_years", parallel.method = "rfsrc", nfolds = 5, 
                   pred.times = seq(5, 50, 5), error.metric = "ibs", bootstrap.strategy = NULL) {
    stopifnot(parallel.method %in% c("rfsrc", "rslurm"))
    nbr.params <- nrow(params.df)
    folds <- createFolds(factor(data[[event_col]]), k = nfolds, list = FALSE)
    final.time <- pred.times[length(pred.times)]
    minmax.times <- lapply(1:nfolds, function(i) { 
                            fold.index = which(folds == i) 
                            min(max(data[fold.index, duration_col]), max(data[-fold.index, duration_col]), final.time)
                        })
    final.time.folds <- floor(min(unlist(minmax.times)))
    pred.times.folds <- pred.times[pred.times <= final.time.folds]
    if (parallel.method == "rfsrc") {
      log_info(paste("Running CV with rfsrc using", getOption("rf.cores"), "workers")) 
      cv.params.df <- mclapply(1:nbr.params, function (i) { 
                                 get.param.cv.error(i, formula, data, params.df, folds,
                                                    bootstrap.strategy, error.metric, pred.times.folds, rsf_logfile) }, 
                               mc.cores = 1)
      cv.params.df <- as.data.frame(t(as.data.frame(cv.params.df)))
    } else if (parallel.method == "rslurm") {
      nb_max_slurm_jobs <- 40
      log_info(paste("Maximum number of slurm jobs:", nb_max_slurm_jobs))
      sopt <- list(time = "02:00:00", "ntasks" = 1, "cpus-per-task" = 1, 
                   partition = "cpu_med", mem = "20G")
      sjob <- slurm_apply(function(idx.row.param) get.param.cv.error(idx.row.param, formula, data,
                          params.df, folds, bootstrap.strategy, error.metric, pred.times.folds, rsf_logfile), 
                          data.frame(idx.row.param = 1:nbr.params), 
                          nodes = nb_max_slurm_jobs, cpus_per_node = 1, processes_per_node = 1, 
                          global_objects = c("get.param.cv.error", "get.ipcw.surv.formula", 
                                             "get.clinical_features", "bootstrap.undersampling"),
                          slurm_options = sopt)
      log_info("Jobs are submitted")
      list.cv.errors <- get_slurm_out(sjob, outtype = "raw", wait = T)
      cv.params.df <- do.call("rbind", list.cv.errors)
      cleanup_files(sjob, wait = T)
      log_info("End of all submitted jobs")
    }
    rownames(cv.params.df) <- NULL
    colnames(cv.params.df) <- c(colnames(params.df), "IBS", "Harrel Cindex", "IPCW Cindex", "Error")
    cv.params.df[order(cv.params.df$Error),]
}

# The job for one hyperparameter vector in cross-validation
get.param.cv.error <- function(idx.row, formula, data, params.df, folds, 
                               bootstrap.strategy, error.metric, pred.times, rsf_logfile) {
    log_appender(appender_file(rsf_logfile, append = TRUE))
    log_info(paste("Error CV:", error.metric))
    duration_col <- all.vars(formula[[2]])[1]
    event_col <- all.vars(formula[[2]])[2]
    elapsed.time <- system.time({
    row <- params.df[idx.row,]
    nbr.params <- nrow(params.df)
    log_info(paste(Sys.getpid(), "- Parameters", idx.row, "/", nbr.params, ": Begin"))
    nfolds <- length(unique(folds))
    final.time.bs <- pred.times[length(pred.times)]
    cindex.folds <- rep(0.0, nfolds)
    cindex.ipcw.folds <- rep(0.0, nfolds)
    ibs.folds <- rep(0.0, nfolds)
    formula_ipcw <- get.ipcw.surv.formula(event_col, colnames(data))
    for (i in 1:nfolds) {
        fold.index <- which(folds == i)
        fold.test <- data[fold.index,]
        fold.train <- data[-fold.index,]
        # RSF model
        if (is.null(bootstrap.strategy)) {
            rsf.fold.bootstrap <- "by.root"
            rsf.fold.samp <- NULL
        } else if (bootstrap.strategy == "undersampling") {
            rsf.fold.bootstrap <- "by.user"
            rsf.fold.samp <- bootstrap.undersampling(fold.train, row$ntree)
        }
        rsf.fold <- rfsrc(formula, data = fold.train, 
                          ntree = row$ntree, nodesize = row$nodesize, nsplit = row$nsplit, 
                          bootstrap = rsf.fold.bootstrap, samp = rsf.fold.samp)
        # C-index
        fold.test.pred <- predict(rsf.fold, newdata = fold.test)
        cindex.fold <- 1 - get.cindex(fold.test[[duration_col]], fold.test[[event_col]], fold.test.pred$predicted)
        cindex.folds[i] <- cindex.fold
        # IPCW C-index
        cindex.ipcw.fold <- pec::cindex(list("RSF test fold" = rsf.fold), formula = formula_ipcw, 
                                        cens.model = "cox", data = fold.test)$AppCindex[["RSF test fold"]]
        cindex.ipcw.folds[i] <- cindex.ipcw.fold
        # IBS
        fold.test.pred.bs <- predictSurvProb(rsf.fold, newdata = fold.test, times = pred.times)
        perror = pec(object = fold.test.pred.bs, 
                     data = fold.test, formula = formula_ipcw, 
                     cens.model = "cox", 
                     times = pred.times, start = pred.times[1], 
                     exact = FALSE, reference = FALSE)
        ibs.fold <- crps(perror, times = final.time.bs)[1]
        ibs.folds[i] <- ibs.fold
    }
    if (error.metric == "ibs") {
        param.error <- mean(ibs.folds)
    } else if (error.metric == "cindex") {
        param.error <- mean(1-cindex.folds)
    } else if (error.metric == "cindex.ipcw") {
        param.error <- mean(1-cindex.ipcw.folds)
    } else {
        stop("Error metric not implemented")
    }
    })[["elapsed"]]
    log_info(paste(Sys.getpid(), "-", elapsed.time, "s -  Parameters ", idx.row, "/", nbr.params, 
                   " (", row$ntree, ",", row$nodesize, ",", row$nsplit, ") : ", param.error))
    c(unlist(row), mean(ibs.folds), mean(cindex.folds), mean(cindex.ipcw.folds), param.error)
}

# Refit RSF with stored best parameters
refit.best.rsf <- function(file_trainset, file_testset, covariates, event_col, duration_col, 
                           analyzes_dir, model_name = "") {
    # Prepare sets
    df_model_train <- read.csv(file_trainset, header = TRUE)[,c(event_col, duration_col, covariates)]
    df_model_test <- read.csv(file_testset, header = TRUE)[,c(event_col, duration_col, covariates)]
    df_model_train <- na.omit(df_model_train)
    df_model_test <- na.omit(df_model_test)
    # Fit RSF
    formula_model <- get.surv.formula(event_col, covariates, duration_col = duration_col)
    params.best <- read.csv(paste0(save_results_dir, "cv.csv"))[1,]
    rsf.best <- rfsrc(formula_model, data = df_model_train, 
                      ntree = params.best$ntree, nodesize = params.best$nodesize, nsplit = params.best$nsplit)
    # Predictions / metrics
    final.time <- floor(min(max(df_model_train[[duration_col]]), max(df_model_test[[duration_col]]), 60))
    pred.times <- seq(1, final.time, by = 1)
    rsf.pred.test <- predict(rsf.best, newdata = df_model_test)
    rsf.survprob.test <- predictSurvProb(rsf.best, newdata = df_model_test, times = pred.times)
    rsf.cindex.ipcw.test <- pec::cindex(list("Best rsf" = rsf.best), formula_model, data = df_model_test)$AppCindex[["Best rsf"]]
    rsf.cindex.harrell.test <- 1-rcorr.cens(rsf.pred.test$predicted, S = Surv(df_model_test[[duration_col]], df_model_test[[event_col]]))[["C Index"]]
    rsf.perror.test <- pec(object= list("test" = rsf.survprob.test), 
                           formula = formula_model, data = df_model_test, 
                           cens.model = "marginal", 
                           times = pred.times, start = pred.times[1], 
                           exact = FALSE, reference = FALSE)
    rsf.bs.final.test <- tail(rsf.perror.test$AppErr$test, 1)
    rsf.ibs.test <- crps(rsf.perror.test)[1]
    results_test <- c(rsf.cindex.harrell.test, rsf.cindex.ipcw.test, rsf.bs.final.test, rsf.ibs.test)
    results_test
}

# Refit RSF with a database
refit.best.rsf.id <- function(id_set, covariates, event_col, duration_col, 
                              analyzes_dir, model_name = "") {
    log_info(id_set)
    file_trainset <- paste0(analyzes_dir, "datasets/trainset_", id_set, ".csv.gz")
    file_testset <- paste0(analyzes_dir, "datasets/testset_", id_set, ".csv.gz")
    refit.best.rsf(file_trainset, file_testset, covariates, event_col, duration_col, analyzes_dir, model_name = model_name)
}

# Under-sampling
bootstrap.undersampling <- function(data,ntree) {
    nsamples <- dim(data)[1]
    index.data.event <- which(data[[event_col]]==1)
    index.data.censored <- which(data[[event_col]]==0)
    sampsize <- floor(0.632*length(index.data.event))
    bootstrap <- array(0, dim=c(nsamples,ntree))
    for (i in 1:ntree) {
        index.bootstrap.event <- sample(index.data.event, sampsize, replace = FALSE)
        index.bootstrap.censored <- sample(index.data.censored, 10*sampsize, replace = FALSE)
        bootstrap[c(index.bootstrap.event,index.bootstrap.censored),i] <- 1
    }
    bootstrap
}

# Plot of features importances for RSF
new.plot.subsample.rfsrc <- function(x, alpha = .01,
                                     standardize = TRUE, normal = TRUE, jknife = TRUE,
                                     target, m.target = NULL, pmax = 75, main = "", 
                                     ...) {
    ##--------------------------------------------------------------
    ##
    ## was subsampling or double-bootstrap used?
    ##
    ##--------------------------------------------------------------
    if (sum(c(grepl("rfsrc", class(x))), grepl("subsample", class(x))) == 2) {
        subsample <- TRUE
    }
    else if (sum(c(grepl("rfsrc", class(x))), grepl("bootsample", class(x))) == 2) {
        subsample <- FALSE
    }
    else {
        stop("object must be obtained from call to 'subsample' function")
    }
    ##--------------------------------------------------------------
    ##
    ## coerce the (potentially) multivariate rf object
    ##
    ##--------------------------------------------------------------
    m.target <- randomForestSRC:::get.univariate.target(x$rf, m.target)
    x$rf <- randomForestSRC:::coerce.multivariate(x$rf, m.target)
    ##--------------------------------------------------------------
    ##
    ## family specific details
    ## - set the target if not supplied
    ## - assign pretty labels for the horizontal axis
    ##
    ##--------------------------------------------------------------
    fmly <- x$rf$family
    ## labels
    if (standardize) {
        if (fmly == "regr") {
            xlab <- "standardized vimp"
        }
        else {
            xlab <- "100 x vimp"
        }
    }
    else {
        xlab <- "vimp"
    }
    ## extract vimp column names - be careful if this is multivariate
    if (is.null(m.target)) {  
        vmp.col.names <- colnames(x$vmp[[1]])
    }
    else {
        vmp.col.names <- colnames(x$vmp[[m.target]])
    }
    if (fmly == "regr" || fmly == "surv") {
        target <- 0
        xlab <- paste(xlab, " (", vmp.col.names, ")", sep = "")
    }
    else if (fmly == "class") {
        if (missing(target)) {
            target <- 0
        }
        else {
            yvar.levels <- levels(x$rf$yvar)
            if (is.character(target)) {
                target <- match(match.arg(target, yvar.levels), yvar.levels)
            }
            else {
                if (target < 0 || target > length(yvar.levels)) {
                    stop("'target' is specified incorrectly")
                }
            }
        }
        xlab <- paste(xlab, " (", vmp.col.names[1 + target], ")", sep = "")
    }
    else if (fmly == "surv-CR") {    
        if (missing(target)) {
            target <- 0
        }
        else {
            n.event <- length(get.event.info(x$rf)$event.type)
            if (target < 1 || target > n.event) {
                stop("'target' is specified incorrectly")
            }
            target <- target - 1
        }
        xlab <- paste(xlab, " (", vmp.col.names[1 + target], ")", sep = "")
    }
    ##--------------------------------------------------------------
    ##
    ## over-ride x label if the user has supplied their own
    ##
    ##--------------------------------------------------------------
    if (!is.null(list(...)$xlab)) {
        xlab <- list(...)$xlab
    }
    ##--------------------------------------------------------------
    ##
    ## extract necessary objects
    ##
    ##--------------------------------------------------------------
    if (subsample) {
        oo <- extract.subsample(x, alpha = alpha, target = target, m.target = m.target, standardize = standardize)
        boxplot.dta <- oo$boxplot.dta
    }
    else {
        oo <- extract.bootsample(x, alpha = alpha, target = target, m.target = m.target, standardize = standardize)
        boxplot.dta <- oo[[1]]
    }
    if (normal) {
        if (subsample && jknife) {
            ci <- oo$ci.jk.Z
        }
        else {
            ci <- oo$ci.Z
        }
    }
    else {
        ci <- oo$ci
    }
    ##--------------------------------------------------------------
    ##
    ## trim the data if too many variables
    ##
    ##--------------------------------------------------------------
    p <- ncol(boxplot.dta)
    pend <- min(p, pmax)
    o.pt <- order(ci[1, ], decreasing = TRUE)[1:pend]
    boxplot.dta <- boxplot.dta[, o.pt]
    ci <- ci[, o.pt]
    ##--------------------------------------------------------------
    ##
    ## skeleton boxplot 
    ##
    ##--------------------------------------------------------------
    bp <- boxplot(boxplot.dta,
                  yaxt="n",
                  outline = FALSE,
                  horizontal = TRUE,
                  plot = FALSE)
    bp$stats <- ci
    col.pt <- bp$stats[1, ] > 0 
    col.pt[is.na(col.pt)] <- FALSE
    colr <- c("blue", "red")[1 + col.pt]
    ##--------------------------------------------------------------
    ##
    ## finesse ... unnamed options to be passed to bxp and axis
    ##
    ##--------------------------------------------------------------
    ## pull the unnamed options
    dots <- list(...)
    bxp.names <- c(names(formals(bxp)),
                   "xaxt", "yaxt", "las", "cex.axis", 
                   "col.axis", "cex.main",
                   "col.main", "sub", "cex.sub", "col.sub", 
                   "ylab", "cex.lab", "col.lab")
    axis.names <- formals(axis)
    axis.names$tick <- axis.names$las <- axis.names$labels <- NULL
    axis.names <- c(names(axis.names), "cex.axis") 
    ## override xlab
    if (!is.null(dots$xlab)) {
        xlab <- dots$xlab
    }
    ## overlay ylab when user mistakenly uses it  
    if (!is.null(dots$ylab) && length(dots$ylab) == p) {
        bp$names <- dots$ylab[o.pt]
        bxp.names <- bxp.names[bxp.names != "ylab"]
    }
    ## overlay names
    if (!is.null(dots$names)) {
        bp$names <- dots$names[o.pt]
    }
    ##--------------------------------------------------------------
    ##
    ## draw the core bxp plot
    ##
    ##--------------------------------------------------------------
    do.call("bxp", c(list(z = bp, main = main, xlab = xlab, 
                          boxfill = colr, xaxt = "n", yaxt = "n",
                          outline = FALSE, horizontal = TRUE),
                 dots[names(dots) %in% bxp.names]))
    do.call("axis", c(list(side = 1, at = pretty(c(bp$stats)), tick = .02), dots[names(dots) %in% axis.names]))
    do.call("axis", c(list(side = 2, at = 1:length(bp$names), labels = bp$names,
                           las = 2, tick = FALSE), dots[names(dots) %in% axis.names]))
    abline(h = 1:length(bp$names), col = gray(.9), lty = 1)
    abline(v = 0, lty = 1, lwd = 1.5, col = gray(.8))
    bxp(bp, boxfill=colr,xaxt="n",yaxt="n",
        outline=FALSE,horizontal=TRUE,add=TRUE,
        whisklty=1,whisklwd=2)
}

