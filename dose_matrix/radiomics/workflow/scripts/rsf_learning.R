
options(show.error.locations = TRUE, error=traceback)

library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("randomForestSRC", quietly = TRUE)
library("pec", quietly = TRUE)
library("Hmisc", quietly = TRUE)
library("logger", quietly = TRUE)
library("parallel", quietly = TRUE)

source("workflow/scripts/utils_rsf.R")

model_rsf <- function(df_trainset, df_testset, covariates, event_col, duration_col, rsf_logfile) {
    log_appender(appender_file(rsf_logfile, append = TRUE))
    df_model_train <- df_trainset[,c(event_col, duration_col, covariates)]
    df_model_test <- df_testset[,c(event_col, duration_col, covariates)]
    # df_model_train[is.na(df_model_train)] <- -1
    # df_model_test[is.na(df_model_test)] <- -1
    df_model_train <- na.omit(df_model_train)
    df_model_test <- na.omit(df_model_test)
    log_info(paste("Covariates:", paste(covariates, collapse = ", ")))
    log_info(paste("Trained:", nrow(df_model_train), "samples"))
    log_info(paste("Testset: ", nrow(df_model_test), " samples)", sep = ""))
    log_info("NAs are omitted")
    formula_model <- get.surv.formula(event_col, covariates, duration_col = duration_col)
    pred.times <- seq(1, 60, by = 1)
    final.time <- tail(pred.times, 1)
    ntrees <- c(50, 100, 300, 1000, 2000)
    nodesizes <- c(15, 50, 100)
    nsplits <- c(10, 700)
    params.df <- create.params.df(ntrees, nodesizes, nsplits)
    test.params.df <- data.frame(ntrees = c(5), nodesizes = c(50), nsplits = c(10))
    cv.params <- cv.rsf(formula_model, df_model_train, test.params.df, event_col, rsf_logfile, pred.times = pred.times, error.metric = "cindex.ipcw")
    # Best RSF
    params.best <- cv.params[1,]
    log_info("Best params:")
    log_info(toString(names(params.best)))
    log_info(toString(params.best))
    rsf.best <- rfsrc(formula_model, data = df_model_train, ntree = params.best$ntree, nodesize = params.best$nodesize, nsplit = params.best$nsplit)
    # Predictions
    rsf.survprob.train <- predictSurvProb(rsf.best, newdata = df_model_train, times = pred.times)
    rsf.survprob.oob <- predictSurvProbOOB(rsf.best, times = pred.times)
    rsf.survprob.test <- predictSurvProb(rsf.best, newdata = df_model_test, times = pred.times)
    rsf.pred.test <- predict(rsf.best, newdata = df_model_test)
    # C-index ipcw (censored free, marginal = KM)
    rsf.cindex.ipcw.train <- pec::cindex(list("Best rsf" = rsf.best), formula_model, data = df_model_train)$AppCindex[["Best rsf"]]
    rsf.cindex.ipcw.oob <- pec::cindex(list("Best rsf" = rsf.best), formula_model, data = df_model_train, method = "OutOfBagCindex")$AppCindex[["Best rsf"]]
    rsf.cindex.ipcw.test <- pec::cindex(list("Best rsf" = rsf.best), formula_model, data = df_model_test)$AppCindex[["Best rsf"]]
    # Harrell's C-index
    rsf.cindex.harrell.train <- 1-rcorr.cens(rsf.best$predicted, S = Surv(df_model_train[[duration_col]], df_model_train[[event_col]]))[["C Index"]]
    rsf.cindex.harrell.oob <- 1-rcorr.cens(rsf.best$predicted.oob, S = Surv(df_model_train[[duration_col]], df_model_train[[event_col]]))[["C Index"]]
    rsf.cindex.harrell.test <- 1-rcorr.cens(rsf.pred.test$predicted, S = Surv(df_model_test[[duration_col]], df_model_test[[event_col]]))[["C Index"]]
    # Cindex rfsrc
    rsf.err.oob <- get.cindex(rsf.best$yvar[[duration_col]], rsf.best$yvar[[event_col]], rsf.best$predicted.oob)
    rsf.err.train <- get.cindex(rsf.best$yvar[[duration_col]], rsf.best$yvar[[event_col]], rsf.best$predicted)
    rsf.err.test <- get.cindex(rsf.pred.test$yvar[[duration_col]], rsf.pred.test$yvar[[event_col]], rsf.pred.test$predicted)
    log_info(paste("Harrell's C-index on trainset: ", rsf.cindex.harrell.train))
    log_info(paste("Harrell's C-index OOB trainset: ", rsf.cindex.harrell.oob))
    log_info(paste("Harrell's C-index on testset: ", rsf.cindex.harrell.test))
    log_info(paste("IPCW C-index on trainset: ", rsf.cindex.ipcw.train))
    log_info(paste("IPCW C-index OOB trainset: ", rsf.cindex.ipcw.oob))
    log_info(paste("IPCW C-index on testset: ", rsf.cindex.ipcw.test))
    log_info(paste("rfsrc C-index on trainset: ", 1-rsf.err.train))
    log_info(paste("rfsrc C-index OOB trainset: ", 1-rsf.err.oob))
    log_info(paste("rfsrc C-index on testset: ", 1-rsf.err.test))
    # IBS
    rsf.perror.train <- pec(object= list("train" = rsf.survprob.train, "oob" = rsf.survprob.oob), formula = formula_model, data = df_model_train, 
                            times = pred.times, start = pred.times[0], exact = FALSE, reference = FALSE)
    rsf.perror.test <- pec(object= list("test" = rsf.survprob.test), formula = formula_model, data = df_model_test, 
                           times = pred.times, start = pred.times[0], exact = FALSE, reference = FALSE)
    rsf.bs.final.train <- tail(rsf.perror.train$AppErr$train, 1)
    rsf.bs.final.oob <- tail(rsf.perror.train$AppErr$oob, 1)
    rsf.bs.final.test <- tail(rsf.perror.test$AppErr$test, 1)
    rsf.ibs.train <- crps(rsf.perror.train)[1]
    rsf.ibs.oob <- crps(rsf.perror.train)[2]
    rsf.ibs.test <- crps(rsf.perror.test)[1]
    log_info(paste("BS at 60 on trainset: ", rsf.bs.final.train))
    log_info(paste("BS OOB at 60 on trainset: ", rsf.bs.final.oob))
    log_info(paste("BS at 60 on testset: ", rsf.bs.final.test))
    log_info(paste("IBS on trainset: ", rsf.ibs.train))
    log_info(paste("IBS OOB on trainset: ", rsf.ibs.oob))
    log_info(paste("IBS on testset: ", rsf.ibs.test))
    old_digits <- getOption("digits")
    options(digits = 2)
    log_info(paste("Train:", rsf.cindex.harrell.train , "&", rsf.cindex.ipcw.train, "&", rsf.bs.final.train, "&", rsf.ibs.train))
    log_info(paste("Test:", rsf.cindex.harrell.test , "&", rsf.cindex.ipcw.test, "&", rsf.bs.final.test, "&", rsf.ibs.test))
    options(digits = old_digits)
    rsf.best
}

plot_vimp <- function(rsf.obj, analyzes_dir, model_name) {
    new_labels <- pretty.labels(rsf.obj$xvar.names)
    subs.rsf.obj <- subsample(rsf.obj, B = 100)
    png(paste(analyzes_dir, "rsf_plots/rsf_vimp_", model_name, ".png", sep = ""), width = 1250, height = 1600, res = 70)
    par(oma = c(0.5, 10, 0.5, 0.5))
    par(cex.axis = 2.0, cex.lab = 2.0, cex.main = 2.0, mar = c(6.0,17,1,1), mgp = c(4, 1, 0))
    pmax = 30
    p = length(new_labels)
    xlab <- `if`(p < pmax, paste("Variable Importance (x 100) -", p, "features"), paste("Variable Importance (x 100) -", pmax, "best features"))
    new.plot.subsample.rfsrc(subs.rsf.obj, xlab = xlab, cex = 1.25, ylab = new_labels, pmax = pmax)
    dev.off()
}

rsf_learning <- function(file_trainset, file_testset, file_features, event_col, analyzes_dir, duration_col, suffix_model) {
    dir.create(paste(analyzes_dir, "rsf_plots/", sep = ""), showWarnings = FALSE)
    ntasks <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
    nworkers <- `if`(is.na(ntasks), parallel::detectCores(), ntasks)
    options(rf.cores = nworkers, mc.cores = nworkers)
    rsf_logfile <- paste(analyzes_dir, "rsf_", suffix_model, ".log", sep = "")
    if (file.exists(rsf_logfile)) { file.remove(rsf_logfile) }
    log_appender(appender_file(rsf_logfile, append = TRUE))
    log_info("Random Survival Forest learning")
    # Dataset
    df_trainset <- read.csv(file_trainset, header = TRUE)
    df_testset <- read.csv(file_trainset, header = TRUE)
    # Select subset of features due to feature elimination
    features <- `if`(file_features == "all", colnames(df_trainset), as.character(read.csv(file_features)[,1]))
    # Add "X" for R colname compatibility
    features <- as.character(lapply(features, function(x) { `if`(str_detect(substr(x, 1, 1), "[0-9]"), paste("X", x, sep = ""), x) }))
    df_trainset <- df_trainset[,features]
    df_testset <- df_testset[,features]
    clinical_vars <- get.clinical_features(colnames(df_trainset), event_col, duration_col)
    log_info(paste("Trainset file:", file_trainset, "with", nrow(df_trainset), "samples"))

    # Model 32X radiomics covariates
    log_info("Model 32X")
    cols_32X <- grep("^X32[0-9]{1}_", colnames(df_trainset), value = TRUE)
    covariates_32X <- c(clinical_vars, cols_32X, "has_radiomics")
    rsf.obj <- model_rsf(df_trainset, df_testset, covariates_32X, event_col, duration_col, rsf_logfile)
    plot_vimp(rsf.obj, analyzes_dir, paste("model_32X_", suffix_model, sep = ""))

    # Model 1320 radiomics covariates
    log_info("Model 1320")
    cols_1320 <- grep("^X1320_", colnames(df_trainset), value = TRUE)
    covariates_1320 <- c(clinical_vars, cols_1320, "has_radiomics")
    rsf.obj <- model_rsf(df_trainset, df_testset, covariates_1320, event_col, duration_col, rsf_logfile)
    plot_vimp(rsf.obj, analyzes_dir, paste("model_1320_", suffix_model, sep = ""))
}

# Script args
args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    file_trainset = args[1]
    file_testset = args[2]
    file_features <- args[3]
    event_col <- args[4]
    analyzes_dir <- args[5]
    suffix_model <- args[6]
    if (length(args) == 7) {
        duration_col <- args[7]
    } else {
        duration_col <- "survival_time_years"
    }

    log_threshold(INFO)
    rsf_learning(file_trainset, file_testset, file_features, event_col, analyzes_dir, duration_col, suffix_model)
} else {
    print("No arguments provided. Skipping.")
}

