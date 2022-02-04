
options(show.error.locations = TRUE, error=traceback)

library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("randomForestSRC", quietly = TRUE)
library("pec", quietly = TRUE)
library("logger", quietly = TRUE)

source("workflow/scripts/utils_rsf.R")

model_rsf <- function(df_trainset, df_testset, covariates, event_col, duration_col, rsf_logfile) {
    log_appender(appender_file(rsf_logfile, append = TRUE))
    df_model_train <- df_trainset[,c(event_col, duration_col, covariates)]
    df_model_train[is.na(df_model_train)] <- -1
    df_model_test <- df_testset[,c(event_col, duration_col, covariates)]
    df_model_test[is.na(df_model_test)] <- -1
    log_info(paste("Covariates:", paste(covariates, collapse = ", ")))
    log_info(paste("Trained on", nrow(df_model_train), "samples (NAs are filled with -1)"))
    log_info("Testset NAs are filled with -1")
    formula_model <- get.surv.formula(event_col, covariates, duration_col = duration_col)
    bs.times <- seq(5, 50, by = 5)
    test.params.df <- data.frame(ntree = c(1000), nodesize = c(20), nsplit = c(500))
    # ntrees <- c(10, 100, 200, 500, 1000, 2000)
    ntrees <- c(10, 100, 200)
    nodesizes <- c(15, 50, 100)
    nsplits <- c(10, 700)
    params.df <- create.params.df(ntrees, nodesizes, nsplits)
    cv.params <- cv.rsf(formula_model, df_model_train, params.df, event_col, rsf_logfile, pred.times = bs.times, error.metric = "ibs")
    # Best RSF
    params.best <- cv.params[1,]
    log_info("Best params:")
    log_info(typeof(params.best))
    log_info(toString(names(params.best)))
    log_info(toString(params.best))
    rsf.best <- rfsrc(formula_model, data = df_model_train, ntree = params.best$ntree, nodesize = params.best$nodesize, nsplit = params.best$nsplit)
    # C-index
    rsf.err_oob <- get.cindex(rsf.best$yvar[[duration_col]], rsf.best$yvar[[event_col]], rsf.best$predicted.oob)
    rsf.err <- get.cindex(rsf.best$yvar[[duration_col]], rsf.best$yvar[[event_col]], rsf.best$predicted)
    pred.testset <- predict(rsf.best, newdata = df_model_test)
    rsf.err.testset <- get.cindex(pred.testset$yvar[[duration_col]], pred.testset$yvar[[event_col]], pred.testset$predicted)
    log_info(paste("C-index on trainset: ", 1-rsf.err_oob))
    log_info(paste("C-index OOB on trainset: ", 1-rsf.err_oob))
    log_info(paste("C-index on testset: ", 1-rsf.err.testset))
    # IBS
    rsf.pred.bs <- predictSurvProb(rsf.best, newdata = df_model_train, times = bs.times)
    rsf.pred.oob.bs <- predictSurvProbOOB(rsf.best, times = bs.times)
    rsf.pred.bs.test <- predictSurvProb(rsf.best, newdata = df_model_test, times = bs.times)
    rsf.perror.train <- pec(object= list(rsf.pred.bs, rsf.pred.oob.bs), formula = formula_model, data = df_model_train, 
                            times = bs.times, start = bs.times[0], exact = FALSE, reference = FALSE)
    rsf.perror.test <- pec(object= list(rsf.pred.bs.test), formula = formula_model, data = df_model_test, 
                           times = bs.times, start = bs.times[0], exact = FALSE, reference = FALSE)
    log_info(paste("IBS on trainset: ", crps(rsf.perror.train)[1]))
    log_info(paste("IBS OOB on trainset: ", crps(rsf.perror.train)[2]))
    log_info(paste("IBS on testset: ", crps(rsf.perror.test)[1]))
}

rsf_learning <- function(file_trainset, file_testset, event_col, analyzes_dir, duration_col, rsf_name_logfile) {
    rsf_logfile <- paste(analyzes_dir, rsf_name_logfile, sep = "")
    if (file.exists(rsf_logfile)) { file.remove(rsf_logfile) }
    log_appender(appender_file(rsf_logfile, append = TRUE))
    log_info("Random Survival Forest learning")
    # Dataset
    df_trainset <- read.csv(file_trainset, header = TRUE)
    df_testset <- read.csv(file_trainset, header = TRUE)
    clinical_vars <- get.clinical_features(colnames(df_trainset), event_col, duration_col)
    log_info(paste("Trainset file:", file_trainset, "with", nrow(df_trainset), "samples"))

    # Model 32X radiomics covariates
    log_info("Model 32X")
    cols_32X <- grep("^X32[0-9]{1}_", colnames(df_trainset), value = TRUE)
    covariates_32X <- c(clinical_vars, cols_32X, "has_radiomics")
    model_rsf(df_trainset, df_testset, covariates_32X, event_col, duration_col, rsf_logfile)

    # Model 1320 radiomics covariates
    log_info("Model 1320")
    cols_1320 <- grep("^X1320_", colnames(df_trainset), value = TRUE)
    covariates_1320 <- c(clinical_vars, cols_1320, "has_radiomics")
}

# Script args
args = commandArgs(trailingOnly = TRUE)
file_trainset = args[1]
file_testset = args[2]
event_col <- args[3]
analyzes_dir <- args[4]
rsf_name_logfile <- args[5]
if (length(args) == 6) {
    duration_col <- args[6]
} else {
    duration_col <- "survival_time_years"
}

log_threshold(INFO)
rsf_learning(file_trainset, file_testset, event_col, analyzes_dir, duration_col, rsf_name_logfile)

