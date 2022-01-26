
library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("randomForestSRC", quietly = TRUE)
library("pec", quietly = TRUE)
library("logger", quietly = TRUE)

source("workflow/scripts/utils_rsf.R")

rsf_learning <- function(file_trainset, file_testset, event_col, analyzes_dir, duration_col) {
    log_appender(appender_file(paste(analyzes_dir, "rsf.log", sep = ""), append = FALSE))
    log_info("Random Survival Forest learning")
    log_info(paste("Trainset file:", file_trainset))
    # Dataset
    df_trainset <- read.csv(file_trainset, header = TRUE)
    #cols_1320 < grep("X1320_", colnames(df_trainset), value = TRUE)
    cols_32X <- grep("X32[0-9]{1}_", colnames(df_trainset), value = TRUE)

    # Model 32X radiomics covariates
    df_heart_32X <- na.omit(df_trainset[,c(event_col, duration_col, cols_32X)])
    formula_32X <- get.surv.formula(event_col, cols_32X, duration_col = duration_col)
    test.params.df <- data.frame(ntree = c(1000), nodesize = c(10), nsplit = c(10))
    ntrees <- c(10, 100, 200, 500, 1000, 1500, 3000)
    nodesizes <- c(15, 50, 100)
    nsplits <- c(10, 700)
    params.df <- create.params.df(ntrees, nodesizes, nsplits)
    cv.params <- cv.rsf(formula_32X, df_heart_32X, test.params.df, event_col, error.metric = "ibs")
    # Best RSF
    params.best <- cv.params[1,]
    print(params.best)
    rsf.obj <- rfsrc(formula_32X, data = df_heart_32X, ntree = params.best$ntree, nodesize = params.best$nodesize, nsplit = params.best$nsplit)
    # C-index
    rsf.pred <- rsf.obj$predicted
    rsf.pred_oob <- rsf.obj$predicted.oob
    yvar <- rsf.obj$yvar
    rsf.err_oob <- get.cindex(yvar[[duration_col]], yvar[[event_col]], rsf.pred_oob)
    rsf.err <- get.cindex(yvar[[duration_col]], yvar[[event_col]], rsf.pred)
    log_info(paste("C-index on trainset: ", rsf.err_oob))
    log_info(paste("C-index OOB on trainset: ", rsf.err_oob))
    # IBS
    #bs.times <- c(0.0, sort(unique(fccss_vols$Attained_Age)))
    bs.times <- seq(5, 50, by = 5)
    rsf.pred.bs <- predictSurvProb(rsf.obj, newdata = df_heart_32X, times = bs.times)
    rsf.pred.oob.bs <- predictSurvProbOOB(rsf.obj, times = bs.times)
    rsf.perror <- pec(object= list(rsf.pred.bs, rsf.pred.oob.bs), formula = formula_32X, data = formula_32X)
    log_info(paste("IBS on trainset: ", crps(rsf.perror)[1]))
    log_info(paste("IBS OOB on trainset: ", crps(rsf.perror)[2]))

    # Model 1320 radiomics covariates
    #df_global_heart <- na.omit(df_trainset[,c(event_col, duration_col, cols_1320)])
}

# Script args
args = commandArgs(trailingOnly = TRUE)
file_trainset = args[1]
file_testset = args[2]
event_col <- args[3]
analyzes_dir <- args[4]
if (length(args) == 5) {
    duration_col <- args[5]
} else {
    duration_col <- "survival_time_years"
}

log_threshold(INFO)
rsf_learning(file_trainset, file_testset, event_col, analyzes_dir, duration_col)

