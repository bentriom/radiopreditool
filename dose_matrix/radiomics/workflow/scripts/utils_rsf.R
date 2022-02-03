
library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("randomForestSRC", quietly = TRUE)
library("pec", quietly = TRUE)
library("foreach", quietly = TRUE)
library("doParallel", quietly = TRUE)

# Get clinical variables from all features
get.clinical_features <- function(columns, event_col, duration_col) {
    regex_non_clinical <- paste("^((X[0-9]{3,4}_)|(",event_col,")|(",duration_col,")|(ctr)|(numcent)|(has_radiomics))", sep = "")
    idx_non_clinical_vars <- grep(regex_non_clinical, columns)
    return (columns[-idx_non_clinical_vars])
}

# Automatically create a survival formula
get.surv.formula <- function(event_col, covariates, duration_col = "survival_time_years") {
    str.surv_formula <- paste("Surv(", duration_col, ",", event_col, ") ~ ", sep = '')
    for (var in covariates) {
        str.surv_formula <- paste(str.surv_formula, var, " + ", sep = '')
    }
    str.surv_formula <- substr(str.surv_formula, 1, nchar(str.surv_formula) - 2)
    as.formula(str.surv_formula)
}

# Brier score computation
predictSurvProbOOB <- function(object, times, ...){
    ptemp <- predict(object, importance="none",...)$survival.oob
    pos <- prodlim::sindex(jump.times=object$time.interest,eval.times=times)
    p <- cbind(1,ptemp)[,pos+1,drop=FALSE]
    if (NROW(p) != dim(ptemp)[1] || NCOL(p) != length(times))
        stop(paste("\nPrediction matrix has wrong dimensions:\nRequested newdata x times: ",NROW(newdata)," x ",length(times),"\nProvided prediction matrix: ",NROW(p)," x ",NCOL(p),"\n\n",sep=""))
    p
}

# Get the proportion of events in data
event_prop<-function(fccss.data, event_col) {
    return(sum(fccss.data[[event_col]] == 1) / nrow(fccss.data))
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

# Job for a parameter


# Cross-validation for RSF
cv.rsf <- function(formula, data, params.df, event_col, rsf_logfile, duration_col = "survival_time_years", 
                   nfolds = 3, pred.times = seq(5, 50, 5), error.metric = "ibs", bootstrap.strategy = NULL) {
    nbr.params <- nrow(params.df)
    final.time.bs <- pred.times[length(pred.times)]
    folds <- createFolds(factor(data[[event_col]]), k = nfolds, list = FALSE)
    ntasks = as.numeric(Sys.getenv("SLURM_NTASKS"))
    if (is.na(ntasks)) {
        nworkers <- parallel::detectCores() - 1
    }
    else {
        nworkers <- ntasks - 1
    }
    cluster <- parallel::makeCluster(nworkers, type = "FORK", outfile = rsf_logfile)
    doParallel::registerDoParallel(cl = cluster)
    log_info(paste("Running CV with", nworkers, "workers")) 
    cv.params.df <- foreach(idx.row = 1:nbr.params, .combine = 'rbind', .packages = c("survival", "randomForestSRC", "pec", "logger")) %dopar% {
        log_appender(appender_file(rsf_logfile, append = TRUE))
        row <- params.df[idx.row,]
        cindex.folds <- rep(0.0, nfolds)
        ibs.folds <- rep(0.0, nfolds)
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
            rsf.fold <- rfsrc(formula, data = fold.train, ntree = row$ntree, nodesize = row$nodesize, nsplit = row$nsplit, bootstrap = rsf.fold.bootstrap, samp = rsf.fold.samp)
            # C-index
            fold.test.pred <- predict(rsf.fold, newdata = fold.test)
            cindex.fold <- get.cindex(fold.test[[duration_col]], fold.test[[event_col]], fold.test.pred$predicted)
            cindex.folds[i] <- cindex.fold
            # IBS
            fold.test.pred.bs <- predictSurvProb(rsf.fold, newdata = fold.test, times = pred.times)
            perror = pec(object = fold.test.pred.bs, data = fold.test, formula = formula, 
                         times = pred.times, start = pred.times[0], exact = FALSE, reference = FALSE)
            ibs.fold <- crps(perror, times = final.time.bs)[1]
            ibs.folds[i] <- ibs.fold
        }
        if (error.metric == "ibs") {
            param.error <- mean(ibs.folds)
        } else if (error.metric == "cindex") {
            param.error <- mean(cindex.folds)
        } else {
            stop("Error metric not implemented")
        }
        log_info(paste(Sys.getpid(), "- Parameters ", idx.row, "/", nbr.params, " (", row$ntree, ",", row$nodesize, ",", row$nsplit, ") : ", param.error))
        c(row, mean(ibs.folds), mean(cindex.folds), param.error)
    }
    colnames(cv.params.df) <- c(colnames(params.df), "IBS", "Cindex", "Error")
    cv.params.df[order(cv.params.df$error),]
}

# Under-sampling
bootstrap.undersampling<-function(data,ntree) {
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

