
library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("randomForestSRC", quietly = TRUE)
library("pec", quietly = TRUE)

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

# Cross-validation for RSF
cv.rsf <- function(formula, data, params.df, event_col, duration_col = "survival_time_years", nfolds = 3, error.metric = "ibs", bootstrap.strategy = NULL) {
  nbr.params <- nrow(params.df)
  params.error <- rep(0.0, nbr.params)
  params.ibs <- rep(0.0, nbr.params)
  params.cindex <- rep(0.0, nbr.params)
  folds <- createFolds(factor(data[[event_col]]), k = nfolds, list = FALSE)
  
  for (idx.row in 1:nrow(params.df)) {
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
      fold.test.pred <- predict(rsf.fold, fold.test)
      cindex.fold <- get.cindex(fold.test[[duration_col]], fold.test[[event_col]], fold.test.pred$predicted)
      cindex.folds[i] <- cindex.fold
      # IBS
      bs.times <- seq(5, 50, by = 5)
      fold.test.pred.bs <- predictSurvProb(rsf.fold, newdata = fold.test, times = bs.times)
      perror = pec(object = fold.test.pred.bs, data = fold.test, reference = FALSE, formula = formula)
      ibs.fold <- crps(perror)[1]
      ibs.folds[i] <- ibs.fold
    }
    if (error.metric == "ibs") {
      param.error <- mean(ibs.folds)
    } else if (error.metric == "cindex") {
      param.error <- mean(cindex.folds)
    } else {
      stop("Error metric not implemented")
    }
    params.error[idx.row] <- param.error
    params.ibs[idx.row] <- mean(ibs.folds)
    params.cindex[idx.row] <- mean(cindex.folds)
    log_info(paste("Parameters ", idx.row, "/", nbr.params, " (", row$ntree, ",", row$nodesize, ",", row$nsplit, ") : ", param.error))
  }
  params.df$error <- params.error
  params.df$cindex <- params.cindex
  params.df$ibs <- params.ibs
  params.df[order(params.error),]
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

