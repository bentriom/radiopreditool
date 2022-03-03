
library("stringr", quietly = TRUE)
library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("randomForestSRC", quietly = TRUE)
library("pec", quietly = TRUE)
library("logger", quietly = TRUE)

# Get clinical variables from all features
get.clinical_features <- function(columns, event_col, duration_col) {
    regex_non_clinical <- paste("^((X[0-9]{3,4}_)|(dv_)|(",event_col,")|(",duration_col,")|(ctr)|(numcent)|(has_radiomics))", sep = "")
    idx_non_clinical_vars <- grep(regex_non_clinical, columns)
    return (columns[-idx_non_clinical_vars])
}

# Pretty label names
pretty.label <- function(label) {
    pattern = "X([0-9]{3,4})_[a-z]+_[a-z]+_(\\w+)"
    if (str_detect(label, pattern)) {
        matches = str_match(label, pattern)
        paste(matches[2], matches[3])
    } else { label }
}

pretty.labels <- function(labels) {
    sapply(labels, pretty.label)
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

# Job for one parameter
get.param.cv.error <- function(idx.row, formula, data, event_col, duration_col, folds, params.df, bootstrap.strategy, error.metric, pred.times, rsf_logfile) {
    log_appender(appender_file(rsf_logfile, append = TRUE))
    log_info(paste("Error CV:", error.metric))
    elapsed.time <- system.time({
    row <- params.df[idx.row,]
    nbr.params <- nrow(params.df)
    log_info(paste(Sys.getpid(), "- Parameters", idx.row, "/", nbr.params, ": Begin"))
    nfolds <- length(unique(folds))
    final.time.bs <- pred.times[length(pred.times)]
    cindex.folds <- rep(0.0, nfolds)
    cindex.ipcw.folds <- rep(0.0, nfolds)
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
        cindex.fold <- 1 - get.cindex(fold.test[[duration_col]], fold.test[[event_col]], fold.test.pred$predicted)
        cindex.folds[i] <- cindex.fold
        # IPCW C-index
        cindex.ipcw.fold <- pec::cindex(list("RSF test fold" = rsf.fold), formula, data = fold.test)$AppCindex[["RSF test fold"]]
        cindex.ipcw.folds[i] <- cindex.ipcw.fold
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
        param.error <- mean(1-cindex.folds)
    } else if (error.metric == "cindex.ipcw") {
        param.error <- mean(1-cindex.ipcw.folds)
    } else {
        stop("Error metric not implemented")
    }
    })[["elapsed"]]
    log_info(paste(Sys.getpid(), "-", elapsed.time, "s -  Parameters ", idx.row, "/", nbr.params, " (", row$ntree, ",", row$nodesize, ",", row$nsplit, ") : ", param.error))
    c(unlist(row), mean(ibs.folds), mean(cindex.folds), mean(cindex.ipcw.folds), param.error)
}

# Cross-validation for RSF
cv.rsf <- function(formula, data, params.df, event_col, rsf_logfile, duration_col = "survival_time_years", 
                   nfolds = 5, pred.times = seq(5, 50, 5), error.metric = "ibs", bootstrap.strategy = NULL) {
    nbr.params <- nrow(params.df)
    folds <- createFolds(factor(data[[event_col]]), k = nfolds, list = FALSE)
    log_info(paste("Running CV with rfsrc using", getOption("rf.cores"), "workers")) 
    cv.params.df <- mclapply(1:nbr.params, function (i) { get.param.cv.error(i, formula, data, event_col, duration_col, folds, params.df, bootstrap.strategy, error.metric, pred.times, rsf_logfile) }, mc.cores = 1)
    cv.params.df <- as.data.frame(t(as.data.frame(cv.params.df)))
    # doMC
    # registerDoMC(nworkers)
    #  log_info(paste("Running doMC CV with", getDoParWorkers(), "workers"))
    #  cv.params.df <- foreach (idx.row = 1:nbr.params, .combine = rbind) %dopar% {
    #     get.param.cv.error(idx.row, formula, data, event_col, duration_col, folds, params.df, bootstrap.strategy, error.metric, pred.times, rsf_logfile)
    #  }
    #  cv.params.df <- as.data.frame(cv.params.df)
    # doParallel
    # cluster <- parallel::makeCluster(nworkers)
    # doParallel::registerDoParallel(cluster)
    # log_info(paste("Running doParallel CV with", getDoParWorkers(), "workers"))
    # cv.params.df <- foreach (idx.row = 1:nbr.params, .combine = rbind, .export = "get.param.cv.error", .packages = c("randomForestSRC", "pec", "logger", "survival")) %dopar% {
    #     get.param.cv.error(idx.row, formula, data, event_col, duration_col, folds, params.df, bootstrap.strategy, error.metric, pred.times, rsf_logfile)
    # }
    # cv.params.df <- as.data.frame(cv.params.df)
    
    rownames(cv.params.df) <- NULL
    colnames(cv.params.df) <- c(colnames(params.df), "IBS", "Harrel Cindex", "IPCW Cindex", "Error")
    cv.params.df[order(cv.params.df$Error),]
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

