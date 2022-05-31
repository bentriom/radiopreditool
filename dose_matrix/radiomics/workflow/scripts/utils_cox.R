
suppressPackageStartupMessages({
library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("glmnet", quietly = TRUE)
library("pec", quietly = TRUE)
library("Hmisc", quietly = TRUE)
library("logger", quietly = TRUE)
library("parallel", quietly = TRUE)
library("ggplot2", quietly = TRUE)
library("reshape2", quietly = TRUE)
library("randomForestSRC", quietly = TRUE)
})

source("workflow/scripts/utils_radiopreditool.R")

select_best_lambda <- function(cox_object, cv.params) {
    lambda.ref <- cox_object$lambda.min
    deviance.params <- data.frame(penalty = cox_object$lambda, deviance = deviance(cox_object$glmnet.fit))
    cv.params.merge <- merge(cv.params, deviance.params, by = "penalty")
    rownames(cv.params.merge) <- cv.params.merge$penalty
    deviance.ref <- cv.params.merge[as.character(lambda.ref), "deviance"]
    nonzeros.ref <- cv.params.merge[as.character(lambda.ref), "non_zero_coefs"]
    cv.params.unique <- cv.params.merge[order(-cv.params.merge$non_zero_coefs, cv.params.merge$penalty), ]
    cv.params.unique <- cv.params.unique[!duplicated(cv.params.unique$non_zero_coefs), ]
    cv.params.unique <- cv.params.unique[(cv.params.unique$penalty > lambda.ref) &
                                         (cv.params.unique$non_zeros_coefs < nonzeros.ref), ]
    log_info("Best lambda selection")
    log_info(paste("lambda ref:", lambda.ref, nonzeros.ref))
    write.csv(cv.params, file = "test_cv_params.csv")
    write.csv(cv.params.unique, file = "test_cv_params_unique.csv")
    lambda.new <- lambda.ref
    if (nrow(cv.params.unique) < 1) return (lambda.ref)
    for (i in 1:nrow(cv.params.unique)) {
        deviance.new <- cv.params.unique[i, "deviance"]
        nonzeros.new <- cv.params.unique[i, "non_zero_coefs"]
        loglik.ratio <- deviance.new - deviance.ref
        df <- nonzeros.ref - nonzeros.new
        pvalue <- 1 - pchisq(loglik.ratio, df)
        log_info(paste("- compared to", cv.params.unique[i, "penalty"], nonzeros.new))
        if (is.na(pvalue) | is.null(pvalue)) {
            log_warn(paste("- lambda ref, nzeros:", lambda.ref, nonzeros.ref))
            log_warn(paste("- lambda new to test, nzeros:", cv.params.unique[i, 'penalty'], nonzeros.new))
            log_warn(paste("- df:", df))
            log_warn(paste("- deviance ref:", deviance.ref))
            log_warn(paste("- deviance new:", deviance.new))
            log_warn(paste("- loglik:", loglik.ratio))
            log_warn(paste("- pvalue:", pvalue))
        }
        if (pvalue < 0.05) break
        lambda.new <- cv.params.unique[i, "penalty"]
    }
    lambda.new
}

model_cox.id <- function(id_set, covariates, event_col, duration_col, analyzes_dir, model_name, coxlasso_logfile, penalty = "lasso") {
    df_trainset <- read.csv(paste0(analyzes_dir, "datasets/trainset_", id_set, ".csv.gz"), header = TRUE)
    df_testset <- read.csv(paste0(analyzes_dir, "datasets/testset_", id_set, ".csv.gz"), header = TRUE)
    log_appender(appender_file(coxlasso_logfile, append = TRUE))
    log_info(id_set)
    model_cox(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, 
              model_name, coxlasso_logfile, penalty = penalty, 
              do_plot = FALSE, save_results = FALSE, load_results = TRUE, level = INFO)
}

model_cox <- function(df_trainset, df_testset, covariates, event_col, duration_col, 
                      analyzes_dir, model_name, coxlasso_logfile,
                      penalty = "lasso", do_plot = TRUE, 
                      save_results = TRUE, load_results = FALSE, level = INFO) {
    log_threshold(level)
    log_appender(appender_file(coxlasso_logfile, append = TRUE))
    ## Preprocessing sets
    filter_train <- !duplicated(as.list(df_trainset[covariates])) & 
                    unlist(lapply(df_trainset[covariates], 
                                  function(col) { length(unique(col)) > 1 } ))
    filtered_covariates <- names(filter_train)[filter_train]
    df_model_train <- df_trainset[,c(event_col, duration_col, filtered_covariates)]
    df_model_test <- df_testset[,c(event_col, duration_col, filtered_covariates)]
    df_model_train <- na.omit(df_model_train)
    df_model_test <- na.omit(df_model_test)
    # Z normalisation
    means_train <- as.numeric(lapply(df_model_train[filtered_covariates], mean))
    stds_train <- as.numeric(lapply(df_model_train[filtered_covariates], sd))
    df_model_train[, filtered_covariates] <- scale(df_model_train[filtered_covariates], center = means_train, scale = stds_train)
    df_model_test[, filtered_covariates] <- scale(df_model_test[filtered_covariates], center = means_train, scale = stds_train)    
    X_train <- as.matrix(df_model_train[filtered_covariates])
    surv_y_train <- Surv(df_model_train[[duration_col]], df_model_train[[event_col]])
    X_test <- as.matrix(df_model_test[filtered_covariates])
    surv_y_test <- Surv(df_model_test[[duration_col]], df_model_test[[event_col]])
    formula_model <- get.surv.formula(event_col, filtered_covariates, duration_col = duration_col)
    log_info(paste("Model name:", model_name))
    log_info(paste0("Covariates (", length(filtered_covariates),"):", paste0(filtered_covariates, collapse = ", ")))
    log_info(paste("Trained:", nrow(df_model_train), "samples"))
    log_info(paste("Testset: ", nrow(df_model_test), " samples"))
    log_info("NAs are omitted")
    pred.times <- seq(1, 60, by = 1)
    final.time <- tail(pred.times, 1)
    ## Model and predictions
    if (penalty == "none") {
        cv.coxlasso <- coxph(formula_model, data = df_model_train, x = TRUE, y = TRUE)
        coxlasso.survprob.train <- predictSurvProb(cv.coxlasso, newdata = data.table::as.data.table(df_model_train), times = pred.times)
        coxlasso.survprob.test <- predictSurvProb(cv.coxlasso, newdata = data.table::as.data.table(df_model_test), times = pred.times)
        coxlasso.predict.train <- predict(cv.coxlasso, type = "survival")
        coxlasso.predict.train <- matrix(coxlasso.predict.train, length(coxlasso.predict.train), 1)
        coxlasso.predict.test <- predict(cv.coxlasso, newdata = data.table::as.data.table(df_model_test), type = "survival")
        coxlasso.predict.test <- matrix(coxlasso.predict.test, length(coxlasso.predict.test), 1)
    } else if (penalty == "lasso") {
        if (load_results) {
            best.lambda <- read.csv(paste0(analyzes_dir, "coxph_R_results/best_params_", model_name, ".csv"))[1, "penalty"]
            cv.coxlasso <- glmnet(X_train, surv_y_train, family = "cox", alpha = 1, lambda = best.lambda, type.measure = "C")
        } else {
            cv.coxlasso <- cv.glmnet(X_train, surv_y_train, family = "cox", alpha = 1, nfolds = 5, type.measure = "C")
            cv.params <- data.frame(non_zero_coefs = as.numeric(cv.coxlasso$nzero), penalty = cv.coxlasso$lambda, mean_score = cv.coxlasso$cvm, std_score = as.numeric(cv.coxlasso$cvsd))
            cv.params <- cv.params[order(cv.params$mean_score, decreasing = TRUE), ] 
            if (save_results) write.csv(cv.params, file = paste0(analyzes_dir, "coxph_R_results/cv_", model_name, ".csv"), row.names = FALSE)
            best.lambda = select_best_lambda(cv.coxlasso, cv.params)
            if (save_results) write.csv(data.frame(penalty = best.lambda, l1_ratio = 1.0), file = paste0(analyzes_dir, "coxph_R_results/best_params_", model_name, ".csv"), row.names = FALSE)
        }
        log_info(paste("Best lambda:", best.lambda))
        coxlasso.survfit.train <- survfit(cv.coxlasso, x = X_train, y = surv_y_train, newx = X_train, s = best.lambda)
        coxlasso.survfit.test <- survfit(cv.coxlasso, x = X_train, y = surv_y_train, newx = X_test, s = best.lambda)
        coxlasso.survprob.train <- t(summary(coxlasso.survfit.train, times = pred.times)$surv)
        coxlasso.survprob.test <- t(summary(coxlasso.survfit.test, times = pred.times)$surv)
        coxlasso.predict.train <- predict(cv.coxlasso, newx = X_train, s = best.lambda)
        coxlasso.predict.test <- predict(cv.coxlasso, newx = X_test, s = best.lambda)
    }
    # C-index ipcw (censored free, marginal = KM)
    coxlasso.cindex.ipcw.train <- 1-pec::cindex(list("Best coxlasso" = coxlasso.predict.train), formula_model, data = df_model_train)$AppCindex[["Best coxlasso"]]
    coxlasso.cindex.ipcw.test <- 1-pec::cindex(list("Best coxlasso" = coxlasso.predict.test), formula_model, data = df_model_test)$AppCindex[["Best coxlasso"]]
    # Harrell's C-index
    coxlasso.cindex.harrell.train <- 1-rcorr.cens(coxlasso.predict.train, S = surv_y_train)[["C Index"]]
    coxlasso.cindex.harrell.test <- 1-rcorr.cens(coxlasso.predict.test, S = surv_y_test)[["C Index"]]
    log_info(paste0("Harrell's C-index on trainset: ", coxlasso.cindex.harrell.train))
    log_info(paste0("Harrell's C-index on testset: ", coxlasso.cindex.harrell.test))
    log_info(paste0("IPCW C-index on trainset: ", coxlasso.cindex.ipcw.train))
    log_info(paste0("IPCW C-index on testset: ", coxlasso.cindex.ipcw.test))
    # IBS
    coxlasso.perror.train <- pec(object= list("train" = coxlasso.survprob.train), 
                                 formula = formula_model, data = df_model_train, 
                                 cens.model = "marginal", 
                                 times = pred.times, start = pred.times[0], 
                                 exact = FALSE, reference = FALSE)
    coxlasso.perror.test <- pec(object= list("test" = coxlasso.survprob.test), 
                                formula = formula_model, data = df_model_test, 
                                cens.model = "marginal", 
                                times = pred.times, start = pred.times[0], 
                                exact = FALSE, reference = FALSE)
    coxlasso.bs.final.train <- tail(coxlasso.perror.train$AppErr$train, 1)
    coxlasso.bs.final.test <- tail(coxlasso.perror.test$AppErr$test, 1)
    coxlasso.ibs.train <- crps(coxlasso.perror.train)[1]
    coxlasso.ibs.test <- crps(coxlasso.perror.test)[1]
    log_info(paste0("BS at 60 on trainset: ", coxlasso.bs.final.train))
    log_info(paste0("BS at 60 on testset: ", coxlasso.bs.final.test))
    log_info(paste0("IBS on trainset: ", coxlasso.ibs.train))
    log_info(paste0("IBS on testset: ", coxlasso.ibs.test))
    results_train <- c(coxlasso.cindex.harrell.train, coxlasso.cindex.ipcw.train, 
                       coxlasso.bs.final.train, coxlasso.ibs.train)
    results_test <- c(coxlasso.cindex.harrell.test, coxlasso.cindex.ipcw.test,
                      coxlasso.bs.final.test, coxlasso.ibs.test)
    log_info(paste("Train:", results_train[1], "&", results_train[2], "&", results_train[3], "&", results_train[4]))
    log_info(paste("Test:", results_test[1], "&", results_test[2], "&", results_test[3], "&", results_test[4]))
    df_results <- data.frame(Train = results_train, Test = results_test)
    rownames(df_results) <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
    if (save_results) write.csv(df_results, file = paste0(analyzes_dir, "coxph_R_results/metrics_", model_name, ".csv"), row.names = TRUE)
    if (do_plot) plot_cox(cv.coxlasso, analyzes_dir, model_name)
    log_threshold(INFO)
    results_test
}

plot_cox <- function(cox_object, analyzes_dir, model_name) {
    # Coefficients of best Cox
    if (class(cox_object) == "cv.glmnet") {
        best.params <- read.csv(paste0(analyzes_dir, "coxph_R_results/best_params_", model_name, ".csv"))
        best.lambda <- best.params[1, "penalty"]
    }
    best.coefs.cox <- `if`(class(cox_object) == "cv.glmnet", coef(cox_object, s = best.lambda)[,1], coef(cox_object))
    names.nonnull.coefs <- pretty.labels(names(best.coefs.cox[abs(best.coefs.cox) > 0]))
    df.coefs = data.frame(labels = pretty.labels(names(best.coefs.cox)), coefs = best.coefs.cox)
    ggplot(subset(df.coefs, labels %in% names.nonnull.coefs), aes(x = labels, y = coefs)) + geom_bar(stat = "identity") + coord_flip() 
    ggsave(paste0(analyzes_dir, "coxph_R_plots/coefs_", model_name, ".png"), device = "png", dpi = 480)
    # Regularization path + mean error for Cox Lasso
    if (class(cox_object) == "cv.glmnet") {
        # Mean errors of cross-validation
        cv.params <- read.csv(paste0(analyzes_dir, "coxph_R_results/cv_", model_name, ".csv"))
        cv.params.unique <- cv.params[order(-cv.params$non_zero_coefs, cv.params$penalty), ]
        cv.params.unique <- cv.params.unique[!duplicated(cv.params.unique$non_zero_coefs), ]
        mask_even = which((1:nrow(cv.params.unique)) %% 2 == 0)
        ggplot(cv.params, aes(x = penalty, y = mean_score)) +
        geom_ribbon(aes(ymin = mean_score - std_score, ymax = mean_score + std_score), fill = "blue", alpha = 0.5) +
        geom_line(color = "blue") + geom_point(color = " red") +
        geom_vline(xintercept = cv.params[1, "penalty"], color = "orange") +
        geom_vline(xintercept = best.params[1, "penalty"], color = "darkgreen", linetype = "dotdash") +
        geom_point(data = cv.params.unique, aes(x = penalty, y = mean_score), color = "purple") +
        geom_text(data = cv.params.unique[mask_even,], aes(x = penalty, y = mean_score, label = non_zero_coefs), size = 3, vjust = -1) +
        geom_text(data = cv.params.unique[-mask_even,], aes(x = penalty, y = mean_score, label = non_zero_coefs), size = 3, vjust = 2) +
        scale_x_log10() +
        labs(x = "Penalty (log10)", y = "Mean score (1 - Cindex)")
        ggsave(paste0(analyzes_dir, "coxph_R_plots/cv_mean_error_", model_name, ".png"), device = "png", dpi = 480)
        # Regularization path
        mat.coefs <- t(as.matrix(coef(cox_object$glmnet.fit)))
        rownames(mat.coefs) <- cox_object$glmnet.fit$lambda
        df.mat.coefs <- melt(mat.coefs)
        colnames(df.mat.coefs) <- c("lambda", "varname", "coef")
        df.mat.coefs[, "varname"] <- pretty.labels(df.mat.coefs[["varname"]])
        first.lambda <- rownames(mat.coefs)[nrow(mat.coefs)]
        ggplot(df.mat.coefs, aes(x = lambda, y = coef, color = varname)) + geom_line() + 
        xlab("Penalty (log10)") + ylab("Coefficient") + theme(legend.position = "none") +
        geom_text(data = subset(df.mat.coefs, lambda == first.lambda & varname %in% names.nonnull.coefs), 
                  aes(x = lambda, y = coef, label = pretty.labels(varname)), size = 3, hjust = 1) +
        coord_cartesian(clip = 'off') +
        theme(plot.margin = unit(c(1,1,1,1), "lines"), axis.title.y = element_text(margin = margin(0,2,0,0, "cm"))) + 
        geom_vline(xintercept = cv.params[1, "penalty"], color = "orange") +
        geom_vline(xintercept = best.params[1, "penalty"], color = "darkgreen", linetype = "dotdash") +
        scale_x_log10()
        ggsave(paste0(analyzes_dir, "coxph_R_plots/regularization_path_", model_name, ".png"), device = "png", dpi = 480)
    }
}

