
library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("glmnet", quietly = TRUE)
library("pec", quietly = TRUE)
library("Hmisc", quietly = TRUE)
library("logger", quietly = TRUE)
library("parallel", quietly = TRUE)
library("ggplot2", quietly = TRUE)
library("reshape2", quietly = TRUE)

source("workflow/scripts/utils_radiopreditool.R")

model_cox <- function(df_trainset, df_testset, covariates, event_col, duration_col, analyzes_dir, model_name, coxlasso_logfile, penalty = "lasso", do_plot = TRUE) {
    log_appender(appender_file(coxlasso_logfile, append = TRUE))
    df_model_train <- df_trainset[,c(event_col, duration_col, covariates)]
    df_model_test <- df_testset[,c(event_col, duration_col, covariates)]
    df_model_train <- na.omit(df_model_train)
    df_model_test <- na.omit(df_model_test)
    # Z normalisation
    means_train <- as.numeric(lapply(df_model_train[covariates], mean))
    stds_train <- as.numeric(lapply(df_model_train[covariates], sd))
    df_model_train[, covariates] <- scale(df_model_train[covariates], center = means_train, scale = stds_train)
    df_model_test[, covariates] <- scale(df_model_test[covariates], center = means_train, scale = stds_train)    
    X_train <- as.matrix(df_model_train[covariates])
    surv_y_train <- Surv(df_model_train[[duration_col]], df_model_train[[event_col]])
    X_test <- as.matrix(df_model_test[covariates])
    surv_y_test <- Surv(df_model_test[[duration_col]], df_model_test[[event_col]])
    formula_model <- get.surv.formula(event_col, covariates, duration_col = duration_col)
    log_info(paste0("Covariates (", length(covariates),"):", paste0(covariates, collapse = ", ")))
    log_info(paste0("Trained:", nrow(df_model_train), "samples"))
    log_info(paste0("Testset: ", nrow(df_model_test), " samples"))
    log_info("NAs are omitted")
    pred.times <- seq(1, 60, by = 1)
    final.time <- tail(pred.times, 1)
    # Model and predictions
    if (penalty == "none") {
        cv.coxlasso <- coxph(formula_model, data = df_model_train, x = TRUE, y = TRUE)
        coxlasso.survprob.train <- predictSurvProb(cv.coxlasso, newdata = data.table::as.data.table(df_model_train), times = pred.times)
        coxlasso.survprob.test <- predictSurvProb(cv.coxlasso, newdata = data.table::as.data.table(df_model_test), times = pred.times)
        coxlasso.predict.train <- predict(cv.coxlasso, type = "risk")
        coxlasso.predict.test <- predict(cv.coxlasso, newdata = data.table::as.data.table(df_model_test), type = "risk")
    } else if (penalty == "lasso") {
        cv.coxlasso <- cv.glmnet(X_train, surv_y_train, family = "cox", alpha = 1, nfolds = 5, type.measure = "C")
        cv.params <- data.frame(non_zero_coefs = as.numeric(cv.coxlasso$nzero), penalty = cv.coxlasso$lambda, mean_score = cv.coxlasso$cvm, std_score = as.numeric(cv.coxlasso$cvsd))
        cv.params <- cv.params[order(cv.params$mean_score, decreasing = TRUE), ] 
        write.csv(cv.params, file = paste0(analyzes_dir, "coxph_R_results/cv_", model_name, ".csv"), row.names = FALSE)
        log_info("Best lambda:")
        log_info(cv.coxlasso$lambda.min)
        coxlasso.survfit.train <- survfit(cv.coxlasso, x = X_train, y = surv_y_train, newx = X_train, s = "lambda.min")
        coxlasso.survfit.test <- survfit(cv.coxlasso, x = X_train, y = surv_y_train, newx = X_test, s = "lambda.min")
        coxlasso.survprob.train <- t(summary(coxlasso.survfit.train, times = pred.times)$surv)
        coxlasso.survprob.test <- t(summary(coxlasso.survfit.test, times = pred.times)$surv)
        coxlasso.predict.train <- predict(cv.coxlasso, newx = X_train, s = "lambda.min")
        coxlasso.predict.test <- predict(cv.coxlasso, newx = X_test, s = "lambda.min")
    }
    # C-index ipcw (censored free, marginal = KM)
    coxlasso.cindex.ipcw.train <- pec::cindex(list("Best coxlasso" = coxlasso.predict.train), formula_model, data = df_model_train)$AppCindex[["Best coxlasso"]]
    coxlasso.cindex.ipcw.test <- pec::cindex(list("Best coxlasso" = coxlasso.predict.test), formula_model, data = df_model_test)$AppCindex[["Best coxlasso"]]
    # Harrell's C-index
    coxlasso.cindex.harrell.train <- 1-rcorr.cens(coxlasso.predict.train, S = surv_y_train)[["C Index"]]
    coxlasso.cindex.harrell.test <- 1-rcorr.cens(coxlasso.predict.test, S = surv_y_test)[["C Index"]]
    log_info(paste0("Harrell's C-index on trainset: ", coxlasso.cindex.harrell.train))
    log_info(paste0("Harrell's C-index on testset: ", coxlasso.cindex.harrell.test))
    log_info(paste0("IPCW C-index on trainset: ", coxlasso.cindex.ipcw.train))
    log_info(paste0("IPCW C-index on testset: ", coxlasso.cindex.ipcw.test))
    # IBS
    coxlasso.perror.train <- pec(object= list("train" = coxlasso.survprob.train), formula = formula_model, data = df_model_train, 
                            times = pred.times, start = pred.times[0], exact = FALSE, reference = FALSE)
    coxlasso.perror.test <- pec(object= list("test" = coxlasso.survprob.test), formula = formula_model, data = df_model_test, 
                           times = pred.times, start = pred.times[0], exact = FALSE, reference = FALSE)
    coxlasso.bs.final.train <- tail(coxlasso.perror.train$AppErr$train, 1)
    coxlasso.bs.final.test <- tail(coxlasso.perror.test$AppErr$test, 1)
    coxlasso.ibs.train <- crps(coxlasso.perror.train)[1]
    coxlasso.ibs.test <- crps(coxlasso.perror.test)[1]
    log_info(paste0("BS at 60 on trainset: ", coxlasso.bs.final.train))
    log_info(paste0("BS at 60 on testset: ", coxlasso.bs.final.test))
    log_info(paste0("IBS on trainset: ", coxlasso.ibs.train))
    log_info(paste0("IBS on testset: ", coxlasso.ibs.test))
    results_train <- c(round(coxlasso.cindex.harrell.train, digits = 3), round(coxlasso.cindex.ipcw.train, digits = 3), 
                      round(coxlasso.bs.final.train, digits = 3), round(coxlasso.ibs.train, digits = 3))
    results_test <- c(round(coxlasso.cindex.harrell.test, digits = 3), round(coxlasso.cindex.ipcw.test, digits = 3), 
                      round(coxlasso.bs.final.test, digits = 3), round(coxlasso.ibs.test, digits = 3))
    log_info(paste0("Train:", results_train[1], "&", results_train[2], "&", results_train[3], "&", results_train[4]))
    log_info(paste0("Test:", results_test[1], "&", results_test[2], "&", results_test[3], "&", results_test[4]))
    df_results <- data.frame(Train = results_train, Test = results_test)
    rownames(df_results) <- c("C-index", "IPCW C-index", "BS at 60", "IBS")
    write.csv(df_results, file = paste0(analyzes_dir, "coxph_R_results/metrics_", model_name, ".csv"), row.names = TRUE)
    best.lambda = cv.coxlasso$lambda.min
    write.csv(data.frame(penalty = best.lambda, l1_ratio = 1.0), file = paste0(analyzes_dir, "coxph_R_results/best_params_", model_name, ".csv"), row.names = FALSE)
    if (do_plot) plot_cox(cv.coxlasso, analyzes_dir, model_name)
    cv.coxlasso
}

plot_cox <- function(cox_object, analyzes_dir, model_name) {
    # Coefficients of best Cox Lasso
    best.coefs.cox <-  `if`(class(cox_object) == "cv.glmnet", coef(cox_object, s = "lambda.min")[,1], coef(cox_object))
    nonnull.best.coefs <- best.coefs.cox[abs(best.coefs.cox) > 0]
    df.coefs = data.frame(labels = names(best.coefs.cox), coefs = best.coefs.cox)
    ggplot(subset(df.coefs, labels %in% names(nonnull.best.coefs)), aes(x = labels, y = coefs)) + geom_bar(stat = "identity") + coord_flip() 
    ggsave(paste0(analyzes_dir, "coxph_R_plots/coefs_", model_name, ".png"), device = "png", dpi = 480)
    # Regularization path + mean error for Cox Lasso
    if (class(cox_object) == "cv.glmnet") {
        # Regularization path
        mat.coefs <- t(as.matrix(coef(cox_object$glmnet.fit)))
        rownames(mat.coefs) <- cox_object$glmnet.fit$lambda
        df.mat.coefs <- melt(mat.coefs)
        colnames(df.mat.coefs) <- c("lambda", "varname", "coef")
        first.lambda <- rownames(mat.coefs)[nrow(mat.coefs)]
        ggplot(df.mat.coefs, aes(x = lambda, y = coef, color = varname)) + geom_line() + 
        xlab("Penalty (log10)") + ylab("Coefficient") + theme(legend.position = "none") +
        geom_text(data = subset(df.mat.coefs, lambda == first.lambda & varname %in% names(nonnull.best.coefs)), 
                  aes(x = lambda, y = coef, label = pretty.labels(varname)), size = 3, hjust = 1) +
        coord_cartesian(clip = 'off') +
        theme(plot.margin = unit(c(1,1,1,1), "lines"), axis.title.y = element_text(margin = margin(0,2,0,0, "cm"))) + 
        scale_x_log10()
        ggsave(paste0(analyzes_dir, "coxph_R_plots/regularization_path_", model_name, ".png"), device = "png", dpi = 480)
        # Mean errors of cross-validation
        best.params <- read.csv(paste0(analyzes_dir, "coxph_R_results/best_params_", model_name, ".csv"))
        cv.params <- read.csv(paste0(analyzes_dir, "coxph_R_results/cv_", model_name, ".csv"))
        cv.params.unique <- cv.params[!duplicated(cv.params$non_zero_coefs), ]
        cv.params.unique <- cv.params.unique[order(cv.params.unique$penalty), ]
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
    }
}

