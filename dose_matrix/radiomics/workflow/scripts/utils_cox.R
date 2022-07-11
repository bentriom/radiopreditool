
suppressPackageStartupMessages({
library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("glmnet", quietly = TRUE)
library("pec", quietly = TRUE)
library("Hmisc", quietly = TRUE)
library("logger", quietly = TRUE)
library("parallel", quietly = TRUE)
library("doParallel", quietly = TRUE)
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

preprocess_data_cox <- function(df_dataset, covariates, event_col, duration_col) {
    stopifnot({
        !(event_col %in% covariates)
        !(duration_col %in% covariates)
    })
    ## Preprocessing sets
    filter_train <- !duplicated(as.list(df_dataset[covariates])) & 
                    unlist(lapply(df_dataset[covariates], 
                                  function(col) { length(unique(col)) > 1 } ))
    filtered_covariates <- names(filter_train)[filter_train]
    df_model <- df_dataset[,c(event_col, duration_col, filtered_covariates)]
    df_model <- na.omit(df_model)
    # Z normalisation
    # means_train <- as.numeric(lapply(df_model[filtered_covariates], mean))
    # stds_train <- as.numeric(lapply(df_model[filtered_covariates], sd))
    # df_model[, filtered_covariates] <- scale(df_model[filtered_covariates], center = means_train, scale = stds_train)
    df_model <- normalize_data(df_model, filtered_covariates, event_col, duration_col)$train
    formula_model <- get.surv.formula(event_col, filtered_covariates, duration_col = duration_col)
    return (list("data" = df_model, "covariates" = filtered_covariates, "formula_model" = formula_model)) 
}

normalize_data <- function(df_train, covariates, event_col, duration_col, df_test = NULL) {
    continuous_vars <- covariates
    discrete_vars <- NULL
    regex_non_continuous <- "^((Sexe)|(iccc)|(has_radiomics)|(categ_age_at_diagnosis)|(chimiotherapie)|(ALKYL)|(ANTHRA)|(radiotherapie_1K))"
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
        df_train[, discrete_vars] <- scale(df_train[,discrete_vars], center = means_unique_train, scale = scales_unique_train)
    }
    if (!is.null(df_test)) {
        df_test[, continuous_vars] <- scale(df_test[continuous_vars], center = means_train, scale = stds_train)
        if (!is.null(discrete_vars))
            df_test[, discrete_vars] <- scale(df_test[,discrete_vars], center = means_unique_train, scale = scales_unique_train)
    }
    list("train" = df_train, "test" = df_test) 
}

coxlasso_data <- function(df, covariates, event_col, duration_col) {
    X <- as.matrix(df[covariates])
    surv_y <- Surv(df[[duration_col]], df[[event_col]])
    list("X" = X, "surv_y" = surv_y)
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
    run_parallel <- load_results & !save_results
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
    norm_data <- normalize_data(df_model_train, filtered_covariates, event_col, duration_col, df_test = df_model_test)
    df_model_train <- norm_data$train; df_model_test <- norm_data$test
    formula_model <- get.surv.formula(event_col, filtered_covariates, duration_col = duration_col)
    log_info(paste("Model name:", model_name))
        log_info(paste0("Covariates (", length(filtered_covariates),"):"))
    if (!run_parallel) {
        log_info(paste0(filtered_covariates, collapse = ", "))
        log_info("NAs are omitted")
        log_info(paste("Trained:", nrow(df_model_train), "samples"))
        log_info(paste("Testset: ", nrow(df_model_test), " samples"))
    }
    pred.times <- seq(1, 60, by = 1)
    final.time <- tail(pred.times, 1)
    ## Model and predictions
    coxlasso_data_train <- coxlasso_data(df_model_train, filtered_covariates, event_col, duration_col)
    X_train <- coxlasso_data_train$X; surv_y_train <- coxlasso_data_train$surv_y
    coxlasso_data_test <- coxlasso_data(df_model_test, filtered_covariates, event_col, duration_col)
    X_test <- coxlasso_data_test$X; surv_y_test <- coxlasso_data_test$surv_y   
    if (penalty == "none") {
        coxmodel <- coxph(formula_model, data = df_model_train, x = TRUE, y = TRUE, control = coxph.control(iter.max = 500))
        coxlasso.survprob.train <- predictSurvProb(coxmodel, newdata = data.table::as.data.table(df_model_train), times = pred.times)
        coxlasso.survprob.test <- predictSurvProb(coxmodel, newdata = data.table::as.data.table(df_model_test), times = pred.times)
        # coxlasso.predict.train <- predict(coxmodel, type = "survival")
        # coxlasso.predict.train <- matrix(coxlasso.predict.train, length(coxlasso.predict.train), 1)
        # coxlasso.predict.test <- predict(coxmodel, newdata = data.table::as.data.table(df_model_test), type = "survival")
        # coxlasso.predict.test <- matrix(coxlasso.predict.test, length(coxlasso.predict.test), 1)
    } else if (penalty == "lasso") {
        if (load_results) {
            best.lambda <- read.csv(paste0(analyzes_dir, "coxph_R_results/best_params_", model_name, ".csv"))[1, "penalty"]
            list.lambda <- read.csv(paste0(analyzes_dir, "coxph_R_results/path_lambda_", model_name, ".csv"))[["lambda"]]
            coxmodel <- glmnet(X_train, surv_y_train, family = "cox", alpha = 1, lambda = list.lambda, type.measure = "C")
        } else {
            coxmodel <- cv.glmnet(X_train, surv_y_train, family = "cox", alpha = 1, 
                                  nfolds = 5, parallel = T, type.measure = "C")
            cv.params <- data.frame(non_zero_coefs = as.numeric(coxmodel$nzero), penalty = coxmodel$lambda, mean_score = coxmodel$cvm, std_score = as.numeric(coxmodel$cvsd))
            cv.params <- cv.params[order(cv.params$mean_score, decreasing = TRUE), ] 
            if (save_results) write.csv(cv.params, file = paste0(analyzes_dir, "coxph_R_results/cv_", model_name, ".csv"), row.names = FALSE)
            best.lambda = select_best_lambda(coxmodel, cv.params)
            if (save_results) {
                write.csv(data.frame(penalty = best.lambda, l1_ratio = 1.0), row.names = F, 
                          file = paste0(analyzes_dir, "coxph_R_results/best_params_", model_name, ".csv"))
                write.csv(data.frame(lambda = coxmodel$lambda[coxmodel$lambda >= best.lambda]), row.names = F, 
                          file = paste0(analyzes_dir, "coxph_R_results/path_lambda_", model_name, ".csv"))
            }
        }
        log_info(paste("Best lambda:", best.lambda))
        coxlasso.survfit.train <- survfit(coxmodel, x = X_train, y = surv_y_train, newx = X_train, s = best.lambda)
        coxlasso.survfit.test <- survfit(coxmodel, x = X_train, y = surv_y_train, newx = X_test, s = best.lambda)
        coxlasso.survprob.train <- t(summary(coxlasso.survfit.train, times = pred.times)$surv)
        coxlasso.survprob.test <- t(summary(coxlasso.survfit.test, times = pred.times)$surv)
        # coxlasso.predict.train <- predict(coxmodel, newx = X_train, s = best.lambda)
        # coxlasso.predict.test <- predict(coxmodel, newx = X_test, s = best.lambda)
    }
    formula_ipcw <- get.ipcw.surv.formula(event_col, filtered_covariates)
    # C-index ipcw (censored free, marginal = KM)
    coxlasso.cindex.ipcw.train <- pec::cindex(list("Best coxlasso" = as.matrix(coxlasso.survprob.train[,1])), formula = formula_ipcw, 
                                                data = df_model_train, cens.model = "cox")$AppCindex[["Best coxlasso"]]
    coxlasso.cindex.ipcw.test <- pec::cindex(list("Best coxlasso" = as.matrix(coxlasso.survprob.test[,1])), formula = formula_ipcw, 
                                               data = df_model_test, cens.model = "cox")$AppCindex[["Best coxlasso"]]
    # Harrell's C-index
    coxlasso.cindex.harrell.train <- rcorr.cens(coxlasso.survprob.train[,1], S = surv_y_train)[["C Index"]]
    coxlasso.cindex.harrell.test <- rcorr.cens(coxlasso.survprob.test[,1], S = surv_y_test)[["C Index"]]
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
    log_info(paste(as.character(formula_ipcw)[c(2,1,3)], collapse = " "))
    log_info(paste0("Harrell's C-index on trainset: ", coxlasso.cindex.harrell.train))
    log_info(paste0("Harrell's C-index on testset: ", coxlasso.cindex.harrell.test))
    log_info(paste0("IPCW C-index on trainset: ", coxlasso.cindex.ipcw.train))
    log_info(paste0("IPCW C-index on testset: ", coxlasso.cindex.ipcw.test))
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
    if (save_results) {
        write.csv(df_results, file = paste0(analyzes_dir, "coxph_R_results/metrics_", model_name, ".csv"), row.names = TRUE)
        saveRDS(coxmodel, file = paste0(analyzes_dir, "coxph_R_results/fitted_models/", model_name, ".rds"))
    }
    if (do_plot) 
        plot_cox(coxmodel, analyzes_dir, model_name)
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

