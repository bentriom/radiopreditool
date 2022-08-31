
options(show.error.locations = TRUE, error=traceback)

source("workflow/scripts/utils_cox.R")
source("workflow/scripts/utils_rsf.R")

glmnet.pec <- function(formula, data, family = "cox", alpha = 1, type.measure = "C", 
                       lambda = NULL, best.lambda = "lambda.min", ...) {
    covariates <- all.vars(formula[[3]])
    duration_col <- all.vars(formula[[2]])[1]
    event_col <- all.vars(formula[[2]])[2]
    if (!("data.table" %in% class(data))) {
        data_lasso <- coxlasso_data(data, covariates, event_col, duration_col)
        X <- data_lasso$X
        surv_y <- data_lasso$surv_y
    } else {
        X <- as.matrix(data[, ..covariates])
        surv_y <- Surv(as.numeric(unlist(data[, ..duration_col])), as.numeric(unlist(data[, ..event_col])))
    }
    fit <- glmnet(X, surv_y, family = "cox", alpha = 1, lambda = lambda, type.measure = "C", ...)
    out <- list("glmnet.fit" = fit, "call" = match.call(), "X" = X, "surv_y" = surv_y, 
                "best.lambda" = best.lambda, "covariates" = covariates)
    class(out) <- "glmnet.pec"
    out
}

predictSurvProb.glmnet.pec <- function(object, newdata, times) {
    covariates <- object$covariates
    newdata_model <- newdata[, ..covariates]
    # saveRDS(newdata, file = "newdata.rds")
    # saveRDS(object$covariates, file = "covariates.rds")
    # saveRDS(newdata_model, file = "newdata_model.rds")
    cvcoxlasso.survfit <- survfit(object$glmnet.fit, x = object$X, y = object$surv_y, newx = as.matrix(newdata_model), s = object$best.lambda)
    t(summary(cvcoxlasso.survfit, times = times)$surv)
}

pec_estimation <- function(file_dataset, event_col, analyzes_dir, duration_col, B = 200) {
    ntasks <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
    nworkers <- `if`(is.na(ntasks), parallel::detectCores()-1, ntasks)
    options(rf.cores = 1, mc.cores = 1)
    dir.create(paste0(analyzes_dir, "pec_plots/"), showWarnings = FALSE)
    dir.create(paste0(analyzes_dir, "pec_results/"), showWarnings = FALSE)
    df_dataset <- read.csv(file_dataset, header = T)

    # Feature elimination
    file_features_hclust_corr <- paste0(analyzes_dir, "features_hclust_corr.csv")
    features_hclust_corr <- as.character(read.csv(file_features_hclust_corr)[,1])
    features_hclust_corr <- as.character(lapply(features_hclust_corr, function(x) { `if`(str_detect(substr(x, 1, 1), "[0-9]"), paste0("X", x), x) }))
    # Covariables models
    cols_rsf_1320 <- filter.gl(grep("^X1320_original_firstorder_", colnames(df_dataset), value = TRUE))
    cols_lasso_32X <- filter.gl(grep("^X32[0-9]{1}_", colnames(df_dataset), value = TRUE))
    cols_lasso_32X <- cols_lasso_32X[cols_lasso_32X %in% features_hclust_corr] # specific to this model
    cols_dv <- grep("dv_\\w+_1320", colnames(df_dataset), value = TRUE)
    clinical_vars <- get.clinical_features(colnames(df_dataset), event_col, duration_col)
    covariates_all <- unique(c(clinical_vars, "X1320_original_firstorder_Mean", cols_dv, cols_rsf_1320, cols_lasso_32X))
    df_dataset <- df_dataset[, c(event_col, duration_col, covariates_all)]

    # Preprocessing data
    infos <- preprocess_data_cox(df_dataset, covariates_all, event_col, duration_col)
    # After the preprocessing, some variables may be deleted
    cols_rsf_1320 <- cols_rsf_1320[cols_rsf_1320 %in% colnames(infos$data)]
    cols_lasso_32X <- cols_lasso_32X[cols_lasso_32X %in% colnames(infos$data)]
    cols_dv <- cols_dv[cols_dv %in% colnames(infos$data)]
    clinical_vars <- clinical_vars[clinical_vars %in% colnames(infos$data)]

    # Model 1320 radiomics full covariates
    suffix_model <- "all"
    model_name_rsf <- paste0("1320_radiomics_full_", suffix_model)
    covariates_rsf_1320 <- c(clinical_vars, cols_rsf_1320)
    formula_model_rsf <- get.surv.formula(event_col, covariates_rsf_1320, duration_col = duration_col)
    rsf_1320_params.best <<- read.csv(paste(analyzes_dir, "rsf_results/cv_", model_name_rsf, ".csv", sep = ""))[1,]
    rsf.best <- rfsrc(formula_model_rsf, data = infos$data, ntree = rsf_1320_params.best$ntree, 
                      nodesize = rsf_1320_params.best$nodesize, nsplit = rsf_1320_params.best$nsplit)

    # Model mean cox
    model_name_coxmean <- "1320_mean"
    covariates_coxmean <- c("X1320_original_firstorder_Mean", clinical_vars)
    # infos_coxmean <- preprocess_data_cox(df_dataset, covariates_coxmean, event_col, duration_col)
    formula_model_coxmean <- get.surv.formula(event_col, covariates_coxmean, duration_col = duration_col)
    coxmean <- coxph(formula_model_coxmean, data = infos$data, x = TRUE, y = TRUE, control = coxph.control(iter.max = 500))
    # coxmean <- rms::cph(formula_model_coxmean, data = infos$data, x = TRUE, y = TRUE, surv = TRUE)
    coxmean$call$formula <- formula_model_coxmean

    # Model lasso dosesvolumes
    model_name_lassodv <- "1320_dosesvol_lasso"
    covariates_lassodv <- c(cols_dv, clinical_vars)
    formula_model_lassodv <<- get.surv.formula(event_col, covariates_lassodv, duration_col = duration_col)
    lassodv_best.lambda <<- read.csv(paste0(analyzes_dir, "coxph_R_results/best_params_", model_name_lassodv, ".csv"))[1, "penalty"]
    lassodv_list.lambda <<- read.csv(paste0(analyzes_dir, "coxph_R_results/path_lambda_", model_name_lassodv, ".csv"))[["lambda"]]
    # coxlassodv <- coxnet.pec(formula_model_lassodv, data = infos$data, family = "cox", alpha = 1, 
    #                          lambda = list.lambda, lambda.pred = best.lambda, type.measure = "C")
    # coxlassodv_data_train <- coxlasso_data(infos$data, covariates_lassodv, event_col, duration_col)
    # write.csv(coxlassodv_data_train$X, "test.csv")
    coxlassodv <- glmnet.pec(formula_model_lassodv, data = infos$data, family = "cox", 
                             alpha = 1, lambda = lassodv_list.lambda, best.lambda = lassodv_best.lambda, type.measure = "C")
    
    # Model 32X lasso features hclust corr
    model_name_lasso_32X <- "32X_radiomics_full_lasso_features_hclust_corr"
    covariates_lasso_32X <- c(cols_dv, clinical_vars)
    formula_model_lasso_32X <<- get.surv.formula(event_col, covariates_lasso_32X, duration_col = duration_col)
    lasso_32X_best.lambda <<- read.csv(paste0(analyzes_dir, "coxph_R_results/best_params_", model_name_lasso_32X, ".csv"))[1, "penalty"]
    lasso_32X_list.lambda <<- read.csv(paste0(analyzes_dir, "coxph_R_results/path_lambda_", model_name_lasso_32X, ".csv"))[["lambda"]]
    # coxlasso_32X <- coxnet.pec(formula_model_lasso_32X, data = infos$data, family = "cox", alpha = 1, 
    #                            lambda = list.lambda, lambda.pred = best.lambda, type.measure = "C")
    # coxlasso_32X_data_train <- coxlasso_data(infos$data, covariates_lasso_32X, event_col, duration_col)
    coxlasso_32X <- glmnet.pec(formula_model_lasso_32X, data = infos$data, family = "cox", 
                               alpha = 1, lambda = lasso_32X_list.lambda, best.lambda = lasso_32X_best.lambda, type.measure = "C")

    # PEC
    formula_ipcw = get.ipcw.surv.formula(event_col, colnames(infos$data), duration_col = duration_col)
    pred.times <- seq(1, 60, 1)
    pec_M = floor(0.7 * nrow(infos$data))
    pec_B = B
    registerDoParallel(nworkers)
    print(paste("PEC, B =", pec_B))
    #system.time({
    fitpec <- pec(list("Cox mean dose" = coxmean, 
                       "Cox Lasso doses-volumes" = coxlassodv, 
                       #"Cox Lasso heart's subparts screened dosiomics" = coxlasso_32X, 
                       "RSF whole-heart screened first-order dosiomics" = rsf.best
                       ), 
                  data = infos$data, formula = formula_ipcw,
                  times = pred.times, start = pred.times[1], 
                  exact = F, splitMethod = "BootCv", reference = F, 
                  B = pec_B, M = pec_M, keep.index = T, keep.matrix = T)
    #})
    print("End pec")
    saveRDS(fitpec, file = paste0(analyzes_dir, "pec_results/fit_pec_", pec_B, ".rds"))
    png(paste0(analyzes_dir, "pec_plots/bootcv.png"), width = 800, height = 600)
    plot(fitpec, what = "BootCvErr", xlim = c(0, 60),
         axis1.at = seq(0, 60, 5), axis1.label = seq(0, 60, 5))
    dev.off()
}

args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    analyzes_dir <- args[1]
    event_col <- args[2]
    duration_col <- `if`(length(args) == 3, args[3], "survival_time_years")
    file_dataset = paste0(analyzes_dir, "datasets/trainset.csv.gz")
    log_threshold(INFO)
    pec_estimation(file_dataset, event_col, analyzes_dir, duration_col)
} else {
    print("No arguments provided. Skipping.")
}


