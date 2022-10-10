
options(show.error.locations = TRUE, error=traceback)

suppressPackageStartupMessages({
library("caret", quietly = TRUE)
library("survival", quietly = TRUE)
library("randomForestSRC", quietly = TRUE)
library("pec", quietly = TRUE)
library("Hmisc", quietly = TRUE)
library("logger", quietly = TRUE)
library("parallel", quietly = TRUE)
})

source("workflow/scripts/utils_rsf.R")

# Script args
args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    nworkers <- get.nworkers()
    options(rf.cores = nworkers, mc.cores = nworkers)
    analyzes_dir <- args[1]
    for (i in 2:length(args)) {
        model_name <- args[i]
        save_results_dir <- paste0(analyzes_dir, "rsf/", model_name, "/")
        rsf.best <- readRDS(file = file = paste0(save_results_dir, "model.rds"))
        plot_vimp(rsf.best, analyzes_dir, model_name)
    }
} else{
    print("No arguments provided. Skipping.")
}

