
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

# Script args
args = commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
    ntasks <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
    nworkers <- `if`(is.na(ntasks), parallel::detectCores(), ntasks)
    options(rf.cores = nworkers, mc.cores = nworkers)
    analyzes_dir <- args[1]
    for (i in 2:length(args)) {
        model_name <- args[i]
        rsf.best <- saveRDS(rsf.best, file = paste0(analyzes_dir, "rsf_results/fitted_models/", model_name, ".rds"))
        plot_vimp(rsf.obj, analyzes_dir, model_name)
    }
} else{
    print("No arguments provided. Skipping.")
}
