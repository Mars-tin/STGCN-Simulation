#!/usr/bin/env Rscript
library(glasso)

args = commandArgs(trailingOnly=TRUE)

in_path = args[1]
out_path = args[2]
rho_ = as.numeric(args[3])
nobs_ = as.numeric(args[4])
x_csv = read.csv(in_path, sep=",", header=FALSE)
x <- as.matrix(x_csv)
dimnames(x) <- list(NULL, NULL)
sigma <- var(x)
out <- glasso(sigma, rho=rho_, nobs=nobs_)
w <- out[[1]]
write.table(x = w, file = out_path, sep = ',', row.names = FALSE, col.names = FALSE)
