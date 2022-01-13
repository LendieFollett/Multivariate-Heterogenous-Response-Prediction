#just run once:
#library(devtools)
#install_github("theodds/SharedForestPaper/SharedForest")

library(SharedForest)
library(dplyr)

#how many observations in training data
n_train <- 100
p <- 10

X <- rnorm(n_train*p) %>% matrix(ncol = p)
X_norm <- W_norm <- X %>%apply(2, function(x){(x-min(x))/(max(x)-min(x))})

Y <- rnorm(n_train)
Y_scale <- Y

delta <- rbinom(n = n_train, size = 1, prob = 0.5)


hypers <- Hypers(X = X, Y = Y_scale, W = W, delta = delta)
opts <- Opts(num_burn = 2000, num_thin = 1, num_save = 2000, num_print = 10)


fit_shared <- SharedBart(X = X_norm, Y = Y_scale, W = W_norm, delta = delta,
                         X_test = X_norm, W_test = W_norm, hypers_ = hypers,
                         opts_ = opts)



