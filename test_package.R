#library(devtools)
#install_github("LendieFollett/Multivariate-Heterogenous-Response-Prediction/SharedForestBinary-master/SharedForestBinary")
#or
install.packages("/Users/000766412/OneDrive - Drake University/Documents/GitHub/Multivariate-Heterogenous-Response-Prediction/SharedForestBinary-master/SharedForestBinary",
                 repos = NULL,
                 type = "source")
library(SharedForestBinary)
library(dplyr)
library(ggplot2)

P = 10
n_train = 300
n_test = 100
W <- matrix(rnorm(P*n_train), ncol = P)
delta1 <- rbinom(n = n_train, size = 1, prob = .5)
delta2 <- rbinom(n = n_train, size = 1, prob = .5)
W_test <- matrix(rnorm(P*n_test), ncol = P)

hypers <- Hypers(W,delta1, delta2)
opts <- Opts(num_burn = 2000, num_thin = 1, num_save = 2000, num_print = 10)

temp <- SharedBart(W = W,
           delta1=delta1,
           delta2=delta2,
            W_test=W_test,
           hypers_ = hypers,
           opts_ = opts)

temp$theta_hat_test1 %>%str()
#inverse link
temp$theta_hat_test1 %>% apply(2, mean) %>% pnorm()



