#for my use: Rcpp::compileAttributes()

#SharedForestBinary is a modification of the SharedForest package
rm(list = ls())
#library(devtools)
#install_github("LendieFollett/Multivariate-Heterogenous-Response-Prediction/SharedForestBinary-master/SharedForestBinary")
#or
#install.packages("/Users/000766412/OneDrive - Drake University/Documents/GitHub/Multivariate-Heterogenous-Response-Prediction/SharedForestBinary-master/SharedForestBinary",
#                 repos = NULL,
#                 type = "source")
library(SharedForestBinary)
library(dplyr)
library(ggplot2)
library(MASS)
library(BART)
library(caret)
library(randomForest)
library(reshape2)

#"s the variable selection task becomes more difficult, the model which does not share information is far more sensitive to irrelevant predictors than the model which does share"
P = 75
n_train = 500
n_test = 250
rho <- 0.0 #Note: shared bart assumes Z1, Z2 are independent GIVEN the Xs
nrep <- 10
d <- array(NA, dim = c(n_train, 2))
d_test <- array(NA, dim = c(n_test, 2))
Sigma <-rho*(1-diag(2)) + diag (2)


#see Section 4. Simulation Study in Linero paper
sigma_theta1 <- 4
sigma_theta2 <- 4


f_fun1 <- function(W){10*sin(pi*W[,1]*W[,2]) + 20*(W[,3]- 0.5)^2 + 10*W[,4] + 5*W[,5]}
f_fun2 <- function(W){5*sin(pi*W[,1]*W[,2]) + 25*(W[,3]- 0.5)^2 + 5*W[,4] + 5*W[,5]}

g0 <- function(x){as.numeric(x > 0)}
m_mean <- function(x){as.numeric(x - mean(x))}

opts <- Opts(num_burn = 5000, num_thin = 1, num_save = 5000, num_print = 1000)

  W <- matrix(runif(P*n_train), ncol = P)
  W_test <- matrix(runif(P*n_test), ncol = P)
  
  means <- c(rep(sigma_theta1, n_train), rep(sigma_theta2, n_train))*(cbind(f_fun1(W), f_fun1(-W))/20-0.7) #%>% apply(2, m_mean)
  means_test <-  c(rep(sigma_theta1, n_test), rep(sigma_theta2, n_test))*(cbind(f_fun1(W_test), f_fun1(-W_test))/20-0.7)#%>% apply(2, m_mean)
  
  
  for (i in 1:n_train){d[i,] <- mvrnorm(n = 1, mu=means[i,], Sigma = Sigma)}
  delta1 <- d[,1] > 0 %>%as.numeric()#rbinom(n = n_train, size = 1, prob = (1 + exp(-1.5*W[,1]))^(-1))
  delta2 <- d[,2] > 0 %>%as.numeric()#rbinom(n = n_train, size = 1, prob = (1 + exp(-1.5*W[,2]))^(-1))
  
  plot(d)
  cor(delta1, delta2)
  
  for (i in 1:n_test){d_test[i,] <- mvrnorm(n = 1, mu=means_test[i,], Sigma = Sigma)}
  delta1_test <- (d_test[,1] > 0) %>%as.numeric()#rbinom(n = n_train, size = 1, prob = (1 + exp(-1.5*W[,1]))^(-1))
  delta2_test <- (d_test[,2] > 0) %>%as.numeric()
  
  
  
  hypers <- Hypers(W,delta1, delta2, num_tree = 100)
  
  #BART with shared forest model
  sb <- SharedBartBinary(W = W,
                         delta1=delta1,
                         delta2=delta2,
                         W_test=W_test,
                         hypers_ = hypers,
                         opts_ = opts)
  
  qplot(as.factor(delta1_test), pnorm(sb$theta_hat_test1 %>% apply(2, mean)), geom = "boxplot")
  
  1-mean((pnorm(sb$theta_hat_test1 %>% apply(2, mean)) > .5) == delta1_test)
  
  qplot(as.factor(delta2_test), pnorm(sb$theta_hat_test2 %>% apply(2, mean)), geom = "boxplot")

  1-mean((pnorm(sb$theta_hat_test2 %>% apply(2, mean)) > .5) == delta2_test)
    
  b1 <- gbart(x.train = W,
              y.train = delta2,
              x.test = W_test, 
              type = "pbart",
              printevery=1000)
  
  qplot(as.factor(delta2_test), pnorm(b1$yhat.test %>% apply(2, mean)), geom = "boxplot")
  
  1-mean((pnorm(b1$yhat.test %>% apply(2, mean)) > .5) == delta2_test)
  
  qplot( pnorm(b1$yhat.test %>% apply(2, mean)), pnorm(sb$theta_hat_test2 %>% apply(2, mean)))
  