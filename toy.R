
library(SharedForest)
library(zeallot)
gen_data_fried <- function(n, P,sigma = 1, sigma_theta = 1) {
  f_fried <- function(x) 10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3] - 0.5)^2 + 
    10 * x[,4] + 5 * x[,5]
  X <- matrix(runif(n * P), nrow = n)
  mu <- f_fried(X)
  theta <- sigma_theta * (f_fried(X) / 20 - .7)
  
  Y <- mu + sigma * rnorm(n)
  
  delta <- ifelse(rnorm(n, theta) < 0, 0, 1)
  
  return(list(X = X, Y = Y, W = X, delta = delta, mu = mu, theta = theta))
}
set.seed(7554450)
c(X,Y,W,delta,mu,theta)            %<-% gen_data_fried(250, 100)
c(Xt, Yt, Wt, deltat, mut, thetat) %<-% gen_data_fried(250,100)

hypers <- SharedForest::Hypers(X = X, Y = Y, W = W, delta = delta)
opts   <- SharedForest::Opts()

fit_shared  <- SharedBart(X = X, Y = Y, W = W, delta = delta, X_test = Xt, 
                          W_test = Wt, hypers_ = hypers, opts_ = opts)

summary(fit_shared$mu_hat_test[,1:10]) #There are some MCMC samples that are NA's
str(fit_shared$mu_hat_test)

which(is.na(fit_shared$mu_hat_test[,10]))
which(is.na(fit_shared$tau_hat_test[,10]))



plot(1:2500, fit_shared$mu_hat_test[,10], type = "l")
summary(fit_shared$tau_hat_test[,1:10])

how_many <- apply(fit_shared$mu_hat_test, 2, function(x){sum(is.na(x))})

summary(X[how_many > 100,1:10])

summary(X[how_many ==0,1:10])
