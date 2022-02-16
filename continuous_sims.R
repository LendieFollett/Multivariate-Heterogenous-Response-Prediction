# Import packages
rm(list = ls())
#library(devtools)
#install_github("theodds/SharedForestPaper/SharedForest")
library(SharedForest)
library(dplyr)
library(ggplot2)
library(MASS)
library(BART)
library(caret)
library(randomForest)
library(reshape2)

# Set file path
out = "/Users/hendersonhl/Documents/Articles/Multivariate-Heterogenous-Response-Prediction"

# Set parameters
P = 150
n_train = 500
n_test = 250
rho <- 0.0 
nrep <- 100
d <- array(NA, dim = c(n_train, 2))
d_test <- array(NA, dim = c(n_test, 2))
Sigma <-rho*(1-diag(2)) + diag (2)
sigma_theta1 <- 1   # Removes scaling for continuous outcome
sigma_theta2 <- 4

# Define helper functions
f_fun1 <- function(W){10*sin(pi*W[,1]*W[,2]) + 20*(W[,3]- 0.5)^2 + 10*W[,4] + 5*W[,5]}
f_fun2 <- function(W){5*sin(pi*W[,1]*W[,2]) + 25*(W[,3]- 0.5)^2 + 5*W[,4] - 10*W[,5]}
g0 <- function(x){as.numeric(x > 0)}
m_mean <- function(x){as.numeric(x - mean(x))}

# Set up for simulations
opts <- Opts(num_burn = 5000, num_thin = 1, num_save = 5000, num_print = 1000)
fitmat <- list()

# Start simulations
for(r in 1:nrep){
  print(paste0("************* Repetition = ", r, " *************"))
  start.time <- Sys.time()

  # Generate covariates
  W <- matrix(runif(P*n_train), ncol = P)
  W_test <- matrix(runif(P*n_test), ncol = P)
  
  # Generate expected values
  means <- c(rep(sigma_theta1, n_train), rep(sigma_theta2, n_train))*(cbind(f_fun1(W), f_fun2(W))/20 - 0.7)
  means_test <-  c(rep(sigma_theta1, n_test), rep(sigma_theta2, n_test))*(cbind(f_fun1(W_test), f_fun2(W_test))/20 - 0.7)
  
  # Generate training outcomes
  for (i in 1:n_train){d[i,] <- mvrnorm(n = 1, mu = means[i,], Sigma = Sigma)}
  Y <- scale(d[,1])
  delta <- (d[,2] > 0) %>% as.numeric()
  
  # Generate testing outcomes
  for (i in 1:n_test){d_test[i,] <- mvrnorm(n = 1, mu = means_test[i,], Sigma = Sigma)}
  Y_test <- scale(d_test[,1])
  delta_test <- (d_test[,2] > 0) %>% as.numeric()
  
  # Shared forest model
  hypers <- SharedForest::Hypers(X = W, Y = Y, W = W, delta = delta, num_tree = 200)
  sb  <- SharedBart(X = W, Y = Y, W = W, delta = delta, X_test = W_test, 
                            W_test = W_test, hypers_ = hypers, opts_ = opts)
  
  # Individual BART model for binary outcome
  b1 <- gbart(x.train = W, y.train = delta, x.test = W_test, type = "pbart", printevery=1000)
  
  # Subsample of testing where delta is predicted to be 0
  predicted_theta <- b1$yhat.test %>% apply(2, mean)
  b_idx0 <- which(predicted_theta < 0)
  
  # Individual BART models for continuous outcome
  b2.1 <- gbart(x.train = W[delta == 0,],
                y.train = Y[delta == 0],
                x.test = W_test[b_idx0,],
                type = "wbart",
                printevery=1000)
  b2.2 <- gbart(x.train = W[delta == 1,],
                y.train = Y[delta == 1],
                x.test = W_test[-b_idx0,],
                type = "wbart",
                printevery=1000)

  # Shared BART predictions
  sb_y_pred <- sb$mu_hat_test %>% apply(2, mean, na.rm=TRUE)
  sb_delta_pred <- ifelse(pnorm(sb$theta_hat_test %>% apply(2, mean)) > 0.5, 1, 0)
  
  # BART predictions
  b_delta_pred <- ifelse(pnorm(b1$yhat.test %>% apply(2, mean)) > 0.5, 1, 0)
  b_y_pred_delta_0 <- b2.1$yhat.test %>% apply(2, mean)
  b_y_pred_delta_1 <- b2.2$yhat.test %>% apply(2, mean)

  # True and observed  P(y | delta) in test set
  fm <- data.frame(true_y_delta_0 = mean(Y_test[delta_test == 0]),
                   true_y_delta_1 = mean(Y_test[delta_test == 1]),
                   b_y_pred_delta_0 = mean(b_y_pred_delta_0),
                   b_y_pred_delta_1 = mean(b_y_pred_delta_1),
                   sb_y_pred_delta_0 = mean(sb_y_pred[sb_delta_pred == 0]),
                   sb_y_pred_delta_1 = mean(sb_y_pred[sb_delta_pred == 1]))
  fitmat[[r]] <- fm
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  print(time.taken)
}

# Save results
fitmatd <- do.call(rbind, fitmat)
path <- paste0(out, "/continuous_sims.csv")
write.csv(fitmatd, path, row.names = FALSE)

# Import results and organize for plotting
path <- paste0(out, "/continuous_sims.csv")
fitmatd <- read.csv(path)
delta_0_se <- (fitmatd$sb_y_pred_delta_0 - fitmatd$true_y_delta_0)^2 # Shared forest squared errors
delta_1_se <- (fitmatd$sb_y_pred_delta_1 - fitmatd$true_y_delta_1)^2
model <- c(rep("SF", length(fitmat)))
sbse <- data.frame(delta_0_se, delta_1_se, model)
delta_0_se <- (fitmatd$b_y_pred_delta_0 - fitmatd$true_y_delta_0)^2  # BART squared errors
delta_1_se <- (fitmatd$b_y_pred_delta_1 - fitmatd$true_y_delta_1)^2
model <- c(rep("BART", length(fitmat)))
bse <- data.frame(delta_0_se, delta_1_se, model)
results <- rbind(sbse, bse)  # Combine all results

# Plot results
ggplot(results, aes(x = delta_0_se, fill = model)) +
  geom_density(position = "identity", alpha = .8) + 
  scale_fill_manual(values=c("grey20", "grey60")) + theme_bw()
ggplot(results, aes(x = delta_1_se, fill = model)) +
  geom_density(position = "identity", alpha = .8) + 
  scale_fill_manual(values=c("grey20", "grey60")) + theme_bw()

# Descriptive stats by group
results %>% group_by(model) %>% summarize(mean = mean(delta_0_se))
results %>% group_by(model) %>% summarize(mean = mean(delta_1_se))




