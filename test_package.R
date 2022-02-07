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

P = 10
n_train = 500
n_test = 500
rho <- .3
nrep <- 10
d <- array(NA, dim = c(n_train, 2))
d_test <- array(NA, dim = c(n_test, 2))
Sigma <-rho*(1-diag(2)) + diag(2)
results <- list()

f_fun <- function(W){10*sin(pi*W[,1]*W[,2]) + 20*(W[,3]- 0.5)^2 + 10*W[,4] + 5*W[,5]}
g0 <- function(x){as.numeric(x > 0)}
m_mean <- function(x){as.numeric(x - mean(x))}



for(r in 1:nrep){
print(paste0("************* Repetition = ", r, " *************"))

W <- matrix(runif(P*n_train), ncol = P)
W_test <- matrix(runif(P*n_test), ncol = P)

means <- cbind(f_fun(W), f_fun(-W)) %>% apply(2, m_mean)
means_test <- cbind(f_fun(W_test), f_fun(-W_test))%>% apply(2, m_mean)


for (i in 1:n_train){d[i,] <- mvrnorm(n = 1, mu=means[i,], Sigma = Sigma)}
delta1 <- d[,1] > 0 %>%as.numeric()#rbinom(n = n_train, size = 1, prob = (1 + exp(-1.5*W[,1]))^(-1))
delta2 <- d[,2] > 0 %>%as.numeric()#rbinom(n = n_train, size = 1, prob = (1 + exp(-1.5*W[,2]))^(-1))



for (i in 1:n_test){d_test[i,] <- mvrnorm(n = 1, mu=means_test[i,], Sigma = Sigma)}
delta1_test <- (d_test[,1] > 0) %>%as.numeric()#rbinom(n = n_train, size = 1, prob = (1 + exp(-1.5*W[,1]))^(-1))
delta2_test <- (d_test[,2] > 0) %>%as.numeric()

hypers <- Hypers(W,delta1, delta2)
opts <- Opts(num_burn = 5000, num_thin = 1, num_save = 5000, num_print = 100)

#BART with shared forest model
sb <- SharedBartBinary(W = W,
           delta1=delta1,
           delta2=delta2,
            W_test=W_test,
           hypers_ = hypers,
           opts_ = opts)

sb$theta_hat_test1 %>%str()
#inverse link
sb$theta_hat_test1 %>% apply(2, mean) %>% pnorm()

#dirichlet probability posterior means - should pick out most important
s_hat <- sb$s %>%apply(2, mean)
ggplot() + geom_bar(aes(x = 1:ncol(W), y = s_hat), stat = "identity") +
  labs(x = "Variable", y = "Inclusion Probability (s-hat)")



#two individual BART models
b1 <- gbart(x.train = W,
            y.train = delta1,
            x.test = W_test,
            type = "pbart",
            printevery=1000)

b2 <- gbart(x.train = cbind(W, delta1), #train: use actual delta1
            y.train = delta2,
            x.test = cbind(W_test,b1$yhat.test %>%apply(2, mean) %>% g0), #test: use predicted delta1
            type = "pbart",
            printevery=1000)



fitmat<- data.frame(delta1_test, delta2_test,
           sb_pred1 = sb$theta_hat_test1%>%apply(2, mean) %>% g0,
           sb_pred2 = sb$theta_hat_test2%>%apply(2, mean) %>% g0,
           b_pred1 = b1$yhat.test %>%apply(2, mean) %>% g0,
           b_pred2 = b2$yhat.test%>%apply(2, mean) %>% g0)
fitmat <-  fitmat %>% mutate_all(as.factor)

#response 1
confusionMatrix(fitmat$sb_pred1, fitmat$delta1_test)$byClass[c(1,2)]
confusionMatrix(fitmat$b_pred1, fitmat$delta1_test)$byClass[c(1,2)]

#response 2
results[[r]] <- data.frame(model = c("Shared BART", "Sequential BARTs"),
                           rbind(confusionMatrix(fitmat$sb_pred2, fitmat$delta2_test)$byClass[c(1,2)]%>%t(),
                                 confusionMatrix(fitmat$b_pred2, fitmat$delta2_test)$byClass[c(1,2)]%>%t()))

}

resultsd <- do.call(rbind, results)

ggplot(data = resultsd) +
  geom_boxplot(aes(x = model, y = Sensitivity))

ggplot(data = resultsd) +
  geom_boxplot(aes(x = model, y = Specificity))



