#include "recbart.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List SharedBartBinary(arma::mat& W,
                arma::uvec& delta1,
                arma::uvec& delta2,
                arma::mat& W_test,
                List hypers_,
                List opts_) {

  arma::uvec group = hypers_["group"];
  Hypers hypers(W, group, hypers_);
  Opts opts(opts_);

  MyData data(W,delta1,delta2, hypers.theta_01, hypers.theta_02);


//SAVE ROOM
  mat theta_hat1 = zeros<mat>(opts.num_save, W.n_rows);
  mat theta_hat2 = zeros<mat>(opts.num_save, W.n_rows);//LRF assuming same dimension
  mat s = zeros<mat>(opts.num_save, hypers.num_groups);
  mat theta_hat_test1 = zeros<mat>(opts.num_save, W_test.n_rows);
  mat theta_hat_test2 = zeros<mat>(opts.num_save, W_test.n_rows);

  std::vector<Node*> forest = init_forest(hypers);

  for(int i = 0; i < opts.num_burn; i++) {
    if(i > opts.num_burn / 2) {
      IterateGibbsWithS(forest, data, opts);
    }
    else {
      IterateGibbsNoS(forest, data, opts);
    }
    UpdateZ(data);
    if(i % opts.num_print == 0) Rcout << "Finishing warmup BLAH " << i << "\t\t\r";
    // if(i % 100 == 0) Rcout << "Finishing warmup " << i << std::endl;
  }

  Rcout << std::endl;

  for(int i = 0; i < opts.num_save; i++) {
    for(int j = 0; j < opts.num_thin; j++) {
      //Rcout << "Started second loop " << i << std::endl;
      IterateGibbsWithS(forest, data, opts);
      //Rcout << "Did IterageGibbsWithS " << i << std::endl;
      UpdateZ(data);
      //Rcout << "Did UpdateZ " << i << std::endl;
    }
    if(i % opts.num_print == 0) Rcout << "Finishing save " << i << "\t\t\r";

    theta_hat1.row(i) = trans(data.theta_hat1);
    theta_hat2.row(i) = trans(data.theta_hat2);
    //Rcout << "Did theta_hat1,2 " << i << std::endl;
    s.row(i) = trans(hypers.s);
    //Rcout << "Did hypers.s " << i << std::endl;
    mat theta_hat = predict_theta(forest, W_test);
    theta_hat_test1.row(i) = trans(theta_hat.col(0)) + hypers.theta_01;
    theta_hat_test2.row(i) = trans(theta_hat.col(1)) + hypers.theta_02;
  }
  Rcout << std::endl;

  Rcout << "Number of leaves at final iterations:\n";
  for(int t = 0; t < hypers.num_trees; t++) {
    Rcout << leaves(forest[t]).size() << " ";
    if((t + 1) % 10 == 0) Rcout << "\n";
  }

  List out;

  out["theta_hat1"] = theta_hat1;
  out["theta_hat2"] = theta_hat2;
  out["theta_hat_mean1"] = mean(theta_hat1, 0);
  out["theta_hat_mean2"] = mean(theta_hat2, 0);
  out["s"] = s;
  out["theta_hat_test1"] = theta_hat_test1;
  out["theta_hat_test2"] = theta_hat_test2;

  return out;
}

