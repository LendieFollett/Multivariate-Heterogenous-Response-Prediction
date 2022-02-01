#include "recbart.h"

using namespace arma;
using namespace Rcpp;

// arma::vec loglik_data(const arma::vec& Y,
//                       const arma::vec& rho,
//                       const Hypers& hypers) {

//   vec out = zeros<vec>(Y.size());
//   for(int i = 0; i < Y.size(); i++) {
//     out(i) = Y(i) * rho(i) - std::exp(rho(i)) - R::lgammafn(Y(i) + 1);
//   }
//   return out;
// }


void IterateGibbsNoS(std::vector<Node*>& forest,
                     MyData& data,
                     const Opts& opts) {

  TreeBackfit(forest, data, opts);
  forest[0]->hypers->UpdateTau(data);
  // UpdateSigmaParam(forest);

  Rcpp::checkUserInterrupt();
}

void IterateGibbsWithS(std::vector<Node*>& forest,
                       MyData& data,
                       const Opts& opts) {

  IterateGibbsNoS(forest, data, opts);
  if(opts.update_s) UpdateS(forest);
  if(opts.update_alpha) forest[0]->hypers->UpdateAlpha();

}

//Section 3.1 in original BART paper
void TreeBackfit(std::vector<Node*>& forest,
                 MyData& data,
                 const Opts& opts) {

  double MH_BD = 0.7;
  Hypers* hypers = forest[0]->hypers;
  for(int t = 0; t < hypers->num_trees; t++) {
    BackFit(forest[t], data);
      //mat mu_tau = predict_reg(tree, data); pred mean, variance parameters for each row in X for just tree t
      //vec theta = predict_theta(tree, data); pred theta for each row in X for just tree t
      //data.mu_hat = data.mu_hat - mu_tau.col(0); what would prediction be without that tree
      //data.tau_hat = data.tau_hat / mu_tau.col(1);
      //data.theta_hat = data.theta_hat - theta;
    if(forest[t]->is_leaf || unif_rand() < MH_BD) {
      birth_death(forest[t], data);
    }
    else {
      change_decision_rule(forest[t], data);
    }
    forest[t]->UpdateParams(data);
    Refit(forest[t], data);
    //mat mu_tau = predict_reg(tree, data);
    //vec theta = predict_theta(tree, data);
    //data.mu_hat = data.mu_hat + mu_tau.col(0);
    //data.tau_hat = data.tau_hat % mu_tau.col(1);
    //data.theta_hat = data.theta_hat + theta;
  }
}

//https://stackoverflow.com/questions/17387617/c-calling-a-function-inside-the-same-functions-definition

arma::mat predict_reg(Node* tree, MyData& data) {
  int N = data.X.n_rows;
  mat out = zeros<mat>(N, 2);
  for(int i = 0; i < N; i++) {
    rowvec x = data.X.row(i);
    out.row(i) = predict_reg(tree,x);
  }
  return out;
}

arma::mat predict_reg(Node* tree, arma::mat& X) {
  int N = X.n_rows;
  mat out = zeros<mat>(N, 2);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out.row(i) = predict_reg(tree,x);
  }
  return out;
}

//recursive function
arma::rowvec predict_reg(Node* n, rowvec& x) {
  if(n->is_leaf) { // if it's a leaf / terminal node
    rowvec out = zeros<rowvec>(2);
    out(0) = n->mu;
    out(1) = n->tau;
    return out;
  }
  if(x(n->var) <= n->val) { //if splitting variable (Xj) is less than cuttoff ()Cj)
    return predict_reg(n->left, x);
  }
  else {
    return predict_reg(n->right, x);
  }
}

arma::vec predict_theta(std::vector<Node*> forest, arma::mat& W) {
  int N = forest.size();
  vec out = zeros<vec>(W.n_rows);
  for(int n = 0 ; n < N; n++) {
    out = out + predict_theta(forest[n], W);
  }
  return out;
}

//predict_reg FOREST input
arma::mat predict_reg(std::vector<Node*> forest, arma::mat& X) {
  int N = forest.size();
  mat out = zeros<mat>(X.n_rows,2);
  out.col(1) = ones<vec>(X.n_rows);
  for(int n = 0 ; n < N; n++) { // iterate over each tree
    mat mutau = predict_reg(forest[n], X);
    out.col(0) = out.col(0) + mutau.col(0); // add each tree's component
    out.col(1) = out.col(1) % mutau.col(1);
  }
  return out;
}

arma::vec predict_theta(Node* tree, arma::mat& W) {
  int N = W.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec w = W.row(i);
    out(i) = predict_theta(tree,w);
  }
  return out;
}

arma::vec predict_theta(Node* tree, MyData& data) {
  int N = data.W.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec w = data.W.row(i);
    out(i) = predict_theta(tree,w);
  }
  return out;
}

//recursive function
double predict_theta(Node* n, rowvec& w) {
  if(n->is_leaf) {
    return n->theta;
  }
  if(w(n->var) <= n->val) {
    return predict_theta(n->left, w);
  }
  else {
    return predict_theta(n->right,w);
  }
}

void BackFit(Node* tree, MyData& data) {
  mat mu_tau = predict_reg(tree, data);
  vec theta = predict_theta(tree, data);
  data.mu_hat = data.mu_hat - mu_tau.col(0);
  data.tau_hat = data.tau_hat / mu_tau.col(1);
  data.theta_hat = data.theta_hat - theta;
}

void Refit(Node* tree, MyData& data) {
  mat mu_tau = predict_reg(tree, data);
  vec theta = predict_theta(tree, data);
  data.mu_hat = data.mu_hat + mu_tau.col(0);
  data.tau_hat = data.tau_hat % mu_tau.col(1);
  data.theta_hat = data.theta_hat + theta;
}

void Node::UpdateParams(MyData& data) {

  UpdateSuffStat(data);
  std::vector<Node*> leafs = leaves(this);
  int num_leaves = leafs.size();
  double a1 = 1.0/pow(hypers->sigma_theta1,2);
  double a2 = 1.0/pow(hypers->sigma_theta2,2);
  for(int i = 0; i < num_leaves; i++) {
    Node* l      = leafs[i];

//LRF:  starting here, will need to modify to include delta1 and delta2
    double theta_hat1 = l->ss.sum_Z1 / (l->ss.n_Z1 + a1);
    double sigma_theta1 = pow(l->ss.n_Z1 + a1, -0.5);
    l->theta1 = theta_hat1 + norm_rand() * sigma_theta1; // update theta

    double theta_hat2 = l->ss.sum_Z2 / (l->ss.n_Z2 + a2);
    double sigma_theta2 = pow(l->ss.n_Z2 + a2, -0.5);
    l->theta2 = theta_hat2 + norm_rand() * sigma_theta2; // update theta

  }
}

double Node::LogLT(const MyData& data) {

  UpdateSuffStat(data);
  std::vector<Node*> leafs = leaves(this);


  double out = 0.0;
  int num_leaves = leafs.size();
  double a1     = 1.0 / (hypers->sigma_theta1 * hypers->sigma_theta1);
  double a2     = 1.0 / (hypers->sigma_theta2 * hypers->sigma_theta2);

  for(int i = 0; i < num_leaves; i++) {

    // Define stuff
    Node* l = leafs[i];
    double n_Z1 = l->ss.n_Z1;
    double R_bar1 = l->ss.sum_Z1 / n_Z1;
    double SSE_Z1 = l->ss.sum_Z_sq1 - n_Z1 * R_bar1 * R_bar1;

    double n_Z2 = l->ss.n_Z2;
    double R_bar2 = l->ss.sum_Z2 / n_Z2;
    double SSE_Z2 = l->ss.sum_Z_sq2 - n_Z2 * R_bar2 * R_bar2;

    // Likelihood for classification
    //LRF: this needs to be done for both binary responses (see notes)
    // ---> update the SuffStat functions
    //NOTE: this assumes Z1 and Z2 have the same dimension
    if(n_Z1 > 0.0) {
      out += 0.5 * log(a1 / (n_Z1 + a1)) - n_Z1 * M_LN_SQRT_2PI
        - 0.5 * (SSE_Z1 + n_Z1 * a1 * R_bar1 * R_bar1 / (n_Z1 + a1)) +
          0.5 * log(a2 / (n_Z2 + a2)) - n_Z2 * M_LN_SQRT_2PI //LRF
      - 0.5 * (SSE_Z2 + n_Z2 * a2 * R_bar2 * R_bar2 / (n_Z2 + a2)); //LRF

    }

  }
  return out;
}

//LRF:will not need first loop
void Node::UpdateSuffStat(const MyData& data) {
  ResetSuffStat();
  int M = data.W.n_rows; //LRF: allow W1 and W2 of differing dimensions? or assume same...
  for(int i = 0; i < M; i++) {
    AddSuffStatZ(data,i);
  }
}


// LRF: There will be two 'Z' elements within data
void Node::AddSuffStatZ(const MyData& data, int i) {
  double Z1 = data.Z1(i) - data.theta_hat1(i);
  double Z2 = data.Z2(i) - data.theta_hat2(i);
  ss.sum_Z1 += Z1;
  ss.sum_Z_sq1 += Z1 * Z1;
  ss.n_Z1 += 1.0;

  ss.sum_Z2 += Z2;
  ss.sum_Z2_sq += Z2 * Z2;
  ss.n_Z2 += 1.0;

  if(!is_leaf) {
    double w = data.W(i,var);
    if(w <= val) {
      left->AddSuffStatZ(data,i);
    } else {
      right->AddSuffStatZ(data,i);
    }
  }
}

void Node::ResetSuffStat() {
  ss.sum_Z1      = 0.0;
  ss.sum_Z_sq1   = 0.0;
  ss.n_Z1        = 0.0;
  ss.sum_Z2      = 0.0;
  ss.sum_Z_sq2   = 0.0;
  ss.n_Z2        = 0.0;
  if(!is_leaf) {
    left->ResetSuffStat();
    right->ResetSuffStat();
  }
}

double cauchy_jacobian(double tau, double sigma_hat) {
  double sigma = pow(tau, -0.5);
  int give_log = 1;

  double out = Rf_dcauchy(sigma, 0.0, sigma_hat, give_log);
  out = out - M_LN2 - 3.0 / 2.0 * log(tau);

  return out;

}

// void UpdateZ(MyData& data) {

//   int N = data.Z.n_elem;

//   for(int i = 0; i < N; i++) {

//     if(data.delta(i) == 0) {
//       data.Z(i) = randnt(data.theta_hat(i), 1.0, R_NegInf, 0.0);
//     }
//     else {
//       data.Z(i) = randnt(data.theta_hat(i), 1.0, 0.0, R_PosInf);
//     }
//   }
// }

//http://web.michaelchughes.com/research/sampling-from-truncated-normal
void UpdateZ(MyData& data) {

  int N = data.Z1.n_elem; // LRF: again, assume same dimensions Z1 vs Z2?

  for(int i = 0; i < N; i++) {

    double u1 = unif_rand();
    double u2 = unif_rand();
    double Z1 = 0.0;
    double Z2 = 0.0;
    //Sampling inversion of numerical approximation to CDF
    if(data.delta(i) == 0) { // if Y = 0
      data.Z1(i) = -R::qnorm((1.0 - u1) * R::pnorm( data.theta_hat1(i), 0.0, 1.0, 1, 0) + u1, 0.0, 1.0,1,0);
      data.Z2(i) = -R::qnorm((1.0 - u2) * R::pnorm( data.theta_hat2(i), 0.0, 1.0, 1, 0) + u2, 0.0, 1.0,1,0);
    }
    else { // if Y = 1
      data.Z1(i) =  R::qnorm((1.0 - u1) * R::pnorm(-data.theta_hat1(i), 0.0, 1.0, 1 ,0) + u1, 0.0, 1.0, 1,0);
      data.Z2(i) =  R::qnorm((1.0 - u2) * R::pnorm(-data.theta_hat2(i), 0.0, 1.0, 1 ,0) + u2, 0.0, 1.0, 1,0);
    }
    data.Z1(i) = data.Z1(i) + data.theta_hat1(i);
    data.Z2(i) = data.Z2(i) + data.theta_hat2(i);
  }
}

void UpdateSigmaParam(std::vector<Node*>& forest) {

  mat theta = get_params(forest);
  vec theta1 = theta.col(0);
  vec theta2 = theta.col(1);

  double Lambda_theta = sum(theta1 % theta1);
  double num_leaves = mu.size();
  double sigma_theta_hat = forest[0]->hypers->sigma_theta_hat1;

  // Update sigma_theta1
  //LRF: UPDATED
  double prec_theta_old = pow(forest[0]->hypers->sigma_theta1, -2.0);
  double prec_theta_new = Rf_rgamma(1.0 + 0.5 * num_leaves, 1.0 / (0.5 * Lambda_theta));
  loglik_rat = cauchy_jacobian(prec_theta_new, sigma_theta_hat) - cauchy_jacobian(prec_theta_old, sigma_theta_hat);
  double prec = log(unif_rand()) < loglik_rat ? prec_theta_new : prec_theta_old;

  forest[0]->hypers->sigma_theta1 = pow(prec, -0.5);

  double Lambda_theta = sum(theta2 % theta2);
  double num_leaves = mu.size();
  double sigma_theta_hat = forest[0]->hypers->sigma_theta_hat2;

  // Update sigma_theta2
  //LRF: UPDATED
  double prec_theta_old = pow(forest[0]->hypers->sigma_theta2, -2.0);
  double prec_theta_new = Rf_rgamma(1.0 + 0.5 * num_leaves, 1.0 / (0.5 * Lambda_theta));
  loglik_rat = cauchy_jacobian(prec_theta_new, sigma_theta_hat) - cauchy_jacobian(prec_theta_old, sigma_theta_hat);
  double prec = log(unif_rand()) < loglik_rat ? prec_theta_new : prec_theta_old;

  forest[0]->hypers->sigma_theta2 = pow(prec, -0.5);

}


arma::mat get_params(std::vector<Node*>& forest) {
  std::vector<double> theta1(0);
  std::vector<double> theta2(0);
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_params(forest[t],  theta1, theta2);
  }

  int num_leaves = theta1.size();
  mat theta = zeros<mat>(num_leaves, 2); // LRF: 2 for theta1, theta2
  for(int i = 0; i < num_leaves; i++) {
    theta(i,0) = theta1[i];
    theta(i,1) = theta2[i];

  }

  return theta;

}

void get_params(Node* n,
                std::vector<double>& theta1,
                std::vector<double>& theta2
                )
{
  if(n->is_leaf) {
    theta.push_back(n->theta);
  }
  else {
    get_params(n->left, theta1, theta2);
    get_params(n->right, theta1, theta2);
  }
}

Cluster::Cluster(const arma::uvec& cluster, const arma::uvec& clusterw) {

  this->cluster = cluster;
  this->clusterw = clusterw;

  int num_cluster = cluster.max() + 1;
  int num_clusterw = clusterw.max() + 1;
  V_mu = zeros<vec>(num_cluster);
  V_theta = zeros<vec>(num_clusterw);
  V_tau = ones<vec>(num_cluster);

  sigma_V_mu    = 0.2 * exp_rand();
  sigma_V_theta = 0.2 * exp_rand();
  a_V_tau       = 100.0;

  cluster_to_idx.resize(num_cluster);
  for(int i = 0; i < num_cluster; i++) {
    cluster_to_idx[i] = find(cluster == i);
  }
  clusterw_to_idx.resize(num_clusterw);
  for(int i = 0; i < num_clusterw; i++) {
    clusterw_to_idx[i] = find(clusterw == i);
  }
}

void UpdateVmu(Cluster& cluster, MyData& data) {

  int nclust = cluster.V_mu.size();
  double a = pow(cluster.sigma_V_mu, -2.0);

  for(int i = 0; i < nclust; i++) {

    double sum_tau = 0.0;
    double sum_tau_R = 0.0;
    int clust_size = cluster.cluster_to_idx[i].size();

    for(int j = 0; j < clust_size; j++) {
      int n = cluster.cluster_to_idx[i](j);
      sum_tau += data.tau_hat(n);
      data.mu_hat(n) = data.mu_hat(n) - cluster.V_mu(i);
      sum_tau_R += data.tau_hat(n) * (data.Y(n) - data.mu_hat(n));
    }

    double mu_up = sum_tau_R / (sum_tau + a);
    double sigma_up = pow(sum_tau + a, -0.5);
    cluster.V_mu(i) = mu_up + sigma_up * norm_rand();

    for(int j = 0; j < clust_size; j++) {
      int n = cluster.cluster_to_idx[i](j);
      data.mu_hat(n) = data.mu_hat(n) + cluster.V_mu(i);
    }
  }
}

void UpdateVtheta(Cluster& cluster, MyData& data) {

  int nclust = cluster.V_theta.size();
  double a = pow(cluster.sigma_V_theta, -2.0);

  for(int i = 0; i < nclust; i++) {
    double sum_tau = 0.0;
    double sum_R = 0.0;
    int clust_size = cluster.clusterw_to_idx[i].size();

    for(int j = 0; j < clust_size; j++) {
      int n = cluster.clusterw_to_idx[i](j);
      sum_tau += 1;
      data.theta_hat(n) = data.theta_hat(n) - cluster.V_theta(i);
      sum_R += data.Z(n) - data.theta_hat(n);
    }

    double mu_up = sum_R / (sum_tau + a);
    double sigma_up = pow(sum_tau + a, -0.5);
    cluster.V_theta(i) = mu_up + sigma_up * norm_rand();

    for(int j = 0; j < clust_size; j++) {
      int n = cluster.clusterw_to_idx[i](j);
      data.theta_hat(n) = data.theta_hat(n) + cluster.V_theta(i);
    }
  }
}

void UpdateVtau(Cluster& cluster, MyData& data) {
  int nclust = cluster.V_tau.size();

  for(int i = 0; i < nclust; i++) {
    double ns = 0.0;
    double SS = 0.0;
    int clust_size = cluster.cluster_to_idx[i].size();

    for(int j = 0; j < clust_size; j++) {
      int n = cluster.cluster_to_idx[i](j);
      ns += 1.0;
      double a = data.tau_hat(n);
      double b = cluster.V_tau(i);
      double c = a + b;
      data.tau_hat(n) = data.tau_hat(n) / cluster.V_tau(i);
      SS += data.tau_hat(n) * pow(data.Y(n) - data.mu_hat(n), 2.0);
    }
    cluster.V_tau(i) = R::rgamma(cluster.a_V_tau + 0.5 * ns, 1.0 / (cluster.a_V_tau + 0.5 * SS));
    for(int j = 0; j < clust_size; j++) {
      int n = cluster.cluster_to_idx[i](j);
      data.tau_hat(n) = data.tau_hat(n) * cluster.V_tau(i);
    }
  }
}

void UpdateShape(Cluster& cluster) {
  ShapeLoglik shape(sum(log(cluster.V_tau)),
                    sum(cluster.V_tau),
                    cluster.V_tau.size());

  double sigma = 1.0 / sqrt(cluster.a_V_tau);
  cluster.a_V_tau = pow(slice_sampler(sigma, &shape, 1.0, 0.0, 10000.0), -2.0);

}
void UpdateVars(Cluster& cluster, MyData& data) {

  double N_mu = cluster.V_mu.size();
  double N_theta = cluster.V_theta.size();
  double SSE_mu = (N_mu - 1.0) * var(cluster.V_mu);
  double SSE_theta = (N_theta - 1.0) * var(cluster.V_theta);

  double tau_mu = R::rgamma(0.5 * N_mu, 2.0 / SSE_mu);
  double tau_theta = R::rgamma(0.5 * N_theta, 2.0 / SSE_theta);

  cluster.sigma_V_mu = pow(tau_mu, -0.5);
  cluster.sigma_V_theta = pow(tau_theta, -0.5);

}

