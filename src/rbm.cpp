#include"rbm.h"

using namespace rbm;

Parametar::Parametar(int visible_dim, int hidden_dim):
  visible_bias(visible_dim), hidden_bias(hidden_dim), weight(visible_dim, hidden_dim)
{}

Model::Model(){}

Model::Model(int visible_dim, int hidden_dim){
  this -> visible_dim = visible_dim;
  this -> hidden_dim = hidden_dim;
}

Model::~Model(){}

Eigen::VectorXd Model::lambda_visible(Parametar parametar, Eigen::VectorXi rand_hidden){
  Eigen::VectorXd lambda(parametar.visible_bias.size());
  lambda = parametar.visible_bias + (parametar.weight * rand_hidden.cast<double>());
  return lambda;
}


Eigen::VectorXd Model::lambda_hidden(Parametar parametar, Eigen::VectorXi rand_visible){
  Eigen::VectorXd lambda(parametar.hidden_bias.size());
  lambda = parametar.hidden_bias + (parametar.weight.transpose() * rand_visible.cast<double>());
  return lambda;
}

double Model::cost_func(Parametar parametar, Eigen::VectorXi rand_visible, Eigen::VectorXi rand_hidden){
  double visible_bias_term, hideen_bias_term, weight_bias_term;
  visible_bias_term = parametar.visible_bias.transpose()*rand_visible.cast<double>();
  hideen_bias_term = parametar.hidden_bias.transpose()*rand_hidden.cast<double>();
  weight_bias_term = (parametar.weight.transpose()*rand_visible.cast<double>()).transpose()*rand_hidden.cast<double>();
  return - (visible_bias_term + hideen_bias_term + weight_bias_term);
}

