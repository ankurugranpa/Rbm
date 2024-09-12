#include<random>
#include<iostream>

#include"model.h"

using namespace rbm;


Model::Model(): visible_dim(0), hidden_dim(0), parametar(0, 0) {}

Model::Model(int visible_dim, int hidden_dim): visible_dim(visible_dim), hidden_dim(hidden_dim), parametar(visible_dim, hidden_dim)
{
  // パラメータの初期化
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  for(int i=0; i<visible_dim; i++){
    parametar.visible_bias(i) = dis(gen);
    for(int j=0; j<hidden_dim; j++){
      parametar.weight(i, j) = dis(gen);
    }
  }
  for(int j=0; j<hidden_dim; j++){
    parametar.hidden_bias(j) = dis(gen);
  }

}

Model::~Model(){}

Eigen::VectorXd Model::lambda_visible(const Parametar& parametar, const Eigen::VectorXi& rand_hidden)const{
  Eigen::VectorXd lambda(parametar.visible_bias.size());
  lambda = parametar.visible_bias + (parametar.weight * rand_hidden.cast<double>());
  return lambda;
}


Eigen::VectorXd Model::lambda_hidden(const Parametar& parametar, const Eigen::VectorXi& rand_visible)const{
  Eigen::VectorXd lambda(parametar.hidden_bias.size());
  lambda = parametar.hidden_bias + (parametar.weight.transpose() * rand_visible.cast<double>());
  return lambda;
}

double Model::cost_func(const Parametar& parametar, const Eigen::VectorXi& rand_visible, const Eigen::VectorXi& rand_hidden)const{
  double visible_bias_term, hideen_bias_term, weight_bias_term;
  visible_bias_term = parametar.visible_bias.transpose()*rand_visible.cast<double>();
  hideen_bias_term = parametar.hidden_bias.transpose()*rand_hidden.cast<double>();
  weight_bias_term = (parametar.weight.transpose()*rand_visible.cast<double>()).transpose()*rand_hidden.cast<double>();
  return - (visible_bias_term + hideen_bias_term + weight_bias_term);
}

double Model::cost_v(const Parametar& parametar, const Eigen::VectorXi& rand_visible) const{
  Eigen::VectorXd lambda, exp_lambda, buf;
  double result;

  lambda= lambda_hidden(parametar, rand_visible);
  exp_lambda = lambda.array().exp();
  buf = - (1 + exp_lambda.array()).log();
  result = buf.sum();
  result =  result - (parametar.visible_bias.transpose() * rand_visible.cast<double>());
  return result;
}

