#include<random>

#include<data.h>
#include<binary.h>

#include"model.h"

using namespace rbm;


Model::Model():  parametar() {}

Model::Model(int visible_dim, int hidden_dim): 
  visible_dim(visible_dim), hidden_dim(hidden_dim), parametar(visible_dim, hidden_dim)
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

Model::Model(Parametar parametar): parametar()
{
  this->visible_dim = parametar.visible_bias.size();
  this->hidden_dim = parametar.hidden_bias.size();

  this->parametar.visible_bias = parametar.visible_bias;
  this->parametar.hidden_bias = parametar.hidden_bias;
  this->parametar.weight = parametar.weight;
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
  weight_bias_term = (rand_visible.transpose().cast<double>() * parametar.weight * rand_hidden.cast<double>());
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


void Model::set_parameter(Parametar parametar){
  this->parametar.visible_bias = parametar.visible_bias;
  this->parametar.hidden_bias = parametar.hidden_bias;
  this->parametar.weight = parametar.weight;
}

double Model::kl_divergence(const Model& model){
  double result=0;
  std::vector<double> q_d, p_v;
  DataSet status = all_status();
  
  for(const auto& st:status){
    q_d.push_back(std::exp(-model.cost_v(model.parametar, st)));
    p_v.push_back(std::exp(-cost_v(parametar, st)));
  }

  double q_d_z = std::reduce(std::begin(q_d), std::end(q_d));
  double p_v_z = std::reduce(std::begin(p_v), std::end(p_v));

  for(auto& item: q_d){
    item = item/q_d_z;
  }
  for(auto& item: p_v){
    item = item/p_v_z;
  }

  // for(int i=0; i<q_d.size(); i++){
  for(auto i = 0u; i < q_d.size(); i++){
    if (q_d[i] > 0 && p_v[i] > 0) {  // ゼロ割りを防ぐ
        result += q_d[i] * std::log(q_d[i]/p_v[i]);
    }
  }
  return result;
}

DataSet Model::all_status(){
  DataSet st;
  rbm_utils::Binary binaryer(visible_dim+hidden_dim);
  for(int i=0; i<std::pow(2, visible_dim+hidden_dim); i++){
    st.push_back(binaryer.num2binary(i));
  }
  return st;
}
