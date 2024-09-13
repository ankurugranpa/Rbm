#include<fstream>
#include<iostream>

#include<sigmoid.h>
#include<sampling.h>
#include<binary.h>

#include"learn.h"

using namespace rbm;

Learn::Learn(){};
Learn::Learn(std::string work_dir){
  this->work_dir = work_dir;
};

Learn::~Learn(){};

Model Learn::exact_calculation(const Model& before_learn_model,
                              const DataSet& data_set,
                              const int max_step,
                              const double nabla,
                              std::string result_file_name,
                              int* result_step){
  std::ofstream result_file(work_dir + "/" + result_file_name);
  
  // 更新していくモデル
  rbm::Model result_model(before_learn_model.visible_dim, before_learn_model.hidden_dim, before_learn_model.parametar);
  Grad diff(result_model.visible_dim, result_model.hidden_dim);
  double grad;

  // 学習
  for(int step=1; step<max_step+1; step++){
    // std::cout << "hidden " <<  diff.hidden_grad.transpose() << std::endl;
    diff = calc_data_mean(result_model, data_set) - calc_model_mean(result_model, data_set);
    grad = calc_grad(diff);
    result_file << "step:" << step << " " << "grad" << grad << "\n";

    
    if(is_zero(grad)){
      // ゼロであれば終了する
      return result_model;
    }
    // パラメーター更新
    result_model.parametar.visible_bias = result_model.parametar.visible_bias + nabla*diff.visible_grad;
    result_model.parametar.hidden_bias = result_model.parametar.hidden_bias + nabla*diff.hidden_grad;
    result_model.parametar.weight = result_model.parametar.weight + nabla*diff.weight_grad;
  }
  return result_model;
}


Grad Learn::calc_data_mean(const Model& model, const DataSet& data_set){
  Grad result_data_mean(model.visible_dim, model.hidden_dim);
  // for(int i=0; i<result_data_mean.visible_grad.size(); i++){
  //   result_data_mean.visible_grad(i) = 0;
  // }

  // for(int i=0; i<result_data_mean.hidden_grad.size(); i++){
  //   result_data_mean.hidden_grad(i) = 0;
  // }

  // for(int i=0; i<result_data_mean.visible_grad.size(); i++){
  //   for(int j=0; j<result_data_mean.hidden_grad.size(); j++){
  //     result_data_mean.weight_grad(i, j) = 0;
  //   }
  // }

  for(const auto& data: data_set){
    
    // dataをvに追加
    for(int i=0; i<model.visible_dim; i++){
      result_data_mean.visible_grad(i) = result_data_mean.visible_grad(i) + data(i);
    }
    for(int j=0; j<model.hidden_dim; j++){
      // std::cout << "B: " << rbm_utils::sig(model.lambda_hidden(model.parametar, data)).transpose() << std::endl;
      result_data_mean.hidden_grad(j) = result_data_mean.hidden_grad(j) + rbm_utils::sig(model.lambda_hidden(model.parametar, data))(j);
      // std::cout << "B: " << result_data_mean.hidden_grad(j) << std::endl;
      
    }
    

    for(int i=0; i<model.visible_dim; i++){
      for(int j=0; j<model.hidden_dim; j++){
        result_data_mean.weight_grad(i, j) = result_data_mean.weight_grad(i, j) + data(i) * rbm_utils::sig(model.lambda_hidden(model.parametar, data))(j);
      }
    }
  }
  // std::cout <<  "あいうえお" << result_data_mean.hidden_grad.transpose() << std::endl;

  result_data_mean.visible_grad /= data_set.size();
  result_data_mean.hidden_grad /= data_set.size();
  result_data_mean.weight_grad /= data_set.size();
  
  // std::cout <<  "hidden_mean: " << result_data_mean.weight_grad << std::endl;
  return result_data_mean;
}

void Learn::seg_epsilon(double value){
  this->epsilon = value;
}

Grad Learn::calc_model_mean(const Model& model, const DataSet& data_set){
  Grad result_model_mean(model.visible_dim, model.hidden_dim);
  rbm_utils::Binary visible_bit(model.visible_dim);
  rbm_utils::Binary hidden_bit(model.hidden_dim);

  // すべての状態の作成
  DataSet all_vissble; //可視変数のすべての状態
  DataSet all_hidden; // 隠れ変数のすべての状態
  double normal_Z=0; // 規格化定数
  double exp_term = 0;;

  for(int i=0; i<std::pow(model.visible_dim, 2); i++){
    all_vissble.push_back(visible_bit.num2binary(i));
  }

  for(int i=0; i<std::pow(model.hidden_dim, 2); i++){
    all_hidden.push_back(hidden_bit.num2binary(i));
  }

  // モデル平均の計算
  for(const auto& v_status: all_vissble){
    for(const auto& h_status: all_hidden){
      exp_term = std::exp(-model.cost_func(model.parametar, v_status, h_status));
      // std::cout << "exp " << exp_term << std::endl;
      normal_Z += exp_term;

      for(int i=0; i<model.visible_dim; i++){
        result_model_mean.visible_grad(i) += v_status(i)*exp_term;
      }

      for(int j=0; j<model.hidden_dim; j++){
        result_model_mean.hidden_grad(j) += h_status(j)*exp_term;
      }

      for(int i=0; i<model.visible_dim; i++){
        for(int j=0; j<model.hidden_dim; j++){
          result_model_mean.weight_grad(i, j) +=  v_status(i)*h_status(j)*exp_term;
        }
      }
    }
  }

  // 
  // for(const auto& v_status: all_vissble){
  //   for(const auto& h_status: all_hidden){
  //     exp_term = std::exp(-model.cost_func(model.parametar, v_status, h_status));
  //     std::cout << "exp " << exp_term << std::endl;
  //     normal_Z += exp_term;

  //     for(int i=0; i<model.visible_dim; i++){
  //       result_model_mean.visible_grad(i) = result_model_mean.visible_grad(i) + v_status(i)*exp_term;
  //     }

  //     for(int j=0; j<model.hidden_dim; j++){
  //       result_model_mean.hidden_grad(j) = result_model_mean.hidden_grad(j) + h_status(j)*exp_term;
  //     }

  //     for(int i=0; i<model.visible_dim; i++){
  //       for(int j=0; j<model.hidden_dim; j++){
  //         result_model_mean.weight_grad(i, j) = result_model_mean.weight_grad(i, j) + v_status(i)*h_status(j)*exp_term;
  //       }
  //     }
  //   }
  // }
  // std::cout << "model " << result_model_mean.visible_grad.transpose() << std::endl;
  result_model_mean.visible_grad /= normal_Z;
  result_model_mean.hidden_grad /= normal_Z;
  result_model_mean.weight_grad /= normal_Z;

  return result_model_mean;
}

double Learn::calc_grad(const Grad& grad_data){
  double numerator=0, denominator=0, grad = 0;
  numerator = grad_data.visible_grad.array().square().sum() + grad_data.hidden_grad.array().square().sum() + grad_data.weight_grad.array().square().sum();
  denominator = grad_data.visible_grad.size() + grad_data.hidden_grad.size() + grad_data.weight_grad.size();
  grad = std::sqrt(numerator) / denominator;
  return grad;
}

bool Learn::is_zero(double value){
  return std::abs(value) < this->epsilon;
}
