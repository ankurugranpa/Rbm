#include<fstream>
#include<iostream>
#include<random>

#include<sigmoid.h>
#include<sampling.h>
#include<binary.h>

#include"learn.h"

using namespace rbm;
using namespace rbm_utils;

Learn::Learn(){};
Learn::Learn(std::string work_dir){
  this->work_dir = work_dir;
};

Learn::~Learn(){};

Model Learn::exact_calculation(const Model& model,
                              const DataSet& data_set,
                              const int max_step,
                              const double nabla,
                              std::string result_file_name,
                              int* result_step){
  std::ofstream result_file(work_dir + "/" + result_file_name);
  
  // 更新していくモデル
  rbm::Model result_model(model.parametar);
  Grad diff(result_model.visible_dim, result_model.hidden_dim);
  double grad;

  // 学習
  for(int step=1; step<max_step+1; step++){
    // *result_step = step;
    
    // std::cout << "hidden " <<  diff.hidden_grad.transpose() << std::endl;
    diff = calc_data_mean(result_model, data_set) - calc_model_mean(result_model);

    // diff = calc_model_mean_cd(model, data_set, 1);
    // diff = calc_data_mean(model, data_set) - calc_model_mean_cd(model, data_set, 1);
    grad = calc_grad(diff);
    result_file << "step:" << step << " " << "grad" << grad << "\n";
    std::cout << "step:" << step << " " << "grad" << grad << std::endl;

    
    // ゼロであれば終了する
    if(is_zero(grad)){ return result_model; }
      
    // パラメーター更新
    result_model.parametar.visible_bias += nabla*diff.visible_grad;
    result_model.parametar.hidden_bias +=  nabla*diff.hidden_grad;
    result_model.parametar.weight +=  nabla*diff.weight_grad;
  }
  return result_model;
}

Model Learn::contrastive_divergence(const Model& model,
                                    const DataSet& data_set,
                                    const double nabla,
                                    const int epoch,
                                    const int batch_size,
                                    std::string result_file_name,
                                    int cd_k_num,
                                    int* result_epoch,
                                    int* result_step){
  std::ofstream result_file(work_dir + "/" + result_file_name);

  File file_engine(work_dir);
  DataSet data_set_buf = data_set;

  rbm::Model result_model(model.parametar);
  int batch_time = data_set_buf.size()/batch_size;
  if(data_set_buf.size()%batch_size>0){batch_time += 1;}
  Grad diff(model.visible_dim, model.hidden_dim);
  double grad;


  // 初期パラメータの保存
  Parametar buf = model.parametar;

  for(int e=0; e<epoch; e++){
    // 学習に使用するパラメータの保存
    file_engine.gen_file(result_model.parametar);

    std::cout << "シャッフル" << std::endl;
    *result_epoch = e+1;
    // 観測データのシャッフル
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(data_set_buf.begin(), data_set_buf.end(), gen);

    for(int b=0; b<batch_time; b++){
      *result_step = b+1;
      
      // 観測データからバッチサイズ分のデータの取り出し
      int iterator_buf = (b)*batch_size;
      DataSet::const_iterator iterator_start;
      DataSet::const_iterator iterator_end;

      if(b == batch_time-1){
        iterator_start = data_set_buf.begin() + iterator_buf;
        iterator_end = data_set_buf.end();
      }else{
        iterator_start = data_set_buf.begin() + iterator_buf;
        iterator_end = data_set_buf.begin() + (iterator_buf+batch_size);
      }
      DataSet batch_data(iterator_start, iterator_end);
      // diff = calc_model_mean_cd(model, batch_data, cd_k_num);
      // std::cout << "ここ" << std::endl;

      diff = calc_data_mean(result_model, batch_data) - calc_model_mean_cd(result_model, batch_data, cd_k_num);
      // std::cout << "あ" << std::endl;
      // diff = calc_data_mean(result_model, batch_data) - calc_model_mean(result_model);
      grad = calc_grad(diff);

      // diff = calc_data_mean(result_model, data_set) - calc_model_mean(result_model);
      // grad = calc_grad(diff);
      // result_file << "step:" << step << " " << "grad" << grad << "\n";
      // std::cout << "step:" << step << " " << "grad" << grad << std::endl;

      result_file << "epoch:" << e+1 << " "  <<"step:" << b+1 <<  " " << "grad" << grad << "\n";
      std::cout << "epoch:" << e+1 << " "  <<"step:" << b+1 <<  " " << "grad" << grad << std::endl;


      if(is_zero(grad)){ return result_model; }
      
    }
    // std::cout << "エポック更新" << std::endl;
    result_model.parametar.visible_bias += nabla*diff.visible_grad;
    result_model.parametar.hidden_bias +=  nabla*diff.hidden_grad;
    result_model.parametar.weight +=  nabla*diff.weight_grad;
  }
  return result_model;
}

Grad Learn::calc_data_mean(const Model& model, const DataSet& data_set){
  Grad result_data_mean(model.visible_dim, model.hidden_dim);
  Eigen::VectorXd buf_sig;

  for(auto mu=0u; mu<data_set.size(); mu++){
    buf_sig = rbm_utils::sig(model.lambda_hidden(model.parametar, data_set[mu]));
    
    // visible
    for(int i=0; i<model.visible_dim; i++){
      result_data_mean.visible_grad(i) += data_set[mu](i);
    }

    for(int  j=0; j<model.hidden_dim; j++){
       result_data_mean.hidden_grad(j) += buf_sig(j); 
    }
    
    // hidden
    for(int  j=0; j<model.hidden_dim; j++){
      for(int i=0; i<model.visible_dim; i++){
        result_data_mean.weight_grad(i, j) += data_set[mu](i)*buf_sig(j);
      }
    }
    
  }

  result_data_mean.visible_grad /= data_set.size();
  result_data_mean.hidden_grad /= data_set.size();
  result_data_mean.weight_grad /= data_set.size();
  // std::cout << "データ平均の計算おわり" << std::endl;
  
  return result_data_mean;
}

void Learn::set_epsilon(double value){
  this->epsilon = value;
}

Grad Learn::calc_model_mean(const Model& model){
  Grad result_model_mean(model.visible_dim, model.hidden_dim);
  rbm_utils::Binary visible_bit(model.visible_dim);
  rbm_utils::Binary hidden_bit(model.hidden_dim);

  // すべての状態の作成
  DataSet all_vissble; //可視変数のすべての状態
  DataSet all_hidden; // 隠れ変数のすべての状態

  for(int i=0; i<std::pow(2, model.visible_dim); i++){
    all_vissble.push_back(visible_bit.num2binary(i));
  }

  for(int j=0; j<std::pow(2, model.hidden_dim); j++){
    all_hidden.push_back(hidden_bit.num2binary(j));
  }

  double normal_Z=0; // 規格化定数
  double exp_term = 0;

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

  result_model_mean.visible_grad /= normal_Z;
  result_model_mean.hidden_grad /= normal_Z;
  result_model_mean.weight_grad /= normal_Z;

  return result_model_mean;
}


Grad Learn::calc_model_mean_cd(const Model& model, const DataSet& data_set, int sampling_rate){
  int data_num = data_set.size();
  Grad result_model_mean(model.visible_dim, model.hidden_dim);

  Sampling sampler;
  auto [visible_t, hidden_t] = sampler.block_gibbs_sampling(data_set, model, sampling_rate);

  for(const auto& v_t: visible_t){
    for(auto i=0; i<v_t.size(); i++){
      result_model_mean.visible_grad(i) += v_t(i);
    }
  }

  for(const auto& h_t: hidden_t) {
    for(auto j=0; j<h_t.size(); j++){
      result_model_mean.hidden_grad(j) += h_t(j);
    }
  }
  
  for(int mu=0; mu<data_num; mu++){
       for(auto i=0; i<model.visible_dim; i++){
         for(auto j=0; j<model.hidden_dim; j++){
           result_model_mean.weight_grad(i, j) += visible_t[mu](i)*hidden_t[mu](j);
         }
       }
  }

  result_model_mean.visible_grad /= data_num;
  result_model_mean.hidden_grad /= data_num;
  result_model_mean.weight_grad /= data_num;
  return result_model_mean;
}

double Learn::calc_grad(const Grad& grad_data){
  double numerator=0, denominator=0, grad = 0;
  
  // std::cout << "visible" << std::endl;
  // std::cout<< grad_data.visible_grad.transpose() << std::endl;
  // std::cout << "hidden" << std::endl;
  // std::cout << grad_data.hidden_grad.transpose() << std::endl;
  // std::cout << "weight" << std::endl;
  // std::cout << grad_data.weight_grad << std::endl;
  // std::cout<< grad_data.visible_grad.size() << " "<< grad_data.hidden_grad.size() << " " << grad_data.weight_grad.size() << std::endl;
  numerator = grad_data.visible_grad.array().square().sum() + grad_data.hidden_grad.array().square().sum() + grad_data.weight_grad.array().square().sum();
  denominator = grad_data.visible_grad.size() + grad_data.hidden_grad.size() + grad_data.weight_grad.size();
  // std::cout << (std::sqrt(numerator)) << std::endl;
  grad = (std::sqrt(numerator) / denominator);
  return grad;
}

bool Learn::is_zero(double value){
  return std::abs(value) < this->epsilon;
}
