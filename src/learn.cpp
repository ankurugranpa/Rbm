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
  rbm::Model result_model(model.visible_dim, model.hidden_dim, model.parametar);
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

  rbm::Model result_model(model.visible_dim, model.hidden_dim, model.parametar);
  int batch_time = data_set_buf.size()/batch_size;
  if(data_set_buf.size()%batch_size>0){batch_time += 1;}
  Grad diff(model.visible_dim, model.hidden_dim);
  double grad;

  for(int e=0; e<epoch; e++){
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
      // for(const auto& data:batch_data){
      //   std::cout << data.transpose() << std::endl;
      // }
      // exit(1);


      // std::cout << "1" << std::endl;
      diff = calc_model_mean_cd(model, batch_data, cd_k_num);

      // diff = calc_data_mean(model, batch_data) - calc_model_mean_cd(model, batch_data, cd_k_num);
      // diff = calc_data_mean(result_model, batch_data) - calc_model_mean(result_model);
      grad = calc_grad(diff);

      // diff = calc_data_mean(result_model, data_set) - calc_model_mean(result_model);
      // grad = calc_grad(diff);
      // result_file << "step:" << step << " " << "grad" << grad << "\n";
      // std::cout << "step:" << step << " " << "grad" << grad << std::endl;

      result_file << "epoch:" << e << " "  <<"step:" << b <<  " " << "grad" << grad << "\n";
      std::cout << "epoch:" << e << " "  <<"step:" << b <<  " " << "grad" << grad << std::endl;

      file_engine.gen_file(result_model.parametar);

      if(is_zero(grad)){ return result_model; }
      
      result_model.parametar.visible_bias += nabla*diff.visible_grad;
      result_model.parametar.hidden_bias +=  nabla*diff.hidden_grad;
      result_model.parametar.weight +=  nabla*diff.weight_grad;
    }
    // std::cout << "エポック更新" << std::endl;
  }
  return result_model;
}

Grad Learn::calc_data_mean(const Model& model, const DataSet& data_set){
  Grad result_data_mean(model.visible_dim, model.hidden_dim);
  Eigen::VectorXd buf_sig;

  for(const auto& data: data_set){

    buf_sig = rbm_utils::sig(model.lambda_hidden(model.parametar, data));

    for(int j=0; j<model.hidden_dim; j++){
      // std::cout << "怪しい所4" << std::endl;
      result_data_mean.hidden_grad(j) += buf_sig(j);
      for(int i=0; i<model.visible_dim; i++){
        if(j==0){
          result_data_mean.visible_grad(i) += data(i);
        }
        result_data_mean.weight_grad(i, j) += data(i) * buf_sig(j);
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

  // visible
  for(auto mu=0u; mu<data_set.size(); mu++){
    for(auto i=0u; i<model.visible_dim; i++){
      result_model_mean.visible_grad(i) += (data_set[mu](i) - visible_t[mu](i));
    }
  }

  // hidden
  for(int mu=0; mu<data_num; mu++){
    Eigen::VectorXd data_mean = sig(model.lambda_hidden(model.parametar, data_set[mu]));
    Eigen::VectorXd model_mean = sig(model.lambda_hidden(model.parametar, visible_t[mu]));

    for(int j=0; j<model.hidden_dim; j++){
      result_model_mean.hidden_grad(j) += (data_mean(j) - model_mean(j) );
      for(auto i=0u; i<model.visible_dim; i++){
        result_model_mean.weight_grad(i, j) += ( data_set[mu](i)*data_mean(j) - visible_t[mu](i)*model_mean(j) );
      }
    }
  }

  
  

  // 周辺化バージョン
  // for(const auto& v_t: visible_t){
  //   for(auto i=0; i<v_t.size(); i++){
  //     result_model_mean.visible_grad(i) += v_t(i);
  //   }
  // }

  // for(const auto& v_t: visible_t){
  //   for(auto j=0; j<model.hidden_dim; j++){
  //     result_model_mean.hidden_grad(j) += model.lambda_hidden(model.parametar, v_t)(j);
  //   }
  // }

  // for(const auto& v_t: visible_t){
  //   for(auto i=0; i<model.visible_dim; i++){
  //     for(auto j=0; j<model.hidden_dim; j++){
  //       result_model_mean.weight_grad(i, j) += v_t(i)*model.lambda_hidden(model.parametar, v_t)(j);
  //     }
  //   }
  // }
  // for(auto i=0u; i<result_model_mean.visible_grad.size(); i++){
  //   if(result_model_mean.visible_grad(i) != 0){
  //     std::cout << "おかしい"<<std::endl;
  //     exit(1);
  //   };
  // }
  // for(auto j=0u; j<result_model_mean.hidden_grad.size(); j++){
  //   if(result_model_mean.hidden_grad(j) != 0){
  //     std::cout << "おかしい2"<<std::endl;
  //     exit(1);
  //   };
  //   for(auto i=0u; i<result_model_mean.visible_grad.size(); i++){
  //     if(result_model_mean.weight_grad(i, j) != 0){
  //     std::cout << "おかしい3"<<std::endl;
  //     exit(1);
  //     }
  //   }
  // }

  // for(const auto& v_t: visible_t){
  //   for(auto i=0; i<v_t.size(); i++){
  //     result_model_mean.visible_grad(i) += v_t(i);
  //   }
  // }

  // for(const auto& h_t: hidden_t) {
  //   for(auto j=0; j<h_t.size(); j++){
  //     result_model_mean.hidden_grad(j) += h_t(j);
  //   }
  // }
  // 
  // // weight
  // for(const auto& v_t: visible_t){
  //   for(const auto& h_t: hidden_t) {
  //     for(auto i=0; i<v_t.size(); i++){
  //       for(auto j=0; j<h_t.size(); j++){
  //         result_model_mean.weight_grad(i, j) += v_t(i)*h_t(j);
  //       }
  //     }
  //   }
  //}

  result_model_mean.visible_grad /= data_num;
  result_model_mean.hidden_grad /= data_num;
  result_model_mean.weight_grad /= data_num;
  // std::cout << "モデル平均の計算おわり" << std::endl;
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
