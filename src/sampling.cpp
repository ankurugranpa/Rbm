#include<random>
#include<cmath>
#include<iostream>

#include<sigmoid.h>

#include"sampling.h"

using namespace rbm;

Sampling::Sampling(){}
Sampling::~Sampling(){}


std::tuple<DataSet, DataSet> Sampling::block_gibbs_sampling(const DataSet& data_set, const rbm::Model& model, int sampling_rate){
  DataSet visible_gen_data_set; // 作成したデータの格納先
  DataSet hidden_gen_data_set; // 作成したデータの格納先
  Data visible_gen_data(model.visible_dim); // 作成したデータ
  Data hidden_gen_data(model.hidden_dim);  // 作成したデータ
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);



  Eigen::VectorXd P_v(model.visible_dim); // 確率P(h|v.theta)
  Eigen::VectorXd P_h(model.hidden_dim);  // 確率P(v|h.theta)
  Eigen::VectorXd lambda_hidden; 
  Eigen::VectorXd lambda_visible; 

  for(auto data_num=0u; data_num<data_set.size(); data_num++){
    // 初期値
    visible_gen_data = data_set[data_num];

    for(int time=0; time<sampling_rate+1; time++){
                                                    
      // V(0)->H(0)のターム
      lambda_hidden = model.lambda_hidden(model.parametar, visible_gen_data); 
      P_h = rbm_utils::sig(lambda_hidden);
      for(int n=0; n<model.hidden_dim; n++){
        if(P_h(n) >= dis(gen)){
          hidden_gen_data(n) = 1;
        }else{
          hidden_gen_data(n) = 0;
        }
      }
      
      // hidden_gen_data_set.push_back(hidden_gen_data);

      // V(T)を保存する
      if(time == sampling_rate){
        hidden_gen_data_set.push_back(hidden_gen_data);
      }

      // if(time!=sampling_rate){
        // H->Vのターム準備
                                                       
        // H(0)->V(1)のターム
        lambda_visible = model.lambda_visible(model.parametar, hidden_gen_data);
        P_v = rbm_utils::sig(lambda_visible);
        for(int n=0; n<model.visible_dim; n++){
          if(P_v(n) >= dis(gen)){
            visible_gen_data(n) = 1;
          }else{
            visible_gen_data(n) = 0;
          }
        }
      // }
      // V(T)を保存する(-tはV(T)の方が1周期早く取得できるから
      if(time == sampling_rate -1){
        visible_gen_data_set.push_back(visible_gen_data);
      }
    }
  }
  // std::cout << "visible_size" << visible_gen_data_set.size() << std::endl;
  // std::cout << "visible_size" << visible_gen_data_set[0].size() << std::endl;

  // std::cout << "hidden_size"  << hidden_gen_data_set.size() << std::endl;
  // std::cout << "hidden_size"  << hidden_gen_data_set[0].size() << std::endl;
  // exit(1);
  return {visible_gen_data_set, hidden_gen_data_set};
}

DataSet Sampling::create_data_set(const rbm::Model& model,int data_num, int sampling_rate){
  // dataの初期値の作成
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  Eigen::VectorXi init_data(model.visible_dim);

  for(int i=0; i<model.visible_dim; i++){
    init_data(i) = dis(gen);
  }
  
  // sampling
  DataSet data_set;
  data_set.push_back(init_data);

  DataSet result;

  
  for(int i=0; i<data_num; i++){
    std::tie(data_set, std::ignore ) = block_gibbs_sampling(data_set, model,  sampling_rate);
    result.push_back(data_set[0]);
  }
  return result;
}

Data Sampling::create_data(const rbm::Model& model, Data base_data, int sampling_rate){
  // dataの初期値の作成
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  Eigen::VectorXi init_data(model.visible_dim);

  
  // sampling
  DataSet data_set;
  data_set.push_back(base_data);

  DataSet result;
  
  std::tie(data_set, std::ignore)= block_gibbs_sampling(data_set, model,  sampling_rate);
  return data_set[0];
}
