#include<random>

#include<sigmoid.h>

#include"sampling.h"

using namespace rbm;

Sampling::Sampling(){}
Sampling::~Sampling(){}


std::tuple<DataSet, DataSet> Sampling::block_gibbs_sampling(const DataSet& observation_data, const rbm::Model& model_object, int sampling_rate){
  DataSet visible_gen_data_set; // 作成したデータの格納先
  DataSet hidden_gen_data_set; // 作成したデータの格納先
  Data visible_gen_data(model_object.visible_dim); // 作成したデータ
  Data hidden_gen_data(model_object.hidden_dim);  // 作成したデータ
  
  // Eigen::VectorXi hidden_gen_data(model_object.hidden_dim);

  Eigen::VectorXd P_v(model_object.visible_dim); // 確率P(h|v.theta)
  Eigen::VectorXd P_h(model_object.hidden_dim);  // 確率P(v|h.theta)



  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);



  for(int data_set_num=0; data_set_num<observation_data.size(); data_set_num++){
    // Data buf_data(observation_data[data_set_num]);
    visible_gen_data = observation_data[data_set_num];

    for(int time=0; time<sampling_rate+1; time++){
      //
      // V->Hのターム(準備)
      Eigen::VectorXd lambda_hidden; 
      Eigen::VectorXd P_h(model_object.hidden_dim); //P_v(ℎ| 𝒗(0), 𝜽)
                                                    
      // V(0)->H(0)のターム
      lambda_hidden = model_object.lambda_hidden(model_object.parametar, visible_gen_data); 
      P_h = rbm_utils::sig(lambda_hidden);
      for(int n=0; n<model_object.hidden_dim; n++){
        if(P_h(n) >= dis(gen)){
          hidden_gen_data(n) = 1;
        }else{
          hidden_gen_data(n) = 0;
        }
      }

      if(time==sampling_rate+1)break;
      

      // H->Vのターム準備
      Eigen::VectorXd lambda_visible; 
      Eigen::VectorXd P_v(model_object.visible_dim); //P_v(ℎ| 𝒗(0), 𝜽)
                                                     
      // H(0)->V(1)のターム
      lambda_visible = model_object.lambda_visible(model_object.parametar, hidden_gen_data);
      P_v = rbm_utils::sig(lambda_visible);
      for(int n=0; n<model_object.visible_dim; n++){
        if(P_v(n) >= dis(gen)){
          visible_gen_data(n) = 1;
        }else{
          visible_gen_data(n) = 0;
        }
      }
    }
    visible_gen_data_set.push_back(visible_gen_data);
    hidden_gen_data_set.push_back(hidden_gen_data);
  }
  return {visible_gen_data_set, hidden_gen_data_set};
}

DataSet Sampling::create_data_set(const rbm::Model& model_object,int data_num, int sampling_rate){
  // dataの初期値の作成
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  Eigen::VectorXi init_data(model_object.visible_dim);

  for(int i=0; i<model_object.visible_dim; i++){
    init_data(i) = dis(gen);
  }
  
  // sampling
  DataSet data_set;
  data_set.push_back(init_data);

  DataSet result;

  
  for(int i=0; i<data_num; i++){
    std::tie(data_set, std::ignore) = block_gibbs_sampling(data_set, model_object,  sampling_rate);
    result.push_back(data_set[0]);
  }
  return result;
}
