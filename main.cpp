// test2
#include<iostream>
#include<tuple>

#include<model.h>
#include<sampling.h>
#include<data.h>
using namespace rbm;



int main(void){
  Model calc_rbm(4, 2);
  Sampling sampler;
  // DataSet data_set = sampler.create_data_set(calc_rbm, 10, 1);
  DataSet data_set = sampler.create_data_set(calc_rbm, 100, 1);
  for(const auto& item: data_set){
    std::cout << item.transpose() << std::endl;
  }

  // std::cout << "Hello World" << std::endl;
  // std::cout << calc_rbm.visible_dim << " " <<  calc_rbm.hidden_dim << std::endl;

  // Sampling sampler;

  // DataSet sample = sampler.create_data_set(calc_rbm, 5, 5);
  // for(const auto& item: sample){
  //   std::cout<<item.transpose() << std::endl;
  // }
  // std::cout<<"####可視変数(4次元)####" << std::endl;

  // auto [data1, data2] = sampler.block_gibbs_sampling(sample, calc_rbm, 5);
  // for(const auto& item: data1){
  //   std::cout<<item.transpose() << std::endl;
  // }

  // std::cout<<"####隠れ変数(2次元)####" << std::endl;
  // for(const auto& item: data2){
  //   std::cout<<item.transpose() << std::endl;
  // }



}
