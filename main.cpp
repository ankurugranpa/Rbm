#include<iostream>
#include<tuple>

#include<rbm.h>
#include<sampling.h>
#include<data.h>



int main(void){
  rbm::Model calc_rbm(4, 2);

  std::cout << "Hello World" << std::endl;
  std::cout << calc_rbm.visible_dim << " " <<  calc_rbm.hidden_dim << std::endl;

  rbm::Sampling sampler;

  rbm::DataSet sample = sampler.create_data_set(calc_rbm, 5, 5);
  for(const auto& item: sample){
    std::cout<<item.transpose() << std::endl;
  }
  // TODO: ギブスサンプリングの成功かどうかの実装確認
}
