#include<iostream>

#include<rbm.h>



int main(void){
  rbm::Model calc_rbm(2, 2);

  std::cout << "Hello World" << std::endl;
  std::cout << calc_rbm.visible_dim << " " <<  calc_rbm.hidden_dim << std::endl;
}
