#include<random>

#include"random.h"
using namespace rbm_utils;

Distribuition::Distribuition(double range_start, double range_end){
  // this->rd = gen(rd())
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(range_start, range_end); 
  dis(gen);
}

Distribuition::~Distribuition(){}


double Distribuition::random_num(){
}
