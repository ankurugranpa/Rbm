#include<cmath>

#include "softplus.h"

double rbm_utils::softplus(double x){
  if (x>=0){
    return x + std::log(exp(-x) + 1);
  }else{
    return std::log(1 + exp(x));
  }
}
