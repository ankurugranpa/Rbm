#include"sigmoid.h"

double rbm_utils::sig(double x){
  double exp_buf;
  if (x>=0){
    return 1/(1+exp(-x));
  }
  else{
    exp_buf = exp(x);
    return exp_buf / (1+exp_buf);
  }
}


Bias rbm_utils::sig(Bias x){
  Bias sig_X(x.size());
  for(int i=0; i<x.size(); i++){
    sig_X(i) = sig(x(i));
  }
  return sig_X;
}

Eigen::VectorXd rbm_utils::sig(Eigen::VectorXd x){
  Eigen::VectorXd sig_x(x.size());
  
  for(int i=0; i<x.size(); i++){
    sig_x(i) = rbm_utils::sig(x(i));
  }
  return sig_x;
}
