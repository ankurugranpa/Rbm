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


rbm::Bias rbm_utils::sig(rbm::Bias X){
  rbm::Bias sig_X(X.size());
  for(int i=0; i<X.size(); i++){
    sig_X(i) = sig(X(i));
  }
  return sig_X;
}
