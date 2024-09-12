#ifndef PARAMETAR_H
#define PARAMETAR_H

#include"bias.h"
#include"weight.h"

namespace rbm{
  class Parametar {
    public:
      Bias visible_bias;
      Bias hidden_bias;
      Weight weight;
      Parametar(int visible_dim, int hidden_dim);
  };
}

#endif
