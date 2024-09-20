#ifndef PARAMETAR_H
#define PARAMETAR_H

#include"bias.h"
#include"weight.h"
using namespace rbm_types;

namespace rbm_types{
  /*! @class Parametar
    @brief  Parametar型の定義
  */
  class Parametar {
    public:
      Bias visible_bias;
      Bias hidden_bias;
      Weight weight;
      Parametar(int visible_dim, int hidden_dim);
      Parametar();
  };
}

#endif
