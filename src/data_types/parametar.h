#ifndef PARAMETAR_H
#define PARAMETAR_H

#include"bias.h"
#include"weight.h"

namespace rbm{
  /*! @class Parametar
    @brief  Parametar型の定義
  */
  class Parametar {
    public:
      Bias visible_bias;
      Bias hidden_bias;
      Weight weight;
      Parametar(int visible_dim, int hidden_dim);
  };
}

#endif
