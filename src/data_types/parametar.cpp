#include"parametar.h"
using namespace rbm_types;

Parametar::Parametar(){}

Parametar::Parametar(int visible_dim, int hidden_dim):
  visible_bias(visible_dim), hidden_bias(hidden_dim), weight(visible_dim, hidden_dim)
{}
