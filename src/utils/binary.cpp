#include"binary.h"
using namespace rbm_utils;

Binary::Binary(){};
Binary::~Binary(){};

Binary::Binary(int bit_num){
  this->bit_num = bit_num;
}

Data Binary::num2binary(int num){
  Data result(bit_num);

  boost::dynamic_bitset<> binary(bit_num, num);
  for(auto i=0u; i< binary.size(); i++){
    result(bit_num - 1 - i) = static_cast<int>(binary[i]);
  }
  return result;
}
