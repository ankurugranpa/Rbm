/**
* @file binary.h
* @brief binary ユーティリティ
* @author ankuru
* @date 2024/9/13
* @details binary ユーティリティ
*/
#ifndef BINARY_H
#define BINARY_H

#include<boost/dynamic_bitset.hpp>
#include<boost/math/special_functions/fpclassify.hpp>

#include<data.h>

using namespace rbm_types;

namespace rbm_utils{
  class Binary{
    public:
      //! データの次元
      int bit_num;
      Binary();
      /**
      *  @brief 
      *  @param[in] bit_num ビット数
      */
      Binary(int bit_num);
      ~Binary();

      /**
      *  @brief num2binary
      *  @param[in] num 変換する数字
      *  @return Data バイナリデータ
      *  @details 数字をバイナリデータに変換する関数
      */
      Data num2binary(int num);
  };
}

#endif
