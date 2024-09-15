// grad.cpp
#include "grad.h"
#include <stdexcept>

using namespace rbm_types;

Grad::Grad(int visible_dim, int hidden_dim)
  : visible_grad(Eigen::VectorXd::Zero(visible_dim)),
    hidden_grad(Eigen::VectorXd::Zero(hidden_dim)),
    weight_grad(Eigen::MatrixXd::Zero(visible_dim, hidden_dim))
{}

Grad Grad::operator+(const Grad& other) const {
    Grad result(*this);
    return result += other;
}

Grad Grad::operator-(const Grad& other) const {
    Grad result(*this);
    return result -= other;
}

Grad Grad::operator*(const Grad& other) const {
    Grad result(*this);
    return result *= other;
}

Grad Grad::operator/(const Grad& other) const {
    Grad result(*this);
    return result /= other;
}

Grad& Grad::operator+=(const Grad& other) {
    visible_grad += other.visible_grad;
    hidden_grad += other.hidden_grad;
    weight_grad += other.weight_grad;
    return *this;
}

Grad& Grad::operator-=(const Grad& other) {
    visible_grad -= other.visible_grad;
    hidden_grad -= other.hidden_grad;
    weight_grad -= other.weight_grad;
    return *this;
}

Grad& Grad::operator*=(const Grad& other) {
    visible_grad = visible_grad.cwiseProduct(other.visible_grad);
    hidden_grad = hidden_grad.cwiseProduct(other.hidden_grad);
    weight_grad = weight_grad.cwiseProduct(other.weight_grad);
    return *this;
}

Grad& Grad::operator/=(const Grad& other) {
    if (other.visible_grad.cwiseEqual(Eigen::VectorXd::Zero(other.visible_grad.size())).any() ||
        other.hidden_grad.cwiseEqual(Eigen::VectorXd::Zero(other.hidden_grad.size())).any() ||
        other.weight_grad.cwiseEqual(Eigen::MatrixXd::Zero(other.weight_grad.rows(), other.weight_grad.cols())).any()) {
        throw std::runtime_error("Division by zero");
    }

    visible_grad = visible_grad.cwiseQuotient(other.visible_grad);
    hidden_grad = hidden_grad.cwiseQuotient(other.hidden_grad);
    weight_grad = weight_grad.cwiseQuotient(other.weight_grad);
    return *this;
}
// #include"grad.h"
// using namespace rbm_types;
// 
// Grad::Grad(int visible_dim, int hidden_dim):
//   visible_grad(visible_dim), hidden_grad(hidden_dim), weight_grad(visible_dim, hidden_dim)
// {
//   for(int i=0; i<visible_dim; i++){
//     this->visible_grad(i) = 0;
//   }
// 
//   for(int i=0; i<hidden_dim; i++){
//     this->hidden_grad(i) = 0;
//   }
// 
//   for(int i=0; i<visible_dim; i++){
//     for(int j=0; j<hidden_dim; j++){
//       this->weight_grad(i, j) = 0;
//     }
//   }
// 
// }
// 
// Grad Grad::operator+(const Grad& other) const {
//     Grad result(*this); // 現在のオブジェクトのコピーを作成
//     result.visible_grad += other.visible_grad;
//     result.hidden_grad += other.hidden_grad;
//     result.weight_grad += other.weight_grad;
//     return result;
// }
// 
// Grad Grad::operator-(const Grad& other) const {
//     Grad result(*this); // 現在のオブジェクトのコピーを作成
//     result.visible_grad -= other.visible_grad;
//     result.hidden_grad -= other.hidden_grad;
//     result.weight_grad -= other.weight_grad;
//     return result;
// }
// 
// Grad Grad::operator*(const Grad& other) const {
//     Grad result(*this); // 現在のオブジェクトのコピーを作成
//     result.visible_grad = visible_grad.cwiseProduct(other.visible_grad);
//     result.hidden_grad = hidden_grad.cwiseProduct(other.hidden_grad);
//     result.weight_grad = weight_grad.cwiseProduct(other.weight_grad);
//     return result;
// }
// 
// Grad Grad::operator/(const Grad& other) const {
//     Grad result(*this);
// 
//     if (other.visible_grad.cwiseEqual(Eigen::VectorXd::Zero(other.visible_grad.size())).any() ||
//         other.hidden_grad.cwiseEqual(Eigen::VectorXd::Zero(other.hidden_grad.size())).any() ||
//         other.weight_grad.cwiseEqual(Eigen::MatrixXd::Zero(other.weight_grad.rows(), other.weight_grad.cols())).any()) {
//         throw std::runtime_error("Division by zero");
//     }
// 
//     result.visible_grad = visible_grad.cwiseQuotient(other.visible_grad);
//     result.hidden_grad = hidden_grad.cwiseQuotient(other.hidden_grad);
//     result.weight_grad = weight_grad.cwiseQuotient(other.weight_grad);
//     return result;
// }
