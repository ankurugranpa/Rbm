// grad.h
#ifndef GRAD_H
#define GRAD_H

#include <Eigen/Dense>

namespace rbm_types {
  class Grad {
  public:
    Eigen::VectorXd visible_grad;
    Eigen::VectorXd hidden_grad;
    Eigen::MatrixXd weight_grad;

    Grad(int visible_dim, int hidden_dim);

    Grad operator+(const Grad& other) const;
    Grad operator-(const Grad& other) const;
    Grad operator*(const Grad& other) const;
    Grad operator/(const Grad& other) const;

    Grad& operator+=(const Grad& other);
    Grad& operator-=(const Grad& other);
    Grad& operator*=(const Grad& other);
    Grad& operator/=(const Grad& other);
  };
}

#endif // GRAD_H
