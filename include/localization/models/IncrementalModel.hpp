#ifndef INCREMENTAL_MODEL_HPP
#define INCREMENTAL_MODEL_HPP

#include "BaseModel.hpp"
class IncrementalModel : public BaseModel {
public:
  IncrementalModel() {
    delta_.resize(2);
    Xk_.resize(3);
  }

  virtual Eigen::VectorXd
  computeModelFromPosition(const Eigen::VectorXd state,
                           const Eigen::VectorXd input) override {

    Eigen::VectorXd state_next = state;
    double px = state(0);
    double py = state(1);
    double theta = state(2);
    // COntrol de incrementos
    double ds = input(0);
    double dtheta = input(1);

    // Predict new state
    state_next(0) = px + ds * std::cos(theta + dtheta * 0.5);
    state_next(1) = py + ds * std::sin(theta + dtheta * 0.5);
    state_next(2) = theta + dtheta;

    while (state_next(2) > M_PI) {
      state_next(2) -= 2.0 * M_PI;
    }
    while (state_next(2) < -M_PI) {
      state_next(2) += 2.0 * M_PI;
    }

    return state_next;
  }

  virtual Eigen::VectorXd
  computeAndUpdate(const Eigen::VectorXd &input) override {
    if (initialized_) {
      update(input);
    }

    double ds = delta_(0);
    double dtheta = delta_(1);

    // Use midpoint heading for integration
    double theta_mid = Xk_(2) + dtheta * 0.5;

    // Predict new state
    double x_new = Xk_(0) + ds * std::cos(theta_mid);
    double y_new = Xk_(1) + ds * std::sin(theta_mid);
    double theta_new = Xk_(2) + dtheta;

    // Normalize heading
    wrapAngle(theta_new);

    last_pose_ = Xk_;

    Xk_(0) = x_new;
    Xk_(1) = y_new;
    Xk_(2) = theta_new;

    return Xk_;
  }

private:
  virtual void update(const Eigen::VectorXd &pos) override {

    // Compute translation delta
    double ds = pos(0);
    double dtheta = pos(1);

    // fixing angle
    while (dtheta > M_PI) {
      dtheta -= 2.0 * M_PI;
    }
    while (dtheta < -M_PI) {
      dtheta += 2.0 * M_PI;
    }

    delta_(0) = ds;
    delta_(1) = dtheta;
  }

  virtual Eigen::MatrixXd stateJacobian() const {}
  virtual Eigen::MatrixXd inputJacobian() const {}
};

#endif