#ifndef INCREMENTAL_MODEL_HPP
#define INCREMENTAL_MODEL_HPP

#include "BaseModel.hpp"
class IncrementalModel : public BaseModel {
public:
  IncrementalModel() {
    delta_.resize(2);
    Xk_.resize(3);
  }

  virtual Eigen::VectorXd computeModelFromPosition(const Eigen::VectorXd &state,
                                                   const Eigen::VectorXd &input,
                                                   double dt = 0.001) override {

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

    wrapAngle(state_next(2));

    return state_next;
  }

  virtual Eigen::VectorXd computeAndUpdate(const Eigen::VectorXd &input,
                                           double dt = 0.001) override {
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

  virtual Eigen::MatrixXd stateJacobian(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &input,
                                        double dt = 0.001) const override {
    double theta = state(2);
    double ds = input(0);
    double dtheta = input(1);

    double theta_mid = theta + 0.5 * dtheta;

    Eigen::Matrix3d F;

    F << 1.0, 0.0, -ds * std::sin(theta_mid), 0.0, 1.0,
        ds * std::cos(theta_mid), 0.0, 0.0, 1.0;

    return F;
  }

  virtual Eigen::MatrixXd inputJacobian(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &input,
                                        double dt = 0.001) const override {
    double theta = state(2);
    double ds = input(0);
    double dtheta = input(1);

    double theta_mid = theta + 0.5 * dtheta;

    Eigen::Matrix<double, 3, 2> G;

    G << std::cos(theta_mid), -0.5 * ds * std::sin(theta_mid),

        std::sin(theta_mid), 0.5 * ds * std::cos(theta_mid),

        0.0, 1.0;

    return G;
  }

private:
  virtual void update(const Eigen::VectorXd &input) override {

    // Compute translation delta
    double ds = input(0);
    double dtheta = input(1);

    // fixing angle
    wrapAngle(dtheta);

    delta_(0) = ds;
    delta_(1) = dtheta;
  }
};

#endif