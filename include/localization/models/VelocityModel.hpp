#ifndef VELOCITY_MODEL_HPP
#define VELOCITY_MODEL_HPP

#include "BaseModel.hpp"

class VelocityModel : public BaseModel {
public:
  VelocityModel() {
    delta_.resize(2);
    Xk_.resize(3);
  }

  virtual Eigen::VectorXd computeModelFromPosition(const Eigen::VectorXd &state,
                                                   const Eigen::VectorXd &input,
                                                   double dt = 0.001) override {

    Eigen::VectorXd state_next = state;

    const double px = state(0);
    const double py = state(1);
    const double theta = state(2);

    const double v = input(0);
    const double omega = input(1);

    const double dtheta = omega * dt;
    const double theta_mid = theta + 0.5 * dtheta;

    state_next(0) = px + v * dt * std::cos(theta_mid);

    state_next(1) = py + v * dt * std::sin(theta_mid);

    state_next(2) = theta + dtheta;

    double angle = state_next(2);
    wrapAngle(angle);
    state_next(2) = angle;

    return state_next;
  }

  virtual Eigen::VectorXd computeAndUpdate(const Eigen::VectorXd &input,
                                           double dt = 0.001) override {
    if (!initialized_) {
      return Xk_;
    }

    update(input);

    const double v = delta_(0);
    const double omega = delta_(1);

    const double dtheta = omega * dt;
    const double theta_mid = Xk_(2) + 0.5 * dtheta;

    last_pose_ = Xk_;

    Xk_(0) = Xk_(0) + v * dt * std::cos(theta_mid);
    Xk_(1) = Xk_(1) + v * dt * std::sin(theta_mid);
    Xk_(2) = Xk_(2) + dtheta;

    double angle = Xk_(2);
    wrapAngle(angle);
    Xk_(2) = angle;

    return Xk_;
  }

  virtual Eigen::MatrixXd stateJacobian(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &input,
                                        double dt = 0.001) const override {
    double theta = state(2);
    double v = input(0);
    double omega = input(1);

    double theta_mid = theta + 0.5 * omega * dt;

    Eigen::Matrix3d F;

    F << 1.0, 0.0, -v * dt * std::sin(theta_mid), 0.0, 1.0,
        v * dt * std::cos(theta_mid), 0.0, 0.0, 1.0;

    return F;
  }

  virtual Eigen::MatrixXd inputJacobian(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &input,
                                        double dt = 0.001) const override {
    double theta = state(2);
    double v = input(0);
    double omega = input(1);

    double theta_mid = theta + 0.5 * omega * dt;

    Eigen::Matrix<double, 3, 2> G;

    G << dt * std::cos(theta_mid), -0.5 * v * dt * dt * std::sin(theta_mid),

        dt * std::sin(theta_mid), 0.5 * v * dt * dt * std::cos(theta_mid),

        0.0, dt;

    return G;
  }

protected:
  virtual void update(const Eigen::VectorXd &input) override {
    delta_(0) = input(0); // v [m/s]
    delta_(1) = input(1); // omega [rad/s]
  }
};

#endif