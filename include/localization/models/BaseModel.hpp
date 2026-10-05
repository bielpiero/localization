#ifndef BASE_MODEL_HPP
#define BASE_MODEL_HPP

#include <eigen3/Eigen/Dense>

class BaseModel {
public:
  BaseModel() : initialized_(false) {}

  virtual ~BaseModel() = default;
  virtual Eigen::VectorXd
  computeModelFromPosition(const Eigen::VectorXd position,
                           const Eigen::VectorXd input) = 0;
  virtual Eigen::VectorXd computeAndUpdate(const Eigen::VectorXd &input) = 0;
  void init(const Eigen::VectorXd &inistialState) {
    Xk_ = inistialState;
    last_pose_ = inistialState;
    initialized_ = true;
  }

  size_t state_size() const { return Xk_.size(); }

  size_t delta_size() const { return delta_.size(); }

  virtual Eigen::MatrixXd stateJacobian() const = 0;
  virtual Eigen::MatrixXd inputJacobian() const = 0;

protected:
  virtual void update(const Eigen::VectorXd &pos) = 0;

  virtual void wrapAngle(double &angle) {
    while (angle > M_PI) {
      angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
      angle += 2.0 * M_PI;
    }
  }

  Eigen::VectorXd Xk_;
  Eigen::VectorXd last_pose_;
  Eigen::VectorXd delta_;
  bool initialized_;
};

#endif