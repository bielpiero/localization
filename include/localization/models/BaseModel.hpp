#ifndef BASE_MODEL_HPP
#define BASE_MODEL_HPP

#include <eigen3/Eigen/Dense>

class BaseModel {
public:
  BaseModel() : initialized_(false) {}

  virtual ~BaseModel() = default;
  virtual Eigen::VectorXd
  computeModelFromPosition(const Eigen::VectorXd position,
                           const Eigen::VectorXd increment) = 0;
  virtual Eigen::VectorXd
  computeAndUpdate(const Eigen::VectorXd &increment) = 0;
  void init(const Eigen::VectorXd &inistialState) {
    Xk_ = inistialState;
    last_pose_ = inistialState;
    initialized_ = true;
  }

  size_t state_size() const { return Xk_.size(); }

  size_t delta_size() const { return delta_.size(); }

protected:
  virtual void update(const Eigen::VectorXd &pos) = 0;
  Eigen::VectorXd Xk_;
  Eigen::VectorXd last_pose_;
  Eigen::VectorXd delta_;
  bool initialized_;
};

#endif