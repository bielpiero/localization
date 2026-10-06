#ifndef IPOSITIONING_SYSTEM_H
#define IPOSITIONING_SYSTEM_H

#include "Landmark.hpp"
#include "models/BaseModel.hpp"
#include <eigen3/Eigen/Dense>
#include <memory>
#include <type_traits>
#include <vector>

class IPositioningSystem {
public:
  IPositioningSystem(const std::vector<Landmark> &landmark)
      : model_(nullptr), landmarks_(landmark.begin(), landmark.end()) {}
  virtual void execute(const std::vector<Landmark> &meas) = 0;
  virtual Eigen::VectorXd getState() const = 0;
  virtual void setIncrement(const Eigen::VectorXd &inc) = 0;

  virtual void log(double x_pose, double y_pose, double th_pose) = 0;

protected:
  std::shared_ptr<BaseModel> model_;
  std::vector<Landmark> landmarks_;
};
#endif