#ifndef IPOSITIONING_SYSTEM_H
#define IPOSITIONING_SYSTEM_H

#include "Landmark.hpp"
#include "models/BaseModel.hpp"

#include <eigen3/Eigen/Dense>
#include <memory>
#include <vector>

class IPositioningSystem {
public:
  explicit IPositioningSystem(const std::vector<Landmark> &landmarks)
      : model_(nullptr), landmarks_(landmarks) {}

  virtual ~IPositioningSystem() = default;

  virtual void execute(const std::vector<Landmark> &meas) = 0;

  virtual Eigen::VectorXd getState() const = 0;

  virtual void setInput(const Eigen::VectorXd &input, double dt) = 0;

  virtual void log(double x_pose, double y_pose, double th_pose) = 0;

protected:
  // The concrete filter template decides which BaseModel-derived class
  // is created, but the common interface owns it through BaseModel.
  std::shared_ptr<BaseModel> model_;

  std::vector<Landmark> landmarks_;
};

#endif
