#ifndef EXTENDED_KF_H
#define EXTENDED_KF_H

#include "IPositioningSystem.h"
#include "Landmark.hpp"

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <memory>
#include <ros/ros.h>
#include <vector>

class ExtendedKF : public IPositioningSystem {
public:
  ExtendedKF(std::shared_ptr<BaseModel> model,
             const std::vector<Landmark> &arucoMarkers,
             const Eigen::VectorXd &initial_pose);

  ~ExtendedKF() override;

  void execute(const std::vector<Landmark> &meas) override;

  Eigen::VectorXd getState() const override { return xk_; }

  void log(double x_pose, double y_pose, double th_pose) override;

  // Requires the corresponding change in IPositioningSystem:
  // virtual void setInput(const Eigen::VectorXd&, double dt) = 0;
  void setInput(const Eigen::VectorXd &input, double dt) override;

  Eigen::MatrixXd getCovariance() const { return Pk_; }

  void setProcessNoise(const Eigen::MatrixXd &Q);
  void setMeasurementNoise(const Eigen::Matrix2d &R);
  void setMahalanobisThreshold(double threshold);

private:
  void prediction();
  void update(const std::vector<Landmark> &meas);

  void logCovariance();
  void logNIS(int landmark_id, double nis, bool accepted);

  static double wrapAngle(double angle);

private:
  Eigen::VectorXd xk_;
  Eigen::VectorXd uk_;

  Eigen::MatrixXd Pk_;
  Eigen::MatrixXd Qk_;
  Eigen::Matrix2d Rk_;

  bool has_new_input_;
  double dt_;

  double mahalanobis_thresh_;

  std::ofstream xy_log_;
  std::ofstream cov_log_;
  std::ofstream nis_log_;
};

#endif
