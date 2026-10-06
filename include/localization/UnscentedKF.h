#ifndef UNSCENTED_KF_H
#define UNSCENTED_KF_H

#include "IPositioningSystem.h"
#include "Landmark.hpp"

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <ros/ros.h>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

template <class ModelT> class UnscentedKF : public IPositioningSystem {
  static_assert(std::is_base_of<BaseModel, ModelT>::value &&
                    !std::is_same<BaseModel, ModelT>::value,
                "ModelT must be a class derived from BaseModel");

public:
  UnscentedKF(const std::vector<Landmark> &arucoMarkers,
              const Eigen::VectorXd &initial_pose,
              const std::string &instance_name = "");

  ~UnscentedKF() override;

  void execute(const std::vector<Landmark> &meas) override;

  Eigen::VectorXd getState() const override { return xk_; }

  void setInput(const Eigen::VectorXd &input, double dt) override;

  void log(double x, double y, double th) override;

  Eigen::MatrixXd getCovariance() const { return Pk_; }

  void setProcessNoise(const Eigen::MatrixXd &Q);
  void setMeasurementNoise(const Eigen::Matrix2d &R);
  void setMahalanobisThreshold(double threshold);

private:
  void prediction();
  void update(const std::vector<Landmark> &meas);

  // Required when two visual updates occur without a prediction in between.
  // It rebuilds sigma points around the current posterior (xk_, Pk_).
  bool regenerateStateSigmaPoints();

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
  bool sigma_points_valid_;
  double dt_;

  double alpha_;
  double beta_;
  double kappa_;
  double lambda_;
  double c_;
  double mahalanobis_thresh_;

  int nx_;
  int nv_;
  int na_;
  int nsig_;

  std::vector<double> wm_;
  std::vector<double> wc_;

  Eigen::VectorXd x_aug_;
  Eigen::MatrixXd p_aug_;
  std::vector<Eigen::VectorXd> xsig_aug_;
  std::vector<Eigen::VectorXd> xsig_pred_;

  std::string instance_name_;

  std::ofstream xy_log_;
  std::ofstream cov_log_;
  std::ofstream nis_log_;
};

#include "UnscentedKF.tpp"

#endif
