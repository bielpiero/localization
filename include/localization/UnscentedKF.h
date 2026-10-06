#ifndef RN_UKF_TASK_H
#define RN_UKF_TASK_H

#include "IPositioningSystem.h"
#include "Landmark.hpp"
#include <cmath> // std::sqrt, std::atan2, M_PI
#include <eigen3/Eigen/Dense>
#include <limits> // std::numeric_limits
#include <list>
#include <ros/ros.h>
#include <unordered_map>
#include <vector>

#include <cmath>
#include <fstream>
#include <iomanip> // para std::setprecision

template <class ModelT> class UnscentedKF : public IPositioningSystem {
  static_assert(std::is_base_of<BaseModel, ModelT>::value &&
                    !std::is_same<BaseModel, ModelT>::value,
                "ModelT must be a class derived from BaseModel");

public:
  UnscentedKF(const std::vector<Landmark> &arucoMarkers,
              const Eigen::VectorXd &initial_pose);
  virtual ~UnscentedKF();

private:
  virtual void execute(const std::vector<Landmark> &meas) override;
  void prediction();
  void update(const std::vector<Landmark> &meas);

  virtual Eigen::VectorXd getState() const override { return xk_; }

  virtual void log(double x, double y, double th) override;

  Eigen::MatrixXd getCovariance() const { return Pk_; }

  virtual void setIncrement(const Eigen::VectorXd &inc) override;

private:
  Eigen::VectorXd xk_;   // current position
  Eigen::VectorXd xk_1_; // previous position

  Eigen::VectorXd uk_; // input actual [d_lin, d_ang]
  bool has_new_increment_;

  // variaces and covariances matrices MAtrices dinamicas, (Xd)
  Eigen::MatrixXd Pk_;
  Eigen::MatrixXd Qk_;
  Eigen::MatrixXd Rk_;

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

  // augmented matrices
  Eigen::VectorXd x_aug_;
  Eigen::MatrixXd p_aug_;
  std::vector<Eigen::VectorXd> xsig_aug_;
  std::vector<Eigen::VectorXd> xsig_pred_;

  // --- Logging of pose and UKF ---
  std::ofstream xy_log_;

  // --- Logging of covariance ---
  std::ofstream cov_log_;
  void logCovariance();
};

#endif