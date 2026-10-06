#include <cmath>
#include <iomanip>
#include <stdexcept>

template <class ModelT>
ExtendedKF<ModelT>::ExtendedKF(
    const std::vector<Landmark> &arucoMarkers,
    const Eigen::VectorXd &initial_pose,
    const std::string &instance_name)
    : IPositioningSystem(arucoMarkers),
      has_new_input_(false),
      dt_(0.0),
      mahalanobis_thresh_(5.991) {

  model_ = std::make_shared<ModelT>();
  model_->init(initial_pose);

  const int nx =
      static_cast<int>(model_->state_size());

  const int nu =
      static_cast<int>(model_->delta_size());

  if (initial_pose.size() != nx) {
    ROS_ERROR("ExtendedKF: initial pose has size %ld but model state size is %d",
              initial_pose.size(), nx);
    throw std::runtime_error("Invalid initial pose dimension");
  }

  if (nx < 3 || nu < 2) {
    throw std::runtime_error(
        "ExtendedKF requires a 3-state pose model and a 2-element input");
  }

  xk_ = initial_pose;
  uk_ = Eigen::VectorXd::Zero(nu);

  Pk_ = Eigen::MatrixXd::Zero(nx, nx);
  Pk_(0, 0) = 0.025;
  Pk_(1, 1) = 0.025;
  Pk_(2, 2) = 0.01;

  Qk_ = Eigen::MatrixXd::Zero(nu, nu);
  Qk_(0, 0) = 0.0025;
  Qk_(1, 1) = 0.005;

  Rk_.setZero();
  Rk_(0, 0) = 0.5;
  Rk_(1, 1) = 0.1;

  instance_name_ =
      instance_name.empty() ? std::string(typeid(ModelT).name()) : instance_name;

  const std::string cov_path =
      "/home/sara/cov_" + instance_name_ + ".csv";

  const std::string xy_path =
      "/home/sara/pose_" + instance_name_ + ".csv";

  const std::string nis_path =
      "/home/sara/nis_" + instance_name_ + ".csv";

  cov_log_.open(cov_path, std::ios::out);
  if (cov_log_) {
    cov_log_ << "time,P_xx,P_xy,P_xtheta,P_yy,P_ytheta,P_thetatheta\n";
  }

  xy_log_.open(xy_path, std::ios::out);
  if (xy_log_) {
    xy_log_ << "time,x_raw,y_raw,theta_raw,x_filter,y_filter,theta_filter\n";
  }

  nis_log_.open(nis_path, std::ios::out);
  if (nis_log_) {
    nis_log_ << "time,landmark_id,nis,accepted\n";
  }

  ROS_INFO("ExtendedKF[%s] initialized with pose [%.3f, %.3f, %.3f]",
           instance_name_.c_str(), xk_(0), xk_(1), xk_(2));
}

template <class ModelT>
ExtendedKF<ModelT>::~ExtendedKF() {
  if (xy_log_.is_open()) {
    xy_log_.close();
  }

  if (cov_log_.is_open()) {
    cov_log_.close();
  }

  if (nis_log_.is_open()) {
    nis_log_.close();
  }
}

template <class ModelT>
void ExtendedKF<ModelT>::execute(
    const std::vector<Landmark> &meas) {

  if (has_new_input_) {
    prediction();
    has_new_input_ = false;
  }

  if (!meas.empty()) {
    update(meas);
  }

  logCovariance();
}

template <class ModelT>
void ExtendedKF<ModelT>::setInput(
    const Eigen::VectorXd &input,
    double dt) {

  if (input.size() != uk_.size()) {
    ROS_ERROR("ExtendedKF::setInput: input size %ld, expected %ld",
              input.size(), uk_.size());
    return;
  }

  if (!std::isfinite(dt) || dt <= 0.0) {
    ROS_ERROR("ExtendedKF::setInput: invalid dt = %.9f", dt);
    return;
  }

  uk_ = input;
  dt_ = dt;
  has_new_input_ = true;
}

template <class ModelT>
void ExtendedKF<ModelT>::prediction() {
  if (!has_new_input_) {
    return;
  }

  const Eigen::VectorXd x_prior = xk_;
  const Eigen::MatrixXd P_prior = Pk_;

  const Eigen::MatrixXd F =
      model_->stateJacobian(
          x_prior,
          uk_,
          dt_);

  const Eigen::MatrixXd G =
      model_->inputJacobian(
          x_prior,
          uk_,
          dt_);

  if (F.rows() != Pk_.rows() ||
      F.cols() != Pk_.cols()) {

    ROS_ERROR("ExtendedKF::prediction: invalid F dimensions");
    return;
  }

  if (G.rows() != Pk_.rows() ||
      G.cols() != Qk_.rows()) {

    ROS_ERROR("ExtendedKF::prediction: invalid G/Q dimensions");
    return;
  }

  xk_ =
      model_->computeModelFromPosition(
          x_prior,
          uk_,
          dt_);

  if (!xk_.allFinite()) {
    ROS_ERROR("ExtendedKF::prediction: non-finite predicted state");
    xk_ = x_prior;
    return;
  }

  Pk_ =
      F * P_prior * F.transpose()
      + G * Qk_ * G.transpose();

  Pk_ = 0.5 * (Pk_ + Pk_.transpose());
  Pk_ += 1e-12 *
         Eigen::MatrixXd::Identity(Pk_.rows(), Pk_.cols());

  uk_.setZero();
}

template <class ModelT>
void ExtendedKF<ModelT>::update(
    const std::vector<Landmark> &meas) {

  const int nx =
      static_cast<int>(xk_.size());

  std::vector<Eigen::Vector2d> accepted_z;
  std::vector<Eigen::Vector2d> accepted_h;
  std::vector<Eigen::Matrix<double, 2, 3>> accepted_H;

  accepted_z.reserve(meas.size());
  accepted_h.reserve(meas.size());
  accepted_H.reserve(meas.size());

  for (const auto &measurement : meas) {
    double lm_x = 0.0;
    double lm_y = 0.0;
    bool found = false;

    for (const auto &lm : landmarks_) {
      if (lm.id() == measurement.id()) {
        lm_x = lm.x();
        lm_y = lm.y();
        found = true;
        break;
      }
    }

    if (!found) {
      continue;
    }

    const double dx =
        lm_x - xk_(0);

    const double dy =
        lm_y - xk_(1);

    const double q =
        dx * dx + dy * dy;

    if (q <= 1e-9) {
      continue;
    }

    const double predicted_range =
        std::sqrt(q);

    const double predicted_bearing =
        wrapAngle(
            std::atan2(dy, dx) - xk_(2));

    Eigen::Vector2d h;
    h << predicted_range,
         predicted_bearing;

    Eigen::Vector2d z;
    z << measurement.range(),
         wrapAngle(measurement.bearing());

    if (!z.allFinite()) {
      continue;
    }

    Eigen::Vector2d innovation =
        z - h;

    innovation(1) =
        wrapAngle(innovation(1));

    Eigen::Matrix<double, 2, 3> H;

    H << -dx / predicted_range,
         -dy / predicted_range,
          0.0,

          dy / q,
         -dx / q,
         -1.0;

    Eigen::Matrix2d S =
        H * Pk_ * H.transpose()
        + Rk_;

    S = 0.5 * (S + S.transpose());

    Eigen::LDLT<Eigen::Matrix2d> ldlt(S);

    if (ldlt.info() != Eigen::Success) {
      continue;
    }

    const Eigen::Vector2d solved =
        ldlt.solve(innovation);

    if (!solved.allFinite()) {
      continue;
    }

    const double nis =
        innovation.dot(solved);

    const bool accepted =
        std::isfinite(nis) &&
        nis <= mahalanobis_thresh_;

    logNIS(
        measurement.id(),
        nis,
        accepted);

    if (!accepted) {
      continue;
    }

    accepted_z.push_back(z);
    accepted_h.push_back(h);
    accepted_H.push_back(H);
  }

  const int n_accepted =
      static_cast<int>(accepted_z.size());

  if (n_accepted == 0) {
    return;
  }

  Eigen::VectorXd z =
      Eigen::VectorXd::Zero(2 * n_accepted);

  Eigen::VectorXd h =
      Eigen::VectorXd::Zero(2 * n_accepted);

  Eigen::MatrixXd H =
      Eigen::MatrixXd::Zero(
          2 * n_accepted,
          nx);

  Eigen::MatrixXd R =
      Eigen::MatrixXd::Zero(
          2 * n_accepted,
          2 * n_accepted);

  for (int i = 0; i < n_accepted; ++i) {
    z.segment<2>(2 * i) =
        accepted_z[i];

    h.segment<2>(2 * i) =
        accepted_h[i];

    H.block<2, 3>(2 * i, 0) =
        accepted_H[i];

    R.block<2, 2>(2 * i, 2 * i) =
        Rk_;
  }

  Eigen::VectorXd innovation =
      z - h;

  for (int i = 0; i < n_accepted; ++i) {
    innovation(2 * i + 1) =
        wrapAngle(
            innovation(2 * i + 1));
  }

  Eigen::MatrixXd S =
      H * Pk_ * H.transpose()
      + R;

  S = 0.5 * (S + S.transpose());

  Eigen::LDLT<Eigen::MatrixXd> ldlt(S);

  if (ldlt.info() != Eigen::Success) {
    ROS_WARN("ExtendedKF: joint innovation covariance decomposition failed");
    return;
  }

  Eigen::MatrixXd K =
      ldlt.solve(
          H * Pk_.transpose()).transpose();

  if (!K.allFinite()) {
    return;
  }

  const Eigen::MatrixXd P_prior =
      Pk_;

  xk_ +=
      K * innovation;

  xk_(2) =
      wrapAngle(xk_(2));

  const Eigen::MatrixXd I =
      Eigen::MatrixXd::Identity(nx, nx);

  const Eigen::MatrixXd I_KH =
      I - K * H;

  Pk_ =
      I_KH * P_prior * I_KH.transpose()
      + K * R * K.transpose();

  Pk_ = 0.5 * (Pk_ + Pk_.transpose());
  Pk_ += 1e-12 *
         Eigen::MatrixXd::Identity(nx, nx);
}

template <class ModelT>
void ExtendedKF<ModelT>::setProcessNoise(
    const Eigen::MatrixXd &Q) {

  if (Q.rows() != Qk_.rows() ||
      Q.cols() != Qk_.cols()) {

    ROS_ERROR(
        "ExtendedKF::setProcessNoise: expected %ld x %ld, got %ld x %ld",
        Qk_.rows(), Qk_.cols(), Q.rows(), Q.cols());

    return;
  }

  Qk_ =
      0.5 * (Q + Q.transpose());
}

template <class ModelT>
void ExtendedKF<ModelT>::setMeasurementNoise(
    const Eigen::Matrix2d &R) {

  Rk_ =
      0.5 * (R + R.transpose());
}

template <class ModelT>
void ExtendedKF<ModelT>::setMahalanobisThreshold(
    double threshold) {

  if (std::isfinite(threshold) &&
      threshold > 0.0) {

    mahalanobis_thresh_ =
        threshold;
  }
}

template <class ModelT>
void ExtendedKF<ModelT>::logCovariance() {
  if (!cov_log_.is_open()) {
    return;
  }

  const double t =
      ros::Time::now().toSec();

  cov_log_
      << std::fixed
      << std::setprecision(9)
      << t << ","
      << Pk_(0, 0) << ","
      << Pk_(0, 1) << ","
      << Pk_(0, 2) << ","
      << Pk_(1, 1) << ","
      << Pk_(1, 2) << ","
      << Pk_(2, 2) << "\n";
}

template <class ModelT>
void ExtendedKF<ModelT>::logNIS(
    int landmark_id,
    double nis,
    bool accepted) {

  if (!nis_log_.is_open()) {
    return;
  }

  const double t =
      ros::Time::now().toSec();

  nis_log_
      << std::fixed
      << std::setprecision(9)
      << t << ","
      << landmark_id << ","
      << nis << ","
      << (accepted ? 1 : 0)
      << "\n";
}

template <class ModelT>
void ExtendedKF<ModelT>::log(
    double x,
    double y,
    double theta) {

  if (!xy_log_.is_open()) {
    return;
  }

  const double t =
      ros::Time::now().toSec();

  xy_log_
      << std::fixed
      << std::setprecision(9)
      << t << ","
      << x << ","
      << y << ","
      << theta << ","
      << xk_(0) << ","
      << xk_(1) << ","
      << xk_(2) << "\n";
}

template <class ModelT>
double ExtendedKF<ModelT>::wrapAngle(
    double angle) {

  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }

  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }

  return angle;
}
