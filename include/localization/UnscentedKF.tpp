#include <cmath>
#include <iomanip>
#include <stdexcept>

template <class ModelT>
UnscentedKF<ModelT>::UnscentedKF(
    const std::vector<Landmark> &arucoMarkers,
    const Eigen::VectorXd &initial_pose,
    const std::string &instance_name)
    : IPositioningSystem(arucoMarkers),
      has_new_input_(false),
      sigma_points_valid_(false),
      dt_(0.0),
      alpha_(1e-3),
      beta_(2.0),
      kappa_(0.0),
      mahalanobis_thresh_(5.991) {

  model_ = std::make_shared<ModelT>();
  model_->init(initial_pose);

  nx_ = static_cast<int>(model_->state_size());
  nv_ = static_cast<int>(model_->delta_size());

  if (initial_pose.size() != nx_) {
    ROS_ERROR("UnscentedKF: initial pose has size %ld but model state size is %d",
              initial_pose.size(), nx_);
    throw std::runtime_error("Invalid initial pose dimension");
  }

  if (nx_ < 3 || nv_ < 2) {
    throw std::runtime_error(
        "UnscentedKF requires a 3-state pose model and a 2-element input");
  }

  xk_ = initial_pose;
  uk_ = Eigen::VectorXd::Zero(nv_);

  // Initial state covariance.
  Pk_ = Eigen::MatrixXd::Zero(nx_, nx_);
  Pk_(0, 0) = 0.025;
  Pk_(1, 1) = 0.025;
  Pk_(2, 2) = 0.01;

  // Process covariance in the MODEL INPUT space.
  // NOTE: the final values must be calibrated separately for incremental
  // and velocity inputs because their units are different.
  Qk_ = Eigen::MatrixXd::Zero(nv_, nv_);
  Qk_(0, 0) = 0.0025;
  Qk_(1, 1) = 0.005;

  // Measurement covariance [range, bearing].
  Rk_.setZero();
  Rk_(0, 0) = 0.5;
  Rk_(1, 1) = 0.1;

  na_ = nx_ + nv_;
  nsig_ = 2 * na_ + 1;

  lambda_ = alpha_ * alpha_ * (na_ + kappa_) - na_;
  c_ = na_ + lambda_;

  wm_.assign(nsig_, 1.0 / (2.0 * c_));
  wc_.assign(nsig_, 1.0 / (2.0 * c_));

  wm_[0] = lambda_ / c_;
  wc_[0] = wm_[0] + (1.0 - alpha_ * alpha_ + beta_);

  x_aug_ = Eigen::VectorXd::Zero(na_);
  p_aug_ = Eigen::MatrixXd::Zero(na_, na_);

  xsig_aug_.assign(nsig_, Eigen::VectorXd::Zero(na_));
  xsig_pred_.assign(nsig_, Eigen::VectorXd::Zero(nx_));

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
  } else {
    ROS_WARN("UnscentedKF: could not open covariance log %s",
             cov_path.c_str());
  }

  xy_log_.open(xy_path, std::ios::out);
  if (xy_log_) {
    xy_log_ << "time,x_raw,y_raw,theta_raw,x_filter,y_filter,theta_filter\n";
  } else {
    ROS_WARN("UnscentedKF: could not open pose log %s", xy_path.c_str());
  }

  nis_log_.open(nis_path, std::ios::out);
  if (nis_log_) {
    nis_log_ << "time,landmark_id,nis,accepted\n";
  } else {
    ROS_WARN("UnscentedKF: could not open NIS log %s", nis_path.c_str());
  }

  ROS_INFO("UnscentedKF[%s] initialized with pose [%.3f, %.3f, %.3f]",
           instance_name_.c_str(), xk_(0), xk_(1), xk_(2));
}

template <class ModelT>
UnscentedKF<ModelT>::~UnscentedKF() {
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
void UnscentedKF<ModelT>::execute(const std::vector<Landmark> &meas) {
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
void UnscentedKF<ModelT>::setInput(
    const Eigen::VectorXd &input,
    double dt) {

  if (input.size() != uk_.size()) {
    ROS_ERROR("UnscentedKF::setInput: input size %ld, expected %ld",
              input.size(), uk_.size());
    return;
  }

  if (!std::isfinite(dt) || dt <= 0.0) {
    ROS_ERROR("UnscentedKF::setInput: invalid dt = %.9f", dt);
    return;
  }

  uk_ = input;
  dt_ = dt;
  has_new_input_ = true;
}

template <class ModelT>
void UnscentedKF<ModelT>::prediction() {
  x_aug_.setZero();
  x_aug_.head(nx_) = xk_;

  p_aug_.setZero();
  p_aug_.topLeftCorner(nx_, nx_) = Pk_;

  // Process noise must ALWAYS be present. The old implementation only
  // inserted Q when both inputs were positive, which removed process noise
  // for straight motion, reverse motion and negative turns.
  p_aug_.bottomRightCorner(nv_, nv_) = Qk_;

  p_aug_ = 0.5 * (p_aug_ + p_aug_.transpose());
  p_aug_ += 1e-9 * Eigen::MatrixXd::Identity(na_, na_);

  Eigen::LLT<Eigen::MatrixXd> llt;
  llt.compute(p_aug_);

  if (llt.info() != Eigen::Success) {
    p_aug_ += 1e-6 * Eigen::MatrixXd::Identity(na_, na_);
    llt.compute(p_aug_);
  }

  if (llt.info() != Eigen::Success) {
    ROS_ERROR("UnscentedKF: LLT failed in prediction()");
    sigma_points_valid_ = false;
    return;
  }

  Eigen::MatrixXd L = llt.matrixL();
  L *= std::sqrt(c_);

  xsig_aug_[0] = x_aug_;

  for (int i = 0; i < na_; ++i) {
    xsig_aug_[i + 1] = x_aug_ + L.col(i);
    xsig_aug_[i + 1 + na_] = x_aug_ - L.col(i);
  }

  for (int i = 0; i < nsig_; ++i) {
    const Eigen::VectorXd x_state =
        xsig_aug_[i].head(nx_);

    const Eigen::VectorXd input_noise =
        xsig_aug_[i].tail(nv_);

    const Eigen::VectorXd x_pred =
        model_->computeModelFromPosition(
            x_state,
            uk_ + input_noise,
            dt_);

    if (x_pred.size() != nx_ || !x_pred.allFinite()) {
      ROS_ERROR("UnscentedKF: invalid predicted sigma point");
      sigma_points_valid_ = false;
      return;
    }

    xsig_pred_[i] = x_pred;
  }

  xk_.setZero();

  double theta_s = 0.0;
  double theta_c = 0.0;

  for (int i = 0; i < nsig_; ++i) {
    xk_(0) += wm_[i] * xsig_pred_[i](0);
    xk_(1) += wm_[i] * xsig_pred_[i](1);

    theta_s += wm_[i] * std::sin(xsig_pred_[i](2));
    theta_c += wm_[i] * std::cos(xsig_pred_[i](2));
  }

  xk_(2) = std::atan2(theta_s, theta_c);

  Pk_.setZero();

  for (int i = 0; i < nsig_; ++i) {
    Eigen::VectorXd diff = xsig_pred_[i] - xk_;
    diff(2) = wrapAngle(diff(2));

    Pk_ += wc_[i] * diff * diff.transpose();
  }

  Pk_ = 0.5 * (Pk_ + Pk_.transpose());
  Pk_ += 1e-12 * Eigen::MatrixXd::Identity(nx_, nx_);

  uk_.setZero();

  // xsig_pred_ now represents the current predicted distribution.
  sigma_points_valid_ = true;
}

template <class ModelT>
bool UnscentedKF<ModelT>::regenerateStateSigmaPoints() {
  x_aug_.setZero();
  x_aug_.head(nx_) = xk_;

  p_aug_.setZero();
  p_aug_.topLeftCorner(nx_, nx_) = Pk_;

  // Keeping Q here is harmless for the state sigma points: the noise-only
  // sigma points have the same state component. It also keeps the augmented
  // covariance positive definite with the same UT dimensionality/weights.
  p_aug_.bottomRightCorner(nv_, nv_) = Qk_;

  p_aug_ = 0.5 * (p_aug_ + p_aug_.transpose());
  p_aug_ += 1e-9 * Eigen::MatrixXd::Identity(na_, na_);

  Eigen::LLT<Eigen::MatrixXd> llt;
  llt.compute(p_aug_);

  if (llt.info() != Eigen::Success) {
    p_aug_ += 1e-6 * Eigen::MatrixXd::Identity(na_, na_);
    llt.compute(p_aug_);
  }

  if (llt.info() != Eigen::Success) {
    ROS_ERROR("UnscentedKF: LLT failed while regenerating state sigma points");
    sigma_points_valid_ = false;
    return false;
  }

  Eigen::MatrixXd L = llt.matrixL();
  L *= std::sqrt(c_);

  xsig_aug_[0] = x_aug_;

  for (int i = 0; i < na_; ++i) {
    xsig_aug_[i + 1] = x_aug_ + L.col(i);
    xsig_aug_[i + 1 + na_] = x_aug_ - L.col(i);
  }

  // No motion propagation here. We only need sigma points that represent
  // the current posterior state distribution for the measurement transform.
  for (int i = 0; i < nsig_; ++i) {
    xsig_pred_[i] = xsig_aug_[i].head(nx_);
  }

  sigma_points_valid_ = true;
  return true;
}

template <class ModelT>
void UnscentedKF<ModelT>::update(
    const std::vector<Landmark> &meas) {

  if (meas.empty()) {
    return;
  }

  // If a visual correction already changed (xk_, Pk_) and no prediction
  // occurred afterwards, the previous sigma points are stale.
  if (!sigma_points_valid_ && !regenerateStateSigmaPoints()) {
    return;
  }

  std::vector<Eigen::Vector2d> accepted_z_meas;
  std::vector<Eigen::Vector2d> accepted_z_pred;
  std::vector<Eigen::MatrixXd> accepted_Zsig;
  std::vector<Eigen::Vector2d> accepted_nu;

  accepted_z_meas.reserve(meas.size());
  accepted_z_pred.reserve(meas.size());
  accepted_Zsig.reserve(meas.size());
  accepted_nu.reserve(meas.size());

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

    Eigen::MatrixXd Zsig_j =
        Eigen::MatrixXd::Zero(2, nsig_);

    for (int i = 0; i < nsig_; ++i) {
      const Eigen::VectorXd &sigma = xsig_pred_[i];

      const double dx = lm_x - sigma(0);
      const double dy = lm_y - sigma(1);

      Zsig_j(0, i) = std::sqrt(dx * dx + dy * dy);
      Zsig_j(1, i) =
          wrapAngle(std::atan2(dy, dx) - sigma(2));
    }

    Eigen::Vector2d z_pred_j =
        Eigen::Vector2d::Zero();

    double bearing_s = 0.0;
    double bearing_c = 0.0;

    for (int i = 0; i < nsig_; ++i) {
      z_pred_j(0) += wm_[i] * Zsig_j(0, i);

      bearing_s +=
          wm_[i] * std::sin(Zsig_j(1, i));

      bearing_c +=
          wm_[i] * std::cos(Zsig_j(1, i));
    }

    z_pred_j(1) =
        std::atan2(bearing_s, bearing_c);

    Eigen::Vector2d z_meas_j;
    z_meas_j << measurement.range(),
                wrapAngle(measurement.bearing());

    if (!z_meas_j.allFinite()) {
      continue;
    }

    Eigen::Vector2d nu_j =
        z_meas_j - z_pred_j;

    nu_j(1) = wrapAngle(nu_j(1));

    Eigen::Matrix2d S_j =
        Eigen::Matrix2d::Zero();

    for (int i = 0; i < nsig_; ++i) {
      Eigen::Vector2d dz =
          Zsig_j.col(i) - z_pred_j;

      dz(1) = wrapAngle(dz(1));

      S_j += wc_[i] * dz * dz.transpose();
    }

    S_j += Rk_;
    S_j = 0.5 * (S_j + S_j.transpose());

    Eigen::LDLT<Eigen::Matrix2d> ldlt(S_j);

    if (ldlt.info() != Eigen::Success) {
      continue;
    }

    const Eigen::Vector2d solved =
        ldlt.solve(nu_j);

    if (!solved.allFinite()) {
      continue;
    }

    const double nis =
        nu_j.dot(solved);

    const bool accepted =
        std::isfinite(nis) &&
        nis <= mahalanobis_thresh_;

    logNIS(measurement.id(), nis, accepted);

    if (!accepted) {
      continue;
    }

    accepted_z_meas.push_back(z_meas_j);
    accepted_z_pred.push_back(z_pred_j);
    accepted_Zsig.push_back(Zsig_j);
    accepted_nu.push_back(nu_j);
  }

  const int n_accepted =
      static_cast<int>(accepted_z_meas.size());

  if (n_accepted == 0) {
    return;
  }

  Eigen::VectorXd z_pred =
      Eigen::VectorXd::Zero(2 * n_accepted);

  Eigen::MatrixXd Zsig =
      Eigen::MatrixXd::Zero(2 * n_accepted, nsig_);

  Eigen::MatrixXd R =
      Eigen::MatrixXd::Zero(
          2 * n_accepted,
          2 * n_accepted);

  Eigen::VectorXd nu =
      Eigen::VectorXd::Zero(2 * n_accepted);

  for (int a = 0; a < n_accepted; ++a) {
    z_pred.segment<2>(2 * a) =
        accepted_z_pred[a];

    Zsig.block(2 * a, 0, 2, nsig_) =
        accepted_Zsig[a];

    R.block<2, 2>(2 * a, 2 * a) =
        Rk_;

    nu.segment<2>(2 * a) =
        accepted_nu[a];
  }

  Eigen::MatrixXd S =
      Eigen::MatrixXd::Zero(
          2 * n_accepted,
          2 * n_accepted);

  Eigen::MatrixXd Pxz =
      Eigen::MatrixXd::Zero(
          nx_,
          2 * n_accepted);

  for (int i = 0; i < nsig_; ++i) {
    Eigen::VectorXd dz =
        Zsig.col(i) - z_pred;

    for (int a = 0; a < n_accepted; ++a) {
      dz(2 * a + 1) =
          wrapAngle(dz(2 * a + 1));
    }

    Eigen::VectorXd dx =
        xsig_pred_[i] - xk_;

    dx(2) = wrapAngle(dx(2));

    S += wc_[i] * dz * dz.transpose();
    Pxz += wc_[i] * dx * dz.transpose();
  }

  S += R;
  S = 0.5 * (S + S.transpose());

  Eigen::LDLT<Eigen::MatrixXd> ldlt(S);

  if (ldlt.info() != Eigen::Success) {
    ROS_WARN("UnscentedKF: joint innovation covariance decomposition failed");
    return;
  }

  // K = Pxz S^-1 without explicitly forming S^-1.
  Eigen::MatrixXd K =
      ldlt.solve(Pxz.transpose()).transpose();

  if (!K.allFinite()) {
    ROS_WARN("UnscentedKF: non-finite Kalman gain");
    return;
  }

  xk_ += K * nu;
  xk_(2) = wrapAngle(xk_(2));

  Pk_ -= K * S * K.transpose();

  Pk_ = 0.5 * (Pk_ + Pk_.transpose());
  Pk_ += 1e-12 * Eigen::MatrixXd::Identity(nx_, nx_);

  // xk_ and Pk_ changed. The sigma points used above are now stale.
  sigma_points_valid_ = false;
}

template <class ModelT>
void UnscentedKF<ModelT>::setProcessNoise(
    const Eigen::MatrixXd &Q) {

  if (Q.rows() != Qk_.rows() ||
      Q.cols() != Qk_.cols()) {

    ROS_ERROR(
        "UnscentedKF::setProcessNoise: expected %ld x %ld, got %ld x %ld",
        Qk_.rows(), Qk_.cols(), Q.rows(), Q.cols());

    return;
  }

  Qk_ = 0.5 * (Q + Q.transpose());
  sigma_points_valid_ = false;
}

template <class ModelT>
void UnscentedKF<ModelT>::setMeasurementNoise(
    const Eigen::Matrix2d &R) {

  Rk_ = 0.5 * (R + R.transpose());
}

template <class ModelT>
void UnscentedKF<ModelT>::setMahalanobisThreshold(
    double threshold) {

  if (std::isfinite(threshold) &&
      threshold > 0.0) {

    mahalanobis_thresh_ = threshold;
  }
}

template <class ModelT>
void UnscentedKF<ModelT>::logCovariance() {
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
void UnscentedKF<ModelT>::logNIS(
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
void UnscentedKF<ModelT>::log(
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
double UnscentedKF<ModelT>::wrapAngle(
    double angle) {

  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }

  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }

  return angle;
}
