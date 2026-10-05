#include "localization/UnscentedKF.h"

void printMatrix(const std::string &name, const Eigen::MatrixXd &M) {
  std::cout << "\n=== " << name << " ===\n";
  std::cout << std::fixed << std::setprecision(6);

  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j < M.cols(); ++j) {
      std::cout << std::setw(12) << M(i, j) << " ";
    }
    std::cout << "\n";
  }
}

void printVector(const std::string &name, const Eigen::VectorXd &v) {
  std::cout << "\n=== " << name << " ===\n";
  std::cout << std::fixed << std::setprecision(6);

  for (int i = 0; i < v.size(); ++i) {
    std::cout << std::setw(12) << v(i) << "\n";
  }
}

void printDiagonal(const std::string &name, const Eigen::MatrixXd &M) {
  std::cout << "\n=== diag(" << name << ") ===\n";
  std::cout << std::fixed << std::setprecision(6);

  for (int i = 0; i < M.rows(); ++i) {
    std::cout << std::setw(12) << M(i, i) << "\n";
  }
}

void printEigenvalues(const std::string &name, const Eigen::MatrixXd &M) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(M);

  if (eig.info() == Eigen::Success) {
    std::cout << "\n=== eig(" << name << ") ===\n";
    std::cout << eig.eigenvalues().transpose() << "\n";
  } else {
    std::cout << "\n=== eig(" << name << ") FAILED ===\n";
  }
}

UnscentedKF::UnscentedKF(std::shared_ptr<BaseModel> model,
                         const std::vector<Landmark> &arucoMarkers,
                         const Eigen::VectorXd &initial_pose)
    : IPositioningSystem(model, arucoMarkers) {
  xk_ = Eigen::VectorXd::Zero(model->state_size());
  xk_1_ = Eigen::VectorXd::Zero(model->state_size());

  uk_ = Eigen::VectorXd::Zero(model->delta_size()); // debería ser 2
  has_new_increment_ = false;

  // Inicializar estado con x, y, theta
  xk_(0) = initial_pose(0); // x
  xk_(1) = initial_pose(1); // y
  xk_(2) = initial_pose(2); // theta (si no lo tienes, poner 0)

  nx_ = model->state_size();

  // Cov Matrix State
  Pk_ = Eigen::MatrixXd::Zero(nx_, nx_);
  Pk_(0, 0) = 0.025;
  Pk_(1, 1) = 0.025;
  Pk_(2, 2) = 0.01;

  // Cov Matrix Process
  Qk_ = Eigen::MatrixXd::Zero(model->delta_size(), model->delta_size());
  Qk_(0, 0) = 0.0025;
  Qk_(1, 1) = 0.005;

  nv_ = Qk_.rows();
  na_ = nx_ + nv_;
  nsig_ = 2 * na_ + 1;

  // Cov Matrix Observations
  Rk_ = Eigen::Matrix2d::Zero();
  Rk_(0, 0) = 0.5;
  Rk_(1, 1) = 0.1;

  ROS_INFO("UnscentedKF initialized with pose [%.2f, %.2f, %.2f]", xk_(0),
           xk_(1), xk_(2));

  // modify tuning parameters
  alpha_ = 1e-3;
  beta_ = 2.0;
  kappa_ = 0.0;
  lambda_ = alpha_ * alpha_ * (na_ + kappa_) - na_;
  c_ = na_ + lambda_;
  mahalanobis_thresh_ = 20.21; // 5.99 inicial

  wm_.assign(nsig_, 1.0 / (2.0 * c_));
  wc_.assign(nsig_, 1.0 / (2.0 * c_));
  wm_[0] = lambda_ / c_;
  wc_[0] = wm_[0] + (1 - alpha_ * alpha_ + beta_);

  x_aug_ = Eigen::VectorXd::Zero(na_);
  p_aug_ = Eigen::MatrixXd::Zero(na_, na_);
  xsig_aug_.assign(nsig_, Eigen::VectorXd(na_));
  xsig_pred_.assign(nsig_, Eigen::VectorXd(nx_));

  // --- Inicialización del log de covarianza ---
  const std::string log_path = "/home/sara/cov_ukf_log.csv";

  cov_log_.open(log_path, std::ios::out);
  if (!cov_log_) {
    ROS_WARN("No se pudo abrir el fichero de log de covarianza: %s",
             log_path.c_str());
  } else {
    // Si el fichero está vacío, escribe cabecera
    if (cov_log_.tellp() == 0) {
      cov_log_ << "time,sigma_x,sigma_y,sigma_theta\n";
    }
  }
  xy_log_.open("/home/sara/ukf_xy_log.csv");
  xy_log_ << "t,x_pose,y_pose,th_pose,x_ukf,y_ukf,th_ukf\n";
}
UnscentedKF::~UnscentedKF() {
  if (cov_log_.is_open()) {
    cov_log_.close();
  }
  if (xy_log_.is_open()) {
    xy_log_.close();
  }
}

void UnscentedKF::execute(const std::vector<Landmark> &meas) {

  if (has_new_increment_) {
    prediction();
    has_new_increment_ = false;
  }
  if (!meas.empty()) {
    update(meas);
  }
  logCovariance();
}

void UnscentedKF::prediction() {
  // ROS_INFO("nx_=%d, nv_=%d, na_=%d, nsig_=%d", nx_, nv_, na_, nsig_);

  x_aug_.head(nx_) = xk_;
  p_aug_.setZero();
  p_aug_.topLeftCorner(nx_, nx_) = Pk_;
  if (uk_(0) > 0.0 and uk_(1) > 0.0) {
    p_aug_.bottomRightCorner(nv_, nv_) = Qk_;
  }

  p_aug_ = 0.5 * (p_aug_ + p_aug_.transpose());
  p_aug_ += 1e-9 * Eigen::MatrixXd::Identity(na_, na_);

  Eigen::LLT<Eigen::MatrixXd> llt(p_aug_);
  if (llt.info() != Eigen::Success) {
    ROS_ERROR("LLT failed in prediction(), adding bigger epsilon and skipping "
              "this step.");
    p_aug_ += 1e-6 * Eigen::MatrixXd::Identity(na_, na_);
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
    Eigen::VectorXd x_state = xsig_aug_[i].head(nx_);
    Eigen::VectorXd noise = xsig_aug_[i].tail(nv_);
    Eigen::VectorXd x_pred =
        model_->computeModelFromPosition(x_state, uk_ + noise);

    if (x_pred.size() != nx_ || !x_pred.allFinite()) {
      ROS_ERROR("Bad predicted sigma point");
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

    double theta_i = xsig_pred_[i](2);
    theta_s += wm_[i] * std::sin(theta_i);
    theta_c += wm_[i] * std::cos(theta_i);
  }

  xk_(2) = std::atan2(theta_s, theta_c);

  Pk_.setZero();
  for (int i = 0; i < nsig_; ++i) {
    Eigen::VectorXd diff = xsig_pred_[i] - xk_;

    while (diff(2) > M_PI) {
      diff(2) -= 2 * M_PI;
    }
    while (diff(2) < -M_PI) {
      diff(2) += 2 * M_PI;
    }

    Pk_ += wc_[i] * diff * diff.transpose();
  }

  // Pk_ = 0.5 * (Pk_ + Pk_.transpose());
  // Pk_ += 1e-12 * Eigen::MatrixXd::Identity(nx_, nx);
  uk_.setZero();
}

void UnscentedKF::update(const std::vector<Landmark> &meas) {
  const int n_meas = static_cast<int>(meas.size());
  if (n_meas == 0)
    return;

  std::vector<int> accepted_idx;
  std::vector<Eigen::Vector2d> accepted_z_meas;
  std::vector<Eigen::Vector2d> accepted_z_pred;
  std::vector<Eigen::MatrixXd> accepted_Zsig;
  std::vector<Eigen::VectorXd> accepted_nu;

  for (int j = 0; j < n_meas; ++j) {
    double lm_x = 0.0;
    double lm_y = 0.0;
    bool found = false;

    for (const auto &lm : landmarks_) {
      if (lm.id() == meas[j].id()) {
        lm_x = lm.x();
        lm_y = lm.y();
        found = true;
        break;
      }
    }

    if (!found) {
      continue;
    }
    Eigen::MatrixXd Zsig_j = Eigen::MatrixXd::Zero(2, nsig_);

    for (int i = 0; i < nsig_; ++i) {
      const Eigen::VectorXd &sigma = xsig_pred_[i];

      const double px = sigma(0);
      const double py = sigma(1);
      const double th = sigma(2);

      const double dx = lm_x - px;
      const double dy = lm_y - py;

      const double range = std::sqrt(dx * dx + dy * dy);
      double bearing = std::atan2(dy, dx) - th;

      while (bearing > M_PI) {
        bearing -= 2.0 * M_PI;
      }
      while (bearing < -M_PI) {
        bearing += 2.0 * M_PI;
      }

      Zsig_j(0, i) = range;
      Zsig_j(1, i) = bearing;
    }

    Eigen::Vector2d z_pred_j = Eigen::Vector2d::Zero();

    z_pred_j(0) = 0.0;
    double s = 0.0;
    double c = 0.0;

    for (int i = 0; i < nsig_; ++i) {
      z_pred_j(0) += wm_[i] * Zsig_j(0, i);

      s += wm_[i] * std::sin(Zsig_j(1, i));
      c += wm_[i] * std::cos(Zsig_j(1, i));
    }

    z_pred_j(1) = std::atan2(s, c);

    Eigen::Vector2d z_meas_j;
    z_meas_j(0) = meas[j].range();
    z_meas_j(1) = meas[j].bearing();

    while (z_meas_j(1) > M_PI) {
      z_meas_j(1) -= 2.0 * M_PI;
    }
    while (z_meas_j(1) < -M_PI) {
      z_meas_j(1) += 2.0 * M_PI;
    }

    Eigen::Vector2d nu_j = z_meas_j - z_pred_j;
    while (nu_j(1) > M_PI) {
      nu_j(1) -= 2.0 * M_PI;
    }
    while (nu_j(1) < -M_PI) {
      nu_j(1) += 2.0 * M_PI;
    }

    Eigen::Matrix2d S_j = Eigen::Matrix2d::Zero();

    for (int i = 0; i < nsig_; ++i) {
      Eigen::Vector2d dz = Zsig_j.col(i) - z_pred_j;

      while (dz(1) > M_PI) {
        dz(1) -= 2.0 * M_PI;
      }
      while (dz(1) < -M_PI) {
        dz(1) += 2.0 * M_PI;
      }

      S_j += wc_[i] * dz * dz.transpose();
    }

    S_j += Rk_;

    Eigen::FullPivLU<Eigen::Matrix2d> lu_j(S_j);
    if (!lu_j.isInvertible()) {
      continue;
    }

    Eigen::Matrix2d S_j_inv = S_j.inverse();
    const double maha_j = nu_j.transpose() * S_j_inv * nu_j;

    if (maha_j > mahalanobis_thresh_) {
      continue;
    }

    accepted_idx.push_back(j);
    accepted_z_meas.push_back(z_meas_j);
    accepted_z_pred.push_back(z_pred_j);
    accepted_Zsig.push_back(Zsig_j);
    accepted_nu.push_back(nu_j);
  }

  const int am = static_cast<int>(accepted_idx.size());
  if (am == 0) {
    ROS_WARN("No accepted ArUco measurements");
    return;
  }

  Eigen::VectorXd z_meas = Eigen::VectorXd::Zero(2 * am);
  Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(2 * am);
  Eigen::MatrixXd Zsig = Eigen::MatrixXd::Zero(2 * am, nsig_);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(2 * am, 2 * am);
  Eigen::VectorXd nu = Eigen::VectorXd::Zero(2 * am);

  for (int a = 0; a < am; ++a) {
    z_meas.segment<2>(2 * a) = accepted_z_meas[a];
    z_pred.segment<2>(2 * a) = accepted_z_pred[a];
    Zsig.block(2 * a, 0, 2, nsig_) = accepted_Zsig[a];
    R.block<2, 2>(2 * a, 2 * a) = Rk_;
    nu.segment<2>(2 * a) = accepted_nu[a];
  }

  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(2 * am, 2 * am);
  Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(nx_, 2 * am);

  for (int i = 0; i < nsig_; ++i) {
    Eigen::VectorXd dz = Zsig.col(i) - z_pred;

    for (int a = 0; a < am; ++a) {
      const int angle_idx = 2 * a + 1;

      while (dz(angle_idx) > M_PI) {
        dz(angle_idx) -= 2.0 * M_PI;
      }
      while (dz(angle_idx) < -M_PI) {
        dz(angle_idx) += 2.0 * M_PI;
      }
    }

    Eigen::VectorXd dx = xsig_pred_[i] - xk_;

    while (dx(2) > M_PI) {
      dx(2) -= 2.0 * M_PI;
    }
    while (dx(2) < -M_PI) {
      dx(2) += 2.0 * M_PI;
    }

    S += wc_[i] * dz * dz.transpose();
    Pxz += wc_[i] * dx * dz.transpose();
  }

  S += R;

  Eigen::FullPivLU<Eigen::MatrixXd> lu(S);
  if (!lu.isInvertible()) {
    ROS_WARN("S is singular, skipping update");
    return;
  }

  Eigen::MatrixXd S_inv = S.inverse();
  Eigen::MatrixXd K = Pxz * S_inv;

  printMatrix("S", S);
  printMatrix("K", K);

  printDiagonal("S", S);
  printDiagonal("P before", Pk_);

  printEigenvalues("S", S);

  // 9) Actualización del estado
  xk_ += K * nu;
  while (xk_(2) > M_PI)
    xk_(2) -= 2.0 * M_PI;
  while (xk_(2) < -M_PI)
    xk_(2) += 2.0 * M_PI;

  Eigen::MatrixXd KSKt = K * S * K.transpose();
  printDiagonal("KSKt", KSKt);

  Pk_ -= KSKt;

  printDiagonal("P after", Pk_);
  printEigenvalues("P after", Pk_);
}

void UnscentedKF::setIncrement(const Eigen::VectorXd &inc) {
  uk_ = inc;
  has_new_increment_ = true;
}

void UnscentedKF::logCovariance() {
  if (!cov_log_.is_open()) {
    return;
  }

  double t = ros::Time::now().toSec();

  cov_log_ << std::fixed << std::setprecision(6) << t << "," << Pk_(0, 0) << ","
           << Pk_(1, 1) << "," << Pk_(2, 2) << "\n";
}

void UnscentedKF::log(double x, double y, double theta) {
  if (!xy_log_.is_open())
    return;

  // Tiempo en ROS en segundos
  double t = ros::Time::now().toSec();

  // UKF state actual
  double x_ukf = xk_(0);
  double y_ukf = xk_(1);
  double th_ukf = xk_(2);

  xy_log_ << std::fixed << std::setprecision(6) // decimals
          << t << ","                           // time
          << x << ","                           // x pose
          << y << ","                           // y pose
          << theta << ","                       // theta pose
          << x_ukf << ","                       // x ukf
          << y_ukf << ","                       // y ukf
          << th_ukf << "\n";                    // th ukf
}
