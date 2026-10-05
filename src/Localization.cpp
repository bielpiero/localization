#include "localization/Localization.h"
#include "localization/UnscentedKF.h"
#include "localization/models/DiffDriveModel.hpp"

Localization::Localization(ros::NodeHandle &nh)
    : nh_(nh), has_last_pose_(false), last_time_(0.0) {
  ros::NodeHandle pnh("~");

  // Debug: mostrar que constructor se llama
  ROS_INFO("Localization ctor: entering constructor");

  std::string landmarks_file;
  if (!pnh.getParam("landmarks_file", landmarks_file)) {
    ROS_ERROR("Parameter ~landmarks_file is required but not set.");
    ros::shutdown();
    return;
  }
  ROS_INFO("Parameter landmarks_file = %s", landmarks_file.c_str());
  if (!loadLandmarksFromYAML(landmarks_file, landmarks_)) {
    ROS_ERROR("Failed to load landmarks from %s", landmarks_file.c_str());
    ros::shutdown();
    return;
  }
  std::vector<double> initial_pose;
  if (!pnh.getParam("initial_pose", initial_pose)) {
    ROS_ERROR("Parameter ~initial_pose is required but not set.");
    ros::shutdown();
    return;
  }
  if (initial_pose.size() < 3) {
    ROS_ERROR("initial_pose must have 3 elements (x,y,theta). Got %zu",
              initial_pose.size());
    ros::shutdown();
    return;
  }
  ROS_INFO("Parameter initial_pose = [%.6f, %.6f, %.6f]", initial_pose[0],
           initial_pose[1], initial_pose[2]);

  model_ = std::make_shared<DiffDriveModel>();
  Eigen::VectorXd init_pose(3);
  init_pose << initial_pose[0], initial_pose[1], initial_pose[2];
  model_->init(init_pose);
  ips_ = std::make_shared<UnscentedKF>(model_, landmarks_, init_pose);

  pnh.param("camera_height", camera_height_, 0.86); // por defecto 1m
  pnh.param("theta_offset", theta_offset_, 0.0);
  pnh.param("theta_sign", theta_sign_, 1.0);

  ROS_INFO("Parameter camera_height = %.3f", camera_height_);
  ROS_INFO("theta_offset = %.6f, theta_sign = %.1f", theta_offset_,
           theta_sign_);

  // DESCOMENTAR CUANDO ESTE MONTADO EN LA SILLA
  pose_sub_ = nh_.subscribe("/pose", 10, &Localization::poseCallback, this);
  increments_sub_ = nh_.subscribe("/kinematic_model_increment", 10,
                                  &Localization::incrementsCallback, this);

  // Suscripciones: usar nh_ (global) para tópicos públicos
  ROS_INFO("Subscribing to /aruco_markers using global namespace");
  aruco_sub_ =
      nh_.subscribe("/aruco_markers", 1, &Localization::arucoCallback, this);

  // Publishers
  pose_pub_ = nh.advertise<geometry_msgs::Pose2D>("/update_pose", 10);
  geometry_msgs::Pose2D pose_msg;
  pose_msg.x = initial_pose[0];
  pose_msg.y = initial_pose[1];
  pose_msg.theta = initial_pose[2];

  pose_pub_.publish(pose_msg);

  timer_ = nh_.createWallTimer(ros::WallDuration(0.05),
                               &Localization::timerCallback, this);

  ROS_INFO("UKF node initialized. Listening for /pose and /aruco_markers …");
}

bool Localization::loadLandmarksFromYAML(const std::string &path,
                                         std::vector<Landmark> &out_landmarks) {
  try {
    YAML::Node root = YAML::LoadFile(path);
    YAML::Node lms = root["landmarks"];
    if (!lms || !lms.IsSequence()) {
      ROS_ERROR("landmarks: node missing or not a sequence in YAML.");
      return false;
    }

    for (const auto &lm_node : lms) {
      out_landmarks.push_back(
          Landmark(lm_node["id"].as<int>(), lm_node["x"].as<double>(),
                   lm_node["y"].as<double>(), lm_node["z"].as<double>()));
    }
  } catch (const std::exception &e) {
    ROS_ERROR("Exception parsing landmarks YAML: %s", e.what());
    return false;
  }
  ROS_INFO("Loaded %zu landmarks from YAML.", out_landmarks.size());
  return true;
}

void Localization::incrementsCallback(
    const canusb::KinematicModelIncrement::ConstPtr &msg) {

  Eigen::VectorXd inc(2);
  inc(0) = msg->d_lin;
  inc(1) = msg->d_ang;
  ips_->setIncrement(inc);
}

// DESCOMENTAR CUANDO ESTE MONTADO EN LA SILLA
void Localization::poseCallback(const geometry_msgs::Pose2D::ConstPtr &msg) {

  ips_->log(msg->x, msg->y, msg->theta);
}

void Localization::arucoCallback(
    const aruco_detector::ArucoMarkers::ConstPtr &msg) {

  for (size_t i = 0; i < msg->markers.size(); ++i) {
    const auto &m = msg->markers[i];
    //ROS_INFO("marker[%zu] id=%u dist=%.6f theta=%.6f phi=%.6f", i, m.id,
    //         m.distance, m.theta, m.phi);
  }

  std::lock_guard<std::mutex> lock(buffer_mutex_);
  buffered_measurements_.clear();

  for (const auto &pwid : msg->markers) {
    double r = pwid.distance; // Distancia en el plano 2D
    if (std::isnan(r) || r <= 0.0) {
      ROS_WARN("Ignoring invalid marker measurement r=%.6f", r);
      continue;
    }

    double bearing = theta_sign_ * (pwid.theta + theta_offset_);
    buffered_measurements_.push_back(Landmark(pwid.id, r, bearing));
  }
}

void Localization::timerCallback(const ros::WallTimerEvent &event) {
  std::vector<Landmark> measurements_copy;
  {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    measurements_copy.swap(buffered_measurements_);
  }
  ROS_INFO("-------------------> TIMER: Meauruements size: %ld",
           measurements_copy.size());
  ips_->execute(measurements_copy);
  // 3) (Optional) publish or log the new fused state:
  Eigen::VectorXd fused = ips_->getState();
  if (fused.size() >= 3) {
    ROS_INFO("Fused state: [%f, %f, %f]", fused(0), fused(1), fused(2));
  } else {
    ROS_WARN("getState() size = %ld, expected >= 3", fused.size());
  }
}