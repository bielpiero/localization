#include "localization/Localization.h"
#include "localization/ExtendedKF.h"
#include "localization/UnscentedKF.h"
#include "localization/models/IncrementalModel.hpp"
#include "localization/models/VelocityModel.hpp"

#include <geometry_msgs/TwistStamped.h>
#include <cmath>

Localization::Localization(ros::NodeHandle &nh)
    : nh_(nh), has_last_increment_time_(false), has_last_speed_time_(false) {

  ros::NodeHandle pnh("~");

  ROS_INFO("Localization ctor: entering constructor");

  // --------------------------------------------------------------------------
  // Map
  // --------------------------------------------------------------------------
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

  // --------------------------------------------------------------------------
  // Initial pose
  // --------------------------------------------------------------------------
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

  Eigen::VectorXd init_pose(3);
  init_pose << initial_pose[0], initial_pose[1], initial_pose[2];

  // --------------------------------------------------------------------------
  // Four estimators
  // --------------------------------------------------------------------------
  ips_ukf_inc_ =
      std::make_shared<UnscentedKF<IncrementalModel>>(landmarks_, init_pose,
                                                       "ukf_incremental");

  ips_ukf_vel_ =
      std::make_shared<UnscentedKF<VelocityModel>>(landmarks_, init_pose,
                                                    "ukf_velocity");

  ips_ekf_inc_ =
      std::make_shared<ExtendedKF<IncrementalModel>>(landmarks_, init_pose,
                                                      "ekf_incremental");

  ips_ekf_vel_ =
      std::make_shared<ExtendedKF<VelocityModel>>(landmarks_, init_pose,
                                                   "ekf_velocity");

  // --------------------------------------------------------------------------
  // Measurement parameters
  // --------------------------------------------------------------------------
  pnh.param("camera_height", camera_height_, 0.86);
  pnh.param("theta_offset", theta_offset_, 0.0);
  pnh.param("theta_sign", theta_sign_, 1.0);

  ROS_INFO("Parameter camera_height = %.3f", camera_height_);
  ROS_INFO("theta_offset = %.6f, theta_sign = %.1f", theta_offset_,
           theta_sign_);

  // --------------------------------------------------------------------------
  // Subscribers
  //
  // IMPORTANT:
  // There is intentionally NO 20-Hz localization timer anymore.
  // Prediction is triggered by kinematic input and correction by ArUco input.
  // With ros::spin() these callbacks are processed sequentially and preserve
  // a deterministic event order.
  // --------------------------------------------------------------------------
  pose_sub_ = nh_.subscribe("/pose/raw", 20, &Localization::poseCallback, this);

  increments_sub_ = nh_.subscribe("/kinematics/increments", 50,
                                  &Localization::incrementsCallback, this);

  speeds_sub_ = nh_.subscribe("/kinematics/speeds", 50,
                              &Localization::speedsCallback, this);

  aruco_sub_ =
      nh_.subscribe("/aruco_markers", 5, &Localization::arucoCallback, this);

  // --------------------------------------------------------------------------
  // Publishers: one topic for every estimator/model combination
  // --------------------------------------------------------------------------
  pose_pub_ukf_inc_ = nh_.advertise<geometry_msgs::Pose2D>(
      "/pose/filtered/ukf/incremental", 10);

  pose_pub_ukf_vel_ =
      nh_.advertise<geometry_msgs::Pose2D>("/pose/filtered/ukf/velocity", 10);

  pose_pub_ekf_inc_ = nh_.advertise<geometry_msgs::Pose2D>(
      "/pose/filtered/ekf/incremental", 10);

  pose_pub_ekf_vel_ =
      nh_.advertise<geometry_msgs::Pose2D>("/pose/filtered/ekf/velocity", 10);

  publishFilteredPoses();

  ROS_INFO("Localization initialized.");
  ROS_INFO("Increment input : /kinematics/increments");
  ROS_INFO("Velocity input  : /kinematics/speeds");
  ROS_INFO("Raw pose        : /pose/raw");
  ROS_INFO("Visual input    : /aruco_markers");
}

// =============================================================================
// Landmark map
// =============================================================================

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

// =============================================================================
// Incremental kinematic input
// =============================================================================

void Localization::incrementsCallback(
    const canusb::KinematicModelIncrement::ConstPtr &msg) {

  const ros::WallTime computation_start = ros::WallTime::now();

  // The current custom increment message has no Header, therefore dt is
  // temporarily obtained from callback arrival time. The IncrementalModel
  // itself does not use dt, but IPositioningSystem has a common interface.
  //
  // Recommended final version:
  // add std_msgs/Header to KinematicModelIncrement and use msg->header.stamp.
  const ros::Time stamp = ros::Time::now();

  if (!has_last_increment_time_) {
    last_increment_time_ = stamp;
    has_last_increment_time_ = true;
    return;
  }

  const double dt = (stamp - last_increment_time_).toSec();

  last_increment_time_ = stamp;

  if (!std::isfinite(dt) || dt <= 0.0) {
    ROS_WARN("Invalid incremental dt: %.9f", dt);
    return;
  }

  Eigen::VectorXd input(2);
  input(0) = msg->d_lin;
  input(1) = msg->d_ang;

  // SAME kinematic sample and SAME dt for both incremental estimators.
  ips_ukf_inc_->setInput(input, dt);
  ips_ukf_inc_->execute({});

  ips_ekf_inc_->setInput(input, dt);
  ips_ekf_inc_->execute({});

  publishFilteredPoses();

  const double elapsed_ms =
      (ros::WallTime::now() - computation_start).toSec() * 1000.0;

  ROS_DEBUG("Incremental UKF+EKF prediction time: %.3f ms", elapsed_ms);
}

// =============================================================================
// Velocity kinematic input
// =============================================================================

void Localization::speedsCallback(
    const geometry_msgs::TwistStamped::ConstPtr &msg) {

  const ros::WallTime computation_start = ros::WallTime::now();

  ros::Time stamp = msg->header.stamp;

  // Defensive fallback if the publisher did not populate the stamp.
  if (stamp.isZero()) {
    stamp = ros::Time::now();
  }

  if (!has_last_speed_time_) {
    last_speed_time_ = stamp;
    has_last_speed_time_ = true;
    return;
  }

  const double dt = (stamp - last_speed_time_).toSec();

  last_speed_time_ = stamp;

  if (!std::isfinite(dt) || dt <= 0.0) {
    ROS_WARN("Invalid velocity dt: %.9f", dt);
    return;
  }

  Eigen::VectorXd input(2);
  input(0) = msg->twist.linear.x;
  input(1) = msg->twist.angular.z;

  // SAME velocity sample and SAME dt for both velocity estimators.
  ips_ukf_vel_->setInput(input, dt);
  ips_ukf_vel_->execute({});

  ips_ekf_vel_->setInput(input, dt);
  ips_ekf_vel_->execute({});

  publishFilteredPoses();

  const double elapsed_ms =
      (ros::WallTime::now() - computation_start).toSec() * 1000.0;

  ROS_DEBUG("Velocity UKF+EKF prediction time: %.3f ms", elapsed_ms);
}

// =============================================================================
// Raw encoder-integrated pose
// =============================================================================

void Localization::poseCallback(const geometry_msgs::Pose2D::ConstPtr &msg) {

  // Raw odometry is NOT fed back into any estimator.
  // It is only logged as a baseline/reference diagnostic.
  ips_ukf_inc_->log(msg->x, msg->y, msg->theta);
  ips_ukf_vel_->log(msg->x, msg->y, msg->theta);

  ips_ekf_inc_->log(msg->x, msg->y, msg->theta);
  ips_ekf_vel_->log(msg->x, msg->y, msg->theta);
}

// =============================================================================
// Visual correction
// =============================================================================

void Localization::arucoCallback(
    const aruco_detector::ArucoMarkers::ConstPtr &msg) {

  const ros::WallTime computation_start = ros::WallTime::now();

  std::vector<Landmark> measurements;
  measurements.reserve(msg->markers.size());

  for (const auto &pwid : msg->markers) {

    const double range = pwid.distance;

    if (!std::isfinite(range) || range <= 0.0) {
      ROS_WARN("Ignoring invalid marker measurement r=%.6f", range);
      continue;
    }

    const double bearing = theta_sign_ * (pwid.theta + theta_offset_);

    if (!std::isfinite(bearing)) {
      ROS_WARN("Ignoring invalid bearing for marker ID %u", pwid.id);
      continue;
    }

    measurements.emplace_back(pwid.id, range, bearing);
  }

  if (measurements.empty()) {
    return;
  }

  // All four estimators receive EXACTLY the same visual measurement set.
  //
  // With a single-threaded ROS spinner no kinematic callback can be interleaved
  // between these four calls.
  ips_ukf_inc_->execute(measurements);
  ips_ukf_vel_->execute(measurements);

  ips_ekf_inc_->execute(measurements);
  ips_ekf_vel_->execute(measurements);

  publishFilteredPoses();

  const double elapsed_ms =
      (ros::WallTime::now() - computation_start).toSec() * 1000.0;

  ROS_DEBUG("Four-filter ArUco update time: %.3f ms", elapsed_ms);
}

// =============================================================================
// Output
// =============================================================================

void Localization::publishFilteredPoses() {

  auto publish_state = [](const std::shared_ptr<IPositioningSystem> &ips,
                          ros::Publisher &publisher) {
    const Eigen::VectorXd state = ips->getState();

    if (state.size() < 3) {
      return;
    }

    geometry_msgs::Pose2D msg;
    msg.x = state(0);
    msg.y = state(1);
    msg.theta = state(2);

    publisher.publish(msg);
  };

  publish_state(ips_ukf_inc_, pose_pub_ukf_inc_);
  publish_state(ips_ukf_vel_, pose_pub_ukf_vel_);

  publish_state(ips_ekf_inc_, pose_pub_ekf_inc_);
  publish_state(ips_ekf_vel_, pose_pub_ekf_vel_);
}
