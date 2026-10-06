#ifndef LOCALIZATION_H
#define LOCALIZATION_H

#include "IPositioningSystem.h"

#include <aruco_detector/ArucoMarkers.h>
#include <canusb/KinematicModelIncrement.h>

#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/TwistStamped.h>
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <vector>

class Localization {
public:
  explicit Localization(ros::NodeHandle &nh);

private:
  bool loadLandmarksFromYAML(
      const std::string &path,
      std::vector<Landmark> &out_landmarks);

  void poseCallback(
      const geometry_msgs::Pose2D::ConstPtr &msg);

  void incrementsCallback(
      const canusb::KinematicModelIncrement::ConstPtr &msg);

  void speedsCallback(
      const geometry_msgs::TwistStamped::ConstPtr &msg);

  void arucoCallback(
      const aruco_detector::ArucoMarkers::ConstPtr &msg);

  void publishFilteredPoses();

private:
  ros::NodeHandle nh_;

  ros::Subscriber pose_sub_;
  ros::Subscriber increments_sub_;
  ros::Subscriber speeds_sub_;
  ros::Subscriber aruco_sub_;

  ros::Publisher pose_pub_ukf_inc_;
  ros::Publisher pose_pub_ukf_vel_;
  ros::Publisher pose_pub_ekf_inc_;
  ros::Publisher pose_pub_ekf_vel_;

  std::shared_ptr<IPositioningSystem> ips_ukf_inc_;
  std::shared_ptr<IPositioningSystem> ips_ukf_vel_;
  std::shared_ptr<IPositioningSystem> ips_ekf_inc_;
  std::shared_ptr<IPositioningSystem> ips_ekf_vel_;

  std::vector<Landmark> landmarks_;

  ros::Time last_increment_time_;
  ros::Time last_speed_time_;

  bool has_last_increment_time_;
  bool has_last_speed_time_;

  double camera_height_;
  double theta_offset_;
  double theta_sign_;
};

#endif
