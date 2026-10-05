#ifndef LOCALIZATION_H
#define LOCALIZATION_H

#include <ros/ros.h>
#include <geometry_msgs/Pose2D.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <mutex>

#include "IPositioningSystem.h"

#include <canusb/KinematicModelIncrement.h>
#include <aruco_detector/ArucoMarkers.h>


class Localization
{
public:
  Localization(ros::NodeHandle &nh);

private:
  ros::NodeHandle nh_;
  ros::Subscriber pose_sub_;
  ros::Subscriber increments_sub_;
  ros::Subscriber aruco_sub_;
  ros::WallTimer timer_;
  ros::Publisher pose_pub_;

  std::shared_ptr<IPositioningSystem> ips_;
  std::shared_ptr<BaseModel> model_;
  std::vector<Landmark> landmarks_;
  std::vector<Landmark> buffered_measurements_;

  std::mutex buffer_mutex_;
  bool has_last_pose_;
  double last_time_;
  Eigen::Vector3d last_pose_; // last [x,y,θ]
  
  double camera_height_;
  double theta_offset_;
  double theta_sign_;

  double last_pose_x_ = 0.0;
  double last_pose_y_ = 0.0;
  
  bool loadLandmarksFromYAML(const std::string &path, std::vector<Landmark> &out_landmarks);

  void poseCallback(const geometry_msgs::Pose2D::ConstPtr &msg);
  void incrementsCallback(const canusb::KinematicModelIncrement::ConstPtr &msg);

  void arucoCallback(const aruco_detector::ArucoMarkers::ConstPtr &msg);

  void timerCallback(const ros::WallTimerEvent &evt);
};

#endif