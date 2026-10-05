#include <localization/Localization.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "localization_node");
  ros::NodeHandle nh; // private namespace, to read ~landmarks_file

  Localization node(nh);
  ros::AsyncSpinner spin(2);
  spin.start();
  ros::waitForShutdown();
  return 0;
}