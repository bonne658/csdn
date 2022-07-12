- 不要在函数中，特别是回调函数中声明TransformListener对象
- TransformListener对象应该被限定为持久化，否则它的缓存将无法填充，并且几乎每个查询都将失败。出现`Lookup would require extrapolation at time 1657511018.391161648, but only time 1657511018.856185462 is in the buffer`之类的问题
- 一个常见的方法是使TransformListener对象成为一个类的成员变量

```cpp
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <geometry_msgs/Quaternion.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/LaserScan.h>

class Callback
{
public:
  tf::TransformListener listener_;
  tf::StampedTransform transform_;
  ros::Subscriber sub_;
  double last_x_;
  double last_y_;
  double last_angle_;

  Callback(ros::NodeHandle n)
  {
    last_x_ = 0;
    last_y_ = 0;
    last_angle_ = 0;
    sub_ = n.subscribe<sensor_msgs::LaserScan>("/scan", 1, &Callback::CB, this);
  }
  void CB(const sensor_msgs::LaserScanConstPtr &scan);
};

void Callback::CB(const sensor_msgs::LaserScanConstPtr &scan)
{

  try
  {
    // base_link to map
    listener_.waitForTransform("/map",
                               "/base_link",
                               ros::Time(0), ros::Duration(0.2));
    listener_.lookupTransform("map",
                              "base_link",
                              ros::Time(0), transform_);
  }
  catch (tf::TransformException &ex)
  {
    ROS_ERROR("%s", ex.what());
    return;
  }
  double dx = transform_.getOrigin().x() - last_x_;
  double dy = transform_.getOrigin().y() - last_y_;
  double distance = sqrt(dx * dx + dy * dy);
  tf::Quaternion quat;
  double qx = transform_.getRotation().x(), qy = transform_.getRotation().y(),
         qz = transform_.getRotation().z(), qw = transform_.getRotation().w();
  geometry_msgs::Quaternion qua;
  qua.x = qx;
  qua.y = qy;
  qua.z = qz;
  qua.w = qw;
  tf::quaternionMsgToTF(qua, quat);
  double roll, pitch, yaw;
  tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
  if (yaw >= 6.27)
    yaw = 0;
  double a = (yaw - last_angle_) * 180 / 3.14159;
  if (a < 0)
    a = -a;
  if (a >= 30 || distance >= 1.1)
  {
    ROS_INFO("%lf, %lf", distance, a);
    last_x_ = transform_.getOrigin().x();
    last_y_ = transform_.getOrigin().y();
    last_angle_ = yaw;
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "lwd_tf");
  ros::NodeHandle n;
  Callback cb(n);
  ros::spin();
  return 0;
}

```
- 下面的就不要看了，历史的眼泪
```cpp
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>

void CB(const sensor_msgs::PointCloud2ConstPtr& pc, ros::Publisher& pub) {
  float x = 2.8, y = z = 1.1;
  tf::TransformListener listener;
  tf::StampedTransform transform;
  try {
  	// camera to base_link
    listener.waitForTransform("/base_link",
                              "/realsense_camera_depth_optical_frame",
                              ros::Time(0), ros::Duration(0.2));
    listener.lookupTransform("base_link",
                             "realsense_camera_depth_optical_frame",
                             ros::Time(0), transform);
  } catch (tf::TransformException& ex) {
    ROS_ERROR("%s", ex.what());
    return;
  }
  
    float R[4][4];
    float qx = transform.getRotation().x(), qy = transform.getRotation().y(),
          qz = transform.getRotation().z(), qw = transform.getRotation().w();
    float scale = 1 / sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    qx *= scale, qy *= scale, qz *= scale, qw *= scale;
    R[0][0] = 1 - 2 * qy * qy - 2 * qz * qz;
    R[0][1] = 2 * qx * qy - 2 * qz * qw;
    R[0][2] = 2 * qx * qz + 2 * qy * qw;
    R[0][3] = 0;
    R[1][0] = 2 * qx * qy + 2 * qz * qw;
    R[1][1] = 1 - 2 * qx * qx - 2 * qz * qz;
    R[1][2] = 2.0f * qy * qz - 2.0f * qx * qw;
    R[1][3] = 0;
    R[2][0] = 2.0f * qx * qz - 2.0f * qy * qw;
    R[2][1] = 2.0f * qy * qz + 2.0f * qx * qw;
    R[2][2] = 1.0f - 2.0f * qx * qx - 2.0f * qy * qy;
    R[2][3] = R[3][0] = R[3][1] = R[3][2] = 0;
    R[3][3] = 1;
    float xx = x * R[0][0] + y * R[0][1] + z * R[0][2] + R[0][3] +
               transform.getOrigin().x();
    float yy = x * R[1][0] + y * R[1][1] + z * R[1][2] + R[1][3] +
               transform.getOrigin().y();
    float zz = x * R[2][0] + y * R[2][1] + z * R[2][2] + R[2][3] +
               transform.getOrigin().z();
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "listener");
  ros::NodeHandle n;
  ros::Publisher pcl_pub =
      n.advertise<sensor_msgs::PointCloud2>("pcl_output", 1);
  ros::Subscriber sub = n.subscribe<sensor_msgs::PointCloud2>(
      "/depth/color/points", 1, boost::bind(&CB, _1, pcl_pub));
  ros::spin();
  return 0;
}
```

