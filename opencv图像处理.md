### 0.头文件
```cpp
#include <opencv2/opencv.hpp>
```
### 1.高斯滤波
```cpp
cv::GaussianBlur(src, src, cv::Size(5,5), 3);
```

### 2.形态学操作
```cpp
cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
cv::morphologyEx(dep_im, dep_im, cv::MORPH_CLOSE, element); 
// 闭操作（去除黑色空洞），消除值小于邻域内的点的孤立异常值
// 开操作先腐蚀再膨胀，用来分离相互靠的很近的部分，消除大于邻域内的孤立异常值
// 顶帽操作显示的是比源图像更亮的环绕部分
// 黑帽操作显示的是比源图像更暗的环绕部分
```

### 3.直线检测
```cpp
// 霍夫变换
vector<cv::Vec4f> plines;
cv::HoughLinesP(dep_im, plines, 1, CV_PI / 180.0, 30, 10, 1);
// LSD
auto ls = cv::createLineSegmentDetector(0);
ls->detect(dep_im, plines);
// FLD
#include <opencv2/ximgproc.hpp>
auto fld = cv::ximgproc::createFastLineDetector(33, 1.414213562f, 50.0, 50.0, 3, true);
fld->detect(dep_im, plines);
```

### 4.Canny边缘检测
```cpp
cv::Canny(dep_im, dep_im, 100, 200);
```

### 1024.参考
- [直线检测](https://blog.csdn.net/WZZ18191171661/article/details/101116949)
- [色彩空间](https://blog.51cto.com/u_15353042/3751269)
- [光照不均匀校正](https://blog.51cto.com/u_13984132/5617762)
- [形态学操作](https://aitechtogether.com/article/28165.html)
