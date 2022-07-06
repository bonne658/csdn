### 1.安装依赖项:
```
sudo apt-get install build-essential libgtk2.0-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen3-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev default-jdk ant libvtk5-qt4-dev
```

### 2.编译安装
```
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
```

### 3.配置动态链接
```
sudo gedit /etc/ld.so.conf.d/opencv.conf
添加opencv的lib的路径:/usr/local/lib
sudo ldconfig
```

### 4.测试:
```
mkdir ~/opencv-lena
cd ~/opencv-lena
gedit DisplayImage.cpp
```

```cpp
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
int main(int argc, char** argv )
{
  if ( argc != 2 )
  {
    printf("usage: DisplayImage.out <Image_Path>\n");
    return -1;
  }
  Mat image;
  image = imread( argv[1], 1 );
  if ( !image.data )
  {
    printf("No image data \n");
    return -1;
  }
  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", image);
  waitKey(0);
  return 0;
}
```

- gedit CMakeLists.txt
```
cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
```
```
cd ~/opencv-lena
cmake .
make
./DisplayImage lena.jpg 
```
