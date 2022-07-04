```cpp
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>
#include <sys/time.h>

using namespace caffe;
// using namespace std;

// std::string CLASSES[81] = { "__background__",
// "plastic_bag", "paper", "car", "motorcycle",
// "airplane", "bus", "train", "truck", "boat",
// "traffic light", "fire hydrant", "stop sign", "parking meter",
// "bench", "bird", "cat",
// "dog", "horse", "sheep", "cow",
// "elephant", "bear", "zebra", "giraffe" ,
// "backpack", "umbrella", "handbag", "tie" ,
// "suitcase", "frisbee", "skis", "snowboard" ,
// "sports ball", "kite", "baseball bat", "baseball glove" ,
// "skateboard", "surfboard", "tennis racket", "bottle" ,
// "wine glass", "cup", "fork", "knife" ,
// "spoon", "bowl", "banana", "apple" ,
// "sandwich", "orange", "broccoli", "carrot" ,
// "hot dog", "pizza", "donut", "cake" ,
// "chair", "couch", "potted plant", "bed" ,
// "dining table", "toilet", "tv", "laptop" ,
// "mouse", "remote", "keyboard", "cell phone" ,
// "microwave", "oven", "toaster", "sink" ,
// "refrigerator", "book", "clock", "vase" ,
// "scissors", "teddy bear", "hair drier", "toothbrush" ,
// };

std::string CLASSES[81] = {
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"};

int main()
{
    // std::string model_file("/home/lwd/code/dl/ssd/caffe/examples/MobileNet-SSD/voc/compare.prototxt");
    // std::string weights_file("/home/lwd/code/dl/ssd/caffe/snapshot/mobilenet_iter_40000.caffemodel");
    std::string weights_file("/home/lwd/ssddetect/MobileNetSSD_deploy.caffemodel");
    std::string model_file("/home/lwd/ssddetect/MobileNetSSD_deploy.prototxt");
    cv::Mat mean_;

    // set net
    Caffe::set_mode(Caffe::CPU);
    shared_ptr<Net<float>> net_;
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
    Blob<float> *input_layer = net_->input_blobs()[0];
    int num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
    cv::Size input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i)
    {
        cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1, 1);
        channels.push_back(channel);
    }
    cv::merge(channels, mean_);
    //
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    net_->Reshape();
    // input shape
    std::vector<cv::Mat> input_channels;
    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }
    // input data
    float cost_time = 0;
    for (int t = 0; t < 1000; ++t)
    {
        cv::Mat img = cv::imread("/home/lwd/ncs/byz.jpg");
        cv::Mat sample = img;
        cv::Mat sample_resized;
        if (sample.size() != input_geometry_)
            cv::resize(sample, sample_resized, input_geometry_);
        else
            sample_resized = sample;
        cv::Mat sample_float;
        if (num_channels_ == 3)
            sample_resized.convertTo(sample_float, CV_32FC3, 0.007843);
        else
            sample_resized.convertTo(sample_float, CV_32FC1, 0.007843);
        cv::Mat sample_normalized;
        cv::subtract(sample_float, mean_, sample_normalized);
        cv::split(sample_normalized, input_channels);
        struct timeval start, end;
        gettimeofday(&start, NULL);
        net_->Forward();
        gettimeofday(&end, NULL);
        cost_time += (end.tv_usec - start.tv_usec) / 1000000.0 + end.tv_sec - start.tv_sec;
        Blob<float> *result_blob = net_->output_blobs()[0];
        const float *result = result_blob->cpu_data();
        const int num_det = result_blob->height();
        std::vector<std::vector<float>> detections;
        for (int k = 0; k < num_det; ++k)
        {
            if (result[0] == -1)
            {
                result += 7;
                continue;
            }
            std::vector<float> detection(result, result + 7);
            detections.push_back(detection);
            result += 7;
        }
        for (int i = 0; i < detections.size(); ++i)
        {
            const std::vector<float> &d = detections[i];
            // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            CHECK_EQ(d.size(), 7);
            const float score = d[2];
            if (score >= 0.1)
            {
                cv::Point pt1, pt2;
                pt1.x = (img.cols * d[3]);
                pt1.y = (img.rows * d[4]);
                pt2.x = (img.cols * d[5]);
                pt2.y = (img.rows * d[6]);
                cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0), 1, 8, 0);
                char label[100];
                //std::cout << CLASSES[0] << std::endl;
                sprintf(label, "%s,%f", CLASSES[static_cast<int>(d[1])].c_str(), score);
                int baseline;
                cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 0, &baseline);
                cv::Point pt3;
                pt3.x = pt1.x + size.width;
                pt3.y = pt1.y - size.height;
                cv::rectangle(img, pt1, pt3, cv::Scalar(0, 255, 0), -1);
                cv::putText(img, label, pt1, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }
        //cv::imshow("show", img);
        //cv::waitKey();
    }
    std::cout << cost_time / 1000 << std::endl;
    return 0;
}

```
- CMakeLists.txt
- 需要把caffe/build/src/caffe/proto和caffe/include/caffe拷贝到项目的include
```
cmake_minimum_required(VERSION 2.8.3)
project(cpp)

add_compile_options(-std=c++14 -g)
set(CMAKE_CXX_FLAGS "-DCPU_ONLY=1")

find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  include
)

link_directories(
	${OpenCV_LIBRARY_DIRS}
)

add_executable(main main.cc)
target_link_libraries(main
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  /home/lwd/caffe/build/lib/libcaffe.so
  glog
  boost_system
)

```
