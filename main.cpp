

#include <iostream>
#include <algorithm>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "opencv_functional.h"

int main() {
  cv::Mat image = cv::imread("../../images_for_study/realpic.jpg", cv::IMREAD_COLOR);
  //cv::copyMakeBorder(image, image, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

  cv::namedWindow("W", cv::WINDOW_NORMAL);
////  cv::imshow("W", image);
  processImage(image);
  cv::imshow("W", image);
  cv::waitKey();
  for (int i = 1; i <= 20; ++i) {
    cv::Mat image = cv::imread("../../images_for_study/test" + std::to_string(i) + ".png", cv::IMREAD_COLOR);
    //cv::copyMakeBorder(image, image, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));


    processImage(image);
//    cv::namedWindow("W", cv::WINDOW_NORMAL);
  cv::imshow("W", image);
  cv::waitKey(0);
  }
  return 0;
}