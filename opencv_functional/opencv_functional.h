//
// Created by katar on 29.07.2022.
//

#ifndef TRY_TO_START__OPENCV_FUNCTIONAL_H_
#define TRY_TO_START__OPENCV_FUNCTIONAL_H_
#include <opencv2/core.hpp>
#include <random>
//#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
void processImage(cv::Mat& input);
double rectSquare(const cv::Rect& r);
std::pair<double, double> compareHistogram(cv::Mat& img1, cv::Mat& img2);

#endif //TRY_TO_START__OPENCV_FUNCTIONAL_H_
