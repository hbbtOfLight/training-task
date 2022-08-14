//
// Created by katar on 06.08.2022.
//

#include "opencv_functional_v2.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

struct DSU {
  DSU(std::size_t size) : parents(size, -1) {}
  int GetParent(std::size_t child_idx) const {
    while (parents[child_idx] >= 0) {
      child_idx = parents[child_idx];
    }
    return child_idx;
  }
  void Merge(int child1, int child2) {
    int parent1 = GetParent(child1), parent2 = GetParent(child2);
    if (parent1 == parent2) {
      return;
    }
    if (parents[parent1] > parents[parent2]) {
      std::swap(parent2, parent1);
    }
    parents[parent1] += parents[parent2];
    parents[parent2] = parent1;
  }

  size_t Size() const {
    return parents.size();
  }
  int operator[](int i) const {
    return parents[i];
  }
 private:
  std::vector<int> parents;
};

cv::Scalar getColor() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<uchar> dis(0, 255);
  return cv::Scalar(dis(gen), dis(gen), dis(gen));
}

void getBlackMask(const cv::Mat& img_hsv, cv::Mat& mask) {
  std::vector<uchar> lower = {0, 0, 1};
  std::vector<uchar> upper = {179, 255, 255};
  cv::inRange(img_hsv, lower, upper, mask);
}

double rectSquare(const cv::Rect& r) {
  return static_cast<double>(r.width) * r.height;
}

double getMatrixMin(cv::Mat& result) {
  double minval, maxval;
  cv::Point minloc, maxloc;
  cv::minMaxLoc(result, &minval, &maxval, &minloc, &maxloc);
  return minval;
}

void getImageHistogram(const cv::Mat& hsv_img, const cv::Mat& hist) {
  cv::Mat mask;
  getBlackMask(hsv_img, mask);
  static const int hist_sizes[] = {50, 65, 65};
  static const float range[] = {0.0, 256.0};
  static const float h_range[] = {0.0, 180.0};

  static const float* hist_ranges[] = {h_range, range, range};
  static int channels[] = {0, 1, 2};
  cv::calcHist(&hsv_img, 1, channels, mask, hist, 2, hist_sizes, hist_ranges);
  cv::normalize(hist, hist, 1, 0, cv::NORM_MINMAX);
}

std::pair<double, double> compareHistogram(const cv::Mat& img1_hsv, const cv::Mat& img2_hsv) {
  cv::Mat hist1, hist2;
  getImageHistogram(img1_hsv, hist1);
  getImageHistogram(img2_hsv, hist2);

  return {cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL), cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA)};
}

DSU getColorDSU(const std::vector<cv::Mat>& hsvd) {
  DSU color_matches(hsvd.size());
  std::vector<std::vector<double>> sorted_stats;
  std::vector<std::vector<std::vector<double>>> shape_tables;
  for (int i = 0; i < hsvd.size(); ++i) {
    for (int j = i + 1; j < hsvd.size(); ++j) {
      auto [corr, bha] = compareHistogram(hsvd[i], hsvd[j]);
      if (corr >= 0.5 && bha <= 0.66) {
        color_matches.Merge(i, j);
      }
    }
  }
  return color_matches;
}

std::vector<cv::Scalar> getColors(const DSU& color_dsu) {
  std::vector<cv::Scalar> colors(color_dsu.Size(), cv::Scalar(255, 255, 255));
  for (int i = 0; i < colors.size(); ++i) {
    int parent = color_dsu.GetParent(i);
    if (colors[parent] == cv::Scalar(255, 255, 255)) {
      colors[parent] = getColor();
    }
    colors[i] = colors[parent];
  }
  return colors;
}

void processImage(cv::Mat& input) {
  cv::Mat in_copy = input.clone();
  cv::medianBlur(input, input, 3);
  cv::GaussianBlur(input, input, cv::Size(7, 7), 0);
  cv::Mat hcv_input;
  cv::cvtColor(input, hcv_input, cv::COLOR_RGB2HSV);
  cv::Mat black_mask;
  cv::Scalar lb(0, 0, 0);
  cv::Scalar ub(180, 255, 40);
  cv::inRange(hcv_input, lb, ub, black_mask);
  cv::Mat mask;
  cv::Mat res_hcv, res_bgr;
  int rows = input.rows, cols = input.cols;
  int morph_rect_height = rows * 0.02, morph_rect_width = cols * 0.02;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_rect_height, morph_rect_width));
  cv::Scalar low_border = {0, 40, 0};
  cv::Scalar high_border = {180, 255, 255};
  cv::inRange(hcv_input, low_border, high_border, mask);
  mask += black_mask;
  cv::bitwise_and(input, input, res_bgr, mask);
  cv::cvtColor(input, hcv_input, cv::COLOR_BGR2HSV);
  cv::bitwise_and(hcv_input, hcv_input, res_hcv, mask);
  std::vector<std::vector<cv::Point>> contours;
  cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_ERODE, kernel);
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  contours.erase(std::remove_if(contours.begin(),
                                contours.end(),
                                [rows = input.rows, cols = input.cols](const std::vector<cv::Point>& c1) {
                                  cv::RotatedRect r1 = cv::minAreaRect(c1);
                                  bool ret = false;
                                  ret |= r1.size.area() <= 0.0005 * rows * cols;
                                  ret |= r1.center.x <= 0.01 * cols || r1.center.y <= 0.01 * rows
                                      || r1.center.x >= 0.99 * cols
                                      || r1.center.y >= 0.99 * rows;
                                  return ret;

                                }), contours.end());
  std::vector<cv::Mat> mats(contours.size());
  std::vector<cv::Mat> hsv_mats(contours.size());
  std::vector<cv::Point2f> box(4);
  std::vector<cv::Point> real_box(4);
  cv::Mat pts;
  cv::Mat temp;
  for (int i = 0; i < contours.size(); ++i) {
    cv::Rect square_bound = cv::boundingRect(contours[i]);
    int x_shift = square_bound.tl().x, y_shift = square_bound.tl().y;
    cv::RotatedRect min_enclosing = cv::minAreaRect(contours[i]);
    temp = res_bgr(square_bound);
    cv::Mat mask = cv::Mat::zeros(temp.size(), CV_8U);
    min_enclosing.points(box.data());
    for (int i = 0; i < 4; ++i) {
      real_box[i].x = box[i].x + 1 - x_shift;
      real_box[i].y = box[i].y + 1 - y_shift;
    }
    cv::fillConvexPoly(mask, real_box, cv::Scalar(255, 255, 255));
    cv::bitwise_and(temp, temp, mats[i], mask);
    cv::cvtColor(mats[i], hsv_mats[i], cv::COLOR_RGB2HSV);
  }
  std::cout << "CONTOURS.size: " << contours.size() << "\n";
  DSU color_dsu = getColorDSU(hsv_mats);
  std::vector<cv::Scalar> colors = getColors(color_dsu);
  for (auto& c: colors) {
    std::cout << "color " << c << "\n";
  }
  for (int i = 0; i < contours.size(); ++i) {
    cv::drawContours(in_copy, contours, i, colors[i]);
    cv::rectangle(in_copy, cv::boundingRect(contours[i]), colors[i], 2);
  }
  input = in_copy;
}
