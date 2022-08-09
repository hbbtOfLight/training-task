//
// Created by katar on 06.08.2022.
//

#include "opencv_functional_v2.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
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

void getBlackMask(cv::Mat& img_hsv, cv::Mat& mask) {
  // cv::cvtColor(img, mask, cv::COLOR_BGR2HSV);
  std::vector<uchar> lower = {0, 0, 5};
  std::vector<uchar> upper = {179, 255, 255};
  cv::inRange(img_hsv, lower, upper, mask);
//  cv::imshow("W", mask);
//  cv::waitKey();
}

std::vector<cv::Point> getBestLocations(cv::Mat& result, double best_value, double threshold = 0) {
  std::vector<cv::Point> best_matches;
  int count = 0;
  cv::Mat_<float> doubled(result);
  for (int i = 0; i < doubled.rows; ++i) {
    for (int j = 0; j < doubled.cols; ++j) {
      if (doubled.at<float>(i, j) <= best_value + threshold) {
        best_matches.emplace_back(cv::Point(j, i));
        ++count;
      }
      if (count > 100) {
        return {};
      }
    }
  }
  return best_matches;
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

void matchOnePictureToMany(cv::Mat& templ,
                           cv::Mat& grayscaled,
                           cv::Mat& input_without_back,
                           const cv::Scalar& color,
                           cv::Mat& input) {
  cv::Mat result;
  int res_cols = grayscaled.cols - grayscaled.cols + 1;
  int res_rows = grayscaled.rows - grayscaled.rows + 1;
  result.create(res_rows, res_cols, CV_32FC1);
  cv::matchTemplate(grayscaled, templ, result, cv::TM_SQDIFF);
  cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
  double best_location_value = getMatrixMin(result);
  auto locations = getBestLocations(result, best_location_value, 1e-3);
  for (auto& location: locations) {
    cv::rectangle(input,
                  location,
                  cv::Point(std::min(location.x + templ.cols, input.cols - 1),
                            std::min(location.y + templ.rows, input.rows - 1)),
                  color);
  }
}

void matchAndDraw(std::vector<cv::Mat>& examples,
                  cv::Mat& input_without_back,
                  cv::Mat grayscaled,
                  const std::vector<cv::Scalar>& colors, cv::Mat& input) {
  for (int i = 0; i < examples.size(); ++i) {
    if (getMatrixMin(examples[i]) < 250)
      matchOnePictureToMany(examples[i], grayscaled, input_without_back, colors[i], input);
  }
  //cv::imshow("W", input);
  //cv::waitKey();
}

void getImageHistogram(cv::Mat& hsv_img, cv::Mat& hist) {
  cv::Mat mask;
  getBlackMask(hsv_img, mask);
  static const int hist_sizes[] = {45, 65, 65};
  static const float range[] = {0.0, 256.0};
  static const float h_range[] = {0.0, 180.0};

  static const float* hist_ranges[] = {h_range, range, range};
  static int channels[] = {0, 1, 2};
  cv::calcHist(&hsv_img, 1, channels, mask, hist, 2, hist_sizes, hist_ranges);
  cv::normalize(hist, hist, 1, 0, cv::NORM_MINMAX);
}

std::pair<double, double> compareHistogram(cv::Mat& img1_hsv, cv::Mat& img2_hsv) {
  cv::Mat hist1, hist2;
  getImageHistogram(img1_hsv, hist1);
  getImageHistogram(img2_hsv, hist2);

  return {cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL), cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA)};
}

std::tuple<double, double, double> getMatch(cv::Mat& shape1, cv::Mat& shape2) {
  double dist1 = cv::matchShapes(shape1, shape2, cv::CONTOURS_MATCH_I1, 0);
  double dist2 = cv::matchShapes(shape1, shape2, cv::CONTOURS_MATCH_I2, 0);
  double dist3 = cv::matchShapes(shape1, shape2, cv::CONTOURS_MATCH_I3, 0);
  return {dist1, dist2, dist3};
}

bool checkMatch(const std::tuple<double, double, double>& dists,
                double max_dist1 = 0.05,
                double max_dist2 = 0.1,
                double max_dist3 = 0.05) {
  auto& [dist1, dist2, dist3] = dists;
  std::cout << dist1 << " " << dist2 << " " << dist3 << " " << "\n";
  return dist2 <= max_dist2 && dist1 <= max_dist1 || dist1 <= max_dist2 && dist3 <= max_dist3
      || dist2 <= max_dist1 && dist3 <= max_dist3;
}

DSU getColorDSU(std::vector<cv::Mat>& colored,
                std::vector<cv::Mat>& grayscaled,
                std::vector<cv::Mat>& hsvd,
                std::vector<cv::Mat>& mask) {
  DSU color_matches(colored.size());
  //cv::namedWindow("W1", cv::WINDOW_NORMAL);
  //cv::namedWindow("W2", cv::WINDOW_NORMAL);
  for (int i = 0; i < colored.size(); ++i) {
    for (int j = i + 1; j < colored.size(); ++j) {
      auto [corr, bha] = compareHistogram(hsvd[i], hsvd[j]);
      std::cout << "HIST:" << corr << " " << bha << std::endl;
  //    cv::imshow("W1", colored[i]);
  //    cv::imshow("W2", colored[j]);
      // cv::waitKey();
      if (corr >= 0.25 && bha <= 0.8) {
        auto matches = getMatch(mask[i], mask[j]);
        if (checkMatch(matches)) {
          color_matches.Merge(i, j);
        }
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
  // gammaCorrection(input);
  //cv::bitwise_not(input, input);
  cv::Mat hcv_input;
  cv::cvtColor(input, hcv_input, cv::COLOR_RGB2HSV);
//  cv::namedWindow("W", cv::WINDOW_NORMAL);
//  cv::imshow("W", hcv_input);
//  cv::waitKey();
//  cv::imshow("W", input);
//  cv::waitKey();
  cv::Mat black_mask;
  cv::Scalar lb(0, 0, 0);
  cv::Scalar ub(179, 255, 40);
  cv::inRange(hcv_input, lb, ub, black_mask);
//  cv::imshow("W", black_mask);
//  cv::waitKey();

  cv::Mat mask;
  cv::Mat res_hcv, res_bgr;
  int rows = input.rows, cols = input.cols;
  int morph_rect_height = rows * 0.02, morph_rect_width = cols * 0.02;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_rect_height, morph_rect_width));
  cv::medianBlur(hcv_input, hcv_input, 5);
//  cv::morphologyEx(hcv_input, hcv_input, cv::MORPH_OPEN, kernel);
//std::vector<uchar>
  cv::Scalar low_border = {0, 50, 0};
//std::vector<uchar>
  cv::Scalar high_border = {179, 255, 255};
  cv::inRange(hcv_input, low_border, high_border, mask);
  mask += black_mask;
  //cv::imshow("W", mask);
  //cv::waitKey();

  cv::bitwise_and(input, input, res_bgr, mask);
  //cv::bitwise_not(input, input, black_mask);
  //cv::imshow("W", input);
  //cv::waitKey();
  cv::cvtColor(input, hcv_input, cv::COLOR_BGR2HSV);
  cv::bitwise_and(hcv_input, hcv_input, res_hcv, mask);

  //cv::imshow("W", res_hcv);
  //cv::waitKey();
  //cv::imshow("W", res_bgr);
  //cv::waitKey();
  std::vector<std::vector<cv::Point>> contours;
  cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_ERODE, kernel);
  //cv::dilate(mask, mask, cv::MORPH_RECT);
//  cv::imshow("W", mask);
//  cv::waitKey();

  //cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  contours.erase(std::remove_if(contours.begin(),
                                contours.end(),
                                [rows = input.rows, cols = input.cols](const std::vector<cv::Point>& c1) {
                                  cv::RotatedRect r1 = cv::minAreaRect(c1);
                                  bool ret = false;
                                  ret |= r1.size.area() <= 200;
                                  ret |= r1.center.x <= 20 || r1.center.y <= 20 || r1.center.x >= cols - 20
                                      || r1.center.y >= rows - 20;
                                  return ret;

                                }), contours.end());
  std::vector<cv::Mat> mats(contours.size());
  std::vector<cv::Mat> gray_mats(contours.size());
  std::vector<cv::Mat> hsv_mats(contours.size());
  std::vector<cv::Mat> masks(contours.size());
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
    //  mats[i].create(mask.size(), CV_8UC3);
    //cv::boxPoints(min_enclosing, pts);
    min_enclosing.points(box.data());
    for (int i = 0; i < 4; ++i) {
      real_box[i].x = box[i].x + 1 - x_shift;
      real_box[i].y = box[i].y + 1 - y_shift;
    }
    cv::imshow("W", temp);
    //cv::waitKey();
    cv::fillConvexPoly(mask, real_box, cv::Scalar(255, 255, 255));
    //cv::imshow("W", mask);
    //cv::waitKey();
    cv::bitwise_and(temp, temp, mats[i], mask);
    cv::imshow("W", mats[i]);
//    cv::waitKey();
    cv::cvtColor(mats[i], gray_mats[i], cv::COLOR_RGB2GRAY);
    //cv::imshow("W", gray_mats[i]);
    //  cv::waitKey();
    cv::cvtColor(mats[i], hsv_mats[i], cv::COLOR_RGB2HSV);
    cv::threshold(gray_mats[i], masks[i], 1, 255, cv::THRESH_BINARY);
    cv::imshow("W", masks[i]);
    //   cv::waitKey();
  }
//  for (int i = 0; i < contours.size(); ++i) {
//    cv::rectangle(input, cv::boundingRect(contours[i]), cv::Scalar(255, 0, 0));
//  }
  std::cout << "CONTOURS.size: " << contours.size() << "\n";

//  cv::imshow("W", res_bgr);
//  cv::waitKey();
  DSU color_dsu = getColorDSU(mats, gray_mats, hsv_mats, masks);
  std::vector<cv::Scalar> colors = getColors(color_dsu);
  for (auto& c : colors) {
    std::cout << "color " << c << "\n";
  }
  for (int i = 0; i < contours.size(); ++i) {
    cv::drawContours(in_copy, contours, i, colors[i]);
  }
  cv::Mat gray_input;
  cv::cvtColor(res_bgr, gray_input, cv::COLOR_RGB2GRAY);
  matchAndDraw(gray_mats, res_bgr, gray_input, colors, in_copy);
  input = in_copy;
  std::cout << "FIN!\n";
 // cv::imshow("W", res_bgr);
//  cv::waitKey();
}
