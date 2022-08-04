//
// Created by katar on 29.07.2022.
//
#include <random>
//#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include "opencv_functional.h"

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

double getCornerDistance(cv::Rect& r1, cv::Rect& r2) {
  return cv::norm(r1.tl() - r2.tl()) + cv::norm(r1.br() - r2.br());
}

std::vector<cv::Point> getBestLocations(cv::Mat& result, double best_value, double threshold = 0) {
  std::vector<cv::Point> best_matches;
  cv::Mat_<float> doubled(result);
  for (int i = 0; i < doubled.rows; ++i) {
    for (int j = 0; j < doubled.cols; ++j) {
      if (doubled.at<float>(i, j) <= best_value + threshold) {
        best_matches.emplace_back(cv::Point(j, i));
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

void matchOnePictureToMany(cv::Mat& templ, cv::Mat& grayscaled, cv::Mat& input, const cv::Scalar& color) {
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
                  cv::Mat& input,
                  cv::Mat grayscaled,
                  const std::vector<cv::Scalar>& colors) {
  for (int i = 0; i < examples.size(); ++i) {
    if (getMatrixMin(examples[i]) < 250)
      matchOnePictureToMany(examples[i], grayscaled, input, colors[i]);
  }
}

std::tuple<double, double, double> getMatch(cv::Mat& shape1, cv::Mat& shape2) {
  double dist1 = cv::matchShapes(shape1, shape2, cv::CONTOURS_MATCH_I1, 0);
  double dist2 = cv::matchShapes(shape1, shape2, cv::CONTOURS_MATCH_I2, 0);
  double dist3 = cv::matchShapes(shape1, shape2, cv::CONTOURS_MATCH_I3, 0);
  return {dist1, dist2, dist3};
}

bool checkMatch(const std::tuple<double, double, double>& dists,
                double max_dist1 = 0.006,
                double max_dist2 = 0.05,
                double max_dist3 = 0.009) {
  auto& [dist1, dist2, dist3] = dists;
  std::cout << dist1 << " " << dist2 << " " << dist3 << " " << "\n";
  return dist2 <= max_dist2 && dist1 <= max_dist1 || dist1 <= max_dist2 && dist3 <= max_dist3
      || dist2 <= max_dist1 && dist3 <= max_dist3;
}

void drawAllContours(cv::Mat& img, const std::vector<std::vector<cv::Point>> contours) {
  for (int i = 0; i < contours.size(); ++i) {
    cv::drawContours(img, contours, i, cv::Scalar(255, 255, 255));
  }
}

void getMask(cv::Mat& img, cv::Mat& mask) {
  cv::cvtColor(img, mask, cv::COLOR_BGR2HSV);
  std::vector<uchar> lower = {0, 50, 0};
  std::vector<uchar> upper = {179, 255, 255};
  cv::inRange(mask, lower, upper, mask);
}

std::pair<double, double> compareHistogram(cv::Mat& img1, cv::Mat& img2) {
  cv::Mat mask1, mask2;
  getMask(img1, mask1);
  getMask(img2, mask2);
  cv::Mat hist1, hist2;
  int hist_sizes[] = {256, 256, 256};
  float range[] = {0.0, 256.0};
  const float* hist_ranges[] = {range, range, range};
  int channels[] = {0, 1, 2};
  cv::calcHist(&img1, 1, channels, mask1, hist1, 2, hist_sizes, hist_ranges);
  cv::calcHist(&img2, 1, channels, mask2, hist2, 2, hist_sizes, hist_ranges);
  cv::normalize(hist1, hist1, 1, 0, cv::NORM_MINMAX);
  cv::normalize(hist2, hist2, 1, 0, cv::NORM_MINMAX);
//  cv::namedWindow("w3", cv::WINDOW_NORMAL);
//  cv::namedWindow("w4", cv::WINDOW_NORMAL);
//  cv::imshow("w3", hist1);
//  cv::imshow("w4", hist2);
  //cv::waitKey();
  return {cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL), cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA)};
}

DSU getColorDSU(std::vector<cv::Mat>& shapes, std::vector<cv::Mat>& colored,
                double max_dist1 = 0.006,
                double max_dist2 = 0.05,
                double max_dist3 = 0.003) {
  cv::Mat compare_shape1, compare_shape2, colored1, colored2;
  cv::namedWindow("wi", cv::WINDOW_NORMAL);
  cv::namedWindow("wj", cv::WINDOW_NORMAL);
  DSU color_matches(shapes.size());
  for (int i = 0; i < shapes.size(); ++i) {
    //   cv::imshow("wi", shapes[i]);
    for (int j = i + 1; j < shapes.size(); ++j) {
      if (shapes[i].size < shapes[j].size) {
        compare_shape1 = shapes[i].clone();
        compare_shape2 = shapes[j].clone();
        colored1 = colored[i].clone();
        colored2 = colored[j].clone();
      } else {
        compare_shape1 = shapes[j].clone();
        compare_shape2 = shapes[i].clone();
        colored1 = colored[j].clone();
        colored2 = colored[i].clone();
      }
      //    cv::imshow("wj", shapes[j]);
      //  cv::waitKey();
      double width = static_cast<double>(compare_shape2.cols) / compare_shape1.cols,
          height = static_cast<double>(compare_shape2.rows) / compare_shape1.rows;
      cv::resize(compare_shape1, compare_shape1, compare_shape2.size(), width, height);
      cv::resize(colored1, colored1, colored2.size(), width, height);
      std::vector<std::vector<cv::Point>> contours_shape1, contours_shape2;
      cv::Mat thresholded1, thresholded2;
      cv::threshold(compare_shape1, thresholded1, 250, 255, cv::THRESH_BINARY_INV);
      cv::threshold(compare_shape2, thresholded2, 250, 255, cv::THRESH_BINARY_INV);


//      cv::findContours(thresholded1, contours_shape1, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
//      cv::findContours(thresholded2, contours_shape2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
//
//      cv::Mat c_shape1 = cv::Mat::zeros(compare_shape1.size(), CV_8UC1), c_shape2 = cv::Mat::zeros(compare_shape2.size(), CV_8UC1);
//      drawAllContours(c_shape1, contours_shape1);
//      drawAllContours(c_shape2, contours_shape2);
//      cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(compare_shape1.cols/64, compare_shape1.rows/64));
//      cv::morphologyEx(c_shape1, c_shape1, cv::MORPH_OPEN, kernel);
//      cv::morphologyEx(c_shape2, c_shape2, cv::MORPH_OPEN, kernel);
//      cv::imshow("wi", c_shape1);
//      cv::imshow("wj", c_shape2);
//      cv::waitKey();
      //     auto dists = getMatch(c_shape1, c_shape2);
      cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                                 cv::Size(std::max(compare_shape1.cols / 32, 5),
                                                          std::max(compare_shape1.rows / 32, 5)));
      cv::morphologyEx(thresholded1, thresholded1, cv::MORPH_OPEN, kernel);
      cv::morphologyEx(thresholded2, thresholded2, cv::MORPH_OPEN, kernel);
      cv::morphologyEx(thresholded1, thresholded1, cv::MORPH_CLOSE, kernel);
      cv::morphologyEx(thresholded2, thresholded2, cv::MORPH_CLOSE, kernel);
//      cv::imshow("wi", thresholded1);
//      cv::imshow("wj", thresholded2);
      //   cv::waitKey();
      auto dists = getMatch(thresholded1, thresholded2);
      if (checkMatch(dists, max_dist1 * 2, max_dist2 * 2, max_dist3 * 2)) {
        auto [c_hist_corr, c_hist_inter] = compareHistogram(colored[i], colored[j]);
        std::cout << "HISTOGRAM: " << c_hist_corr << " " << c_hist_inter << "\n";
        if (c_hist_corr >= 0.6 && c_hist_inter <= 0.7) {
          color_matches.Merge(i, j);
        }
      }
    }
  }
  return color_matches;
}

DSU getInnerRectangleDSU(std::vector<cv::Rect>& identified) {
  DSU rectangle_dsu(identified.size());
  for (int i = 0; i < identified.size(); ++i) {
    for (int j = i + 1; j < identified.size(); ++j) {
      auto intersection = identified[i] & identified[j];
      auto square_threshold = 0.25 * std::min(rectSquare(identified[i]), rectSquare(identified[j]));
      if (rectSquare(intersection) >= square_threshold) {
        rectangle_dsu.Merge(i, j);
      }
    }
  }
  return rectangle_dsu;
}

std::vector<cv::Rect> getRectanglesWithoutInside(std::vector<cv::Rect>& rectangles) {
  auto intersect_dsu = getInnerRectangleDSU(rectangles);
  std::unordered_map<int, cv::Rect> maximal_in_sets;
  for (int i = 0; i < rectangles.size(); ++i) {
    int parent = intersect_dsu.GetParent(i);
    if (!maximal_in_sets.count(parent)) {
      maximal_in_sets[parent] = rectangles[i];
    } else {
      maximal_in_sets[parent] = cv::Rect(cv::Point(std::min(maximal_in_sets[parent].tl().x, rectangles[i].tl().x),
                                                   std::min(maximal_in_sets[parent].tl().y, rectangles[i].tl().y)),
                                         cv::Point(std::max(maximal_in_sets[parent].br().x, rectangles[i].br().x),
                                                   std::max(maximal_in_sets[parent].br().y, rectangles[i].br().y)));
    }
  }
  std::vector<cv::Rect> maximal_bounding;
  maximal_bounding.reserve(maximal_in_sets.size());
  for (auto& [key, value]: maximal_in_sets) {
    maximal_bounding.emplace_back(value);
  }
  return maximal_bounding;
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
  std::vector<std::vector<cv::Point>> contours;
  cv::Mat src_gray;
  cv::floodFill(input,
                cv::Point(10, 10),
                cv::Scalar(255, 255, 255),
                nullptr,
                cv::Scalar(5, 5, 5),
                cv::Scalar(5, 5, 5));
  cv::cvtColor(input, src_gray, cv::COLOR_BGR2GRAY);
  cv::Mat grayscaled_copy = src_gray.clone();
  cv::threshold(src_gray, src_gray, 240, 255, cv::THRESH_BINARY);
  cv::Mat canny_out;
  //cv::GaussianBlur(src_gray, src_gray, cv::Size(3, 3), 0, 0);
  // cv::erode(src_gray, src_gray, cv::MORPH_ELLIPSE);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(src_gray, src_gray, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(src_gray, src_gray, cv::MORPH_CLOSE, kernel);
  cv::copyMakeBorder(input, input, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  cv::copyMakeBorder(src_gray, src_gray, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  cv::copyMakeBorder(grayscaled_copy, grayscaled_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

//
//  cv::Canny(src_gray, canny_out, 0, 200);
  cv::namedWindow("W1", cv::WINDOW_NORMAL);
  // cv::namedWindow("W2", cv::WINDOW_NORMAL);
  cv::imshow("W1", src_gray);
  // cv::imshow("W2", canny_out);
  cv::waitKey();
  cv::findContours(src_gray, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
  cv::Scalar color = cv::Scalar(255, 0, 0);
  std::vector<cv::Rect> bounds(contours.size());

  for (size_t i = 0; i < contours.size(); ++i) {
    bounds[i] = cv::boundingRect(contours[i]);
  }
  bounds.erase(std::unique(bounds.begin(), bounds.end(), [](const cv::Rect& p1, const cv::Rect& p2) {
    return cv::norm(p1.tl() - p2.tl()) + cv::norm(p1.br() - p2.br()) <= 20;
  }), bounds.end());
  bounds.erase(std::remove_if(bounds.begin(),
                              bounds.end(),
                              [matsize = cv::Point(grayscaled_copy.size())](const cv::Rect& r) {
                                return cv::norm(r.tl()) + cv::norm(r.br() - matsize) < 50 || rectSquare(r) < 100;
                              }), bounds.end());
  bounds = getRectanglesWithoutInside(bounds);
  std::vector<cv::Mat> images_on_white(bounds.size());
  std::vector<cv::Mat> colored_images(bounds.size());
  if (images_on_white.empty()) {
    std::cerr << "No images!";
    return;
  }
  for (int i = images_on_white[0].size == input.size ? 1 : 0; i < images_on_white.size(); ++i) {
    if (bounds[i].br().x >= images_on_white[i].cols || bounds[i].br().y >= images_on_white[i].rows) {
      std::cerr << "MORE!";
    }
    images_on_white[i] = grayscaled_copy(bounds[i]);
    colored_images[i] = input(bounds[i]);
  }
  DSU color_parents = getColorDSU(images_on_white, colored_images);
  auto colors = getColors(color_parents);
  for (int i = 0; i < colors.size(); ++i) {
    //ensure colors really different
    std::cout << "color [" << i << "] = " << colors[i] << "\n";
  }
  matchAndDraw(images_on_white, input, grayscaled_copy, colors);
}