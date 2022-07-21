#include <iostream>
#include <algorithm>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


//May be used for color matching for resized or rotated matrixes
struct DSU {
  DSU(int size) : parents(size, -1){}
  int GetParent(int child_idx) {
    while(parents[child_idx] > 0) {
      child_idx = parents[child_idx];
    }
    return child_idx;
  }
  int Merge(int child1, int child2) {
    int parent1 = GetParent(child1), parent2 = GetParent(child2);
    if (parents[parent1] > parents[parent2]) {
      std::swap(parent2, parent1);
    }
    parents[parent1] += parents[parent2];
    parents[parent2] = parent1;

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

double getBestLocationValue(cv::Mat& result) {
  double minval, maxval;
  cv::Point minloc, maxloc;
  cv::minMaxLoc(result, &minval, &maxval, &minloc, &maxloc);
  return minval;
}

void matchOnePictureToMany(cv::Mat& templ, cv::Mat& grayscaled, cv::Mat& input) {
  cv::Mat result;
  int res_cols = grayscaled.cols - grayscaled.cols + 1;
  int res_rows = grayscaled.rows - grayscaled.rows + 1;
  result.create(res_rows, res_cols, CV_32FC1);
  cv::matchTemplate(grayscaled, templ, result, cv::TM_SQDIFF);
  cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
  double best_location_value = getBestLocationValue(result);
  auto locations = getBestLocations(result, best_location_value, 1e-3);
  cv::Scalar color = getColor();
  for (auto& location : locations) {
    cv::rectangle(input, location, cv::Point(location.x + templ.cols, location.y + templ.rows), color);
  }
 }

void matchAndDraw(std::vector<cv::Mat>& examples, cv::Mat& input, cv::Mat grayscaled) {
  for (int i = 0; i < examples.size(); ++i) {
    matchOnePictureToMany(examples[i], grayscaled, input);
  }
}


std::array<uchar, 256> getPrecounts(int mod) {
  std::array<uchar, 256> precounts;
  for (int i = 0; i < 256; ++i) {
    precounts[i] = i / mod * mod;
  }
  return precounts;
}

void processImage(cv::Mat& input) {
  std::vector<std::vector<cv::Point>> contours;
  cv::Mat src_gray;
  cv::cvtColor(input, src_gray, cv::COLOR_BGR2GRAY);
  cv::Mat grayscaled_copy = src_gray.clone();
  cv::threshold(src_gray, src_gray, 250, 255, cv::THRESH_BINARY);
  cv::Mat canny_out;
  cv::GaussianBlur(src_gray, src_gray, cv::Size(7, 7), 0, 0);
  cv::dilate(src_gray, src_gray, cv::MORPH_ELLIPSE);
  cv::Canny(src_gray, canny_out, 0, 400);
  cv::findContours(canny_out, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
  cv::Scalar color = cv::Scalar(255, 0, 0);
  std::vector<cv::Rect> bounds(contours.size());
  for (size_t i = 0; i < contours.size(); ++i) {
    bounds[i]= cv::boundingRect(contours[i]);
   // std::cout << bounds[i].tl() << " " << bounds[i].br() << "\n";
  }
  bounds.erase(std::unique(bounds.begin(), bounds.end(), [](const cv::Rect& p1, const cv::Rect & p2) {return p1.br().x == p2.br().x && p1.tl().x == p2.tl().x
  && p1.tl().y == p2.tl().y;}));
  std::vector<cv::Mat> images_on_white(bounds.size());
  for (int i = 0; i < images_on_white.size(); ++i) {
    images_on_white[i] = grayscaled_copy(bounds[i]);
  }
  matchAndDraw(images_on_white, input, grayscaled_copy);
}

int main() {
  cv::Mat image = cv::imread("../../images_for_study/test3.png", cv::IMREAD_COLOR);
  processImage(image);
  cv::namedWindow("W", cv::WINDOW_NORMAL);
  cv::imshow("W", image);
  cv::waitKey(0);
  return 0;
}
