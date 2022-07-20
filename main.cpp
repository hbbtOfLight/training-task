#include <iostream>
#include <algorithm>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

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

std::vector<cv::Point> getBestLocations(cv::Mat& result, double threshold) {
//`  std::cout << result << "\n";
  std::vector<cv::Point> best_matches;
  cv::Mat_<float> doubled(result);
  for (int i = 0; i < doubled.rows; ++i) {
    for (int j = 0; j < doubled.cols; ++j) {
      try {
        if (doubled.at<float>(i, j) < threshold) {
          best_matches.emplace_back(cv::Point(i, j));
        }
      } catch (...) {
        std::cout << "OOps\n";
      }
    }
  }
  return best_matches;
}

void matchOnePictureToMany(cv::Mat& templ, cv::Mat& grayscaled, cv::Mat& input) {
  cv::Mat result;
  int res_cols = grayscaled.cols - templ.cols + 1;
  int res_rows = grayscaled.rows - templ.rows + 1;
  result.create(res_rows, res_cols, CV_32FC1);
  cv::matchTemplate(grayscaled, templ, result, cv::TM_SQDIFF_NORMED);
 // std::cout << result.row(0);
  cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
//  std::cout << result << "\n";
//  cv::namedWindow("w2");
//  cv::imshow("w2", result);
//  cv::waitKey();
  auto locations = getBestLocations(result, 0.0001);

  std::mt19937 gen(time(0));
  std::uniform_int_distribution<uchar> dis(0, 255);
  cv::Scalar color (dis(gen), dis(gen), dis(gen));
  std::cout << "Color = " << color << "\n";
  for (auto& location : locations) {
    cv::rectangle(input, location, cv::Point(location.x + templ.cols, location.y + templ.rows), color);
  }
  cv::namedWindow("w1", cv::WINDOW_NORMAL);
  cv::imshow("w1", input);
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
int countImages(cv::Mat& input) {
  std::vector<std::vector<cv::Point>> contours;
  cv::Mat src_gray;
  cv::cvtColor(input, src_gray, cv::COLOR_BGR2GRAY);
  cv::Mat grayscaled_copy = src_gray.clone();
  cv::threshold(src_gray, src_gray, 250, 255, cv::THRESH_BINARY);
  cv::Mat canny_out;
  auto precounts = getPrecounts(255);
  cv::LUT(src_gray, precounts, src_gray);
  cv::GaussianBlur(src_gray, src_gray, cv::Size(7, 7), 0, 0);
  cv::dilate(src_gray, src_gray, cv::MORPH_ELLIPSE);
  cv::Canny(src_gray, canny_out, 0, 400);
  cv::namedWindow("W", cv::WINDOW_NORMAL);
  cv::imshow("W", src_gray);
  cv::waitKey(0);
  cv::findContours(canny_out, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
  std::vector<cv::Mat> images_on_white(contours.size());
  cv::Scalar color = cv::Scalar(255, 0, 0);
  for (size_t i = 0; i < contours.size(); ++i) {
    cv::Rect curr = cv::boundingRect(contours[i]);
    images_on_white[i] = cv::Mat(grayscaled_copy, curr);
    std::cout << curr.tl() << " " << curr.br() << "\n";
    //cv::rectangle(input, curr.tl(), curr.br(), color);
  }
  matchAndDraw(images_on_white, input, grayscaled_copy);
//  int cmpop;
//  cv::Mat dst;
 // cv::compare(images_on_white[0], images_on_white[2], dst, cv::CMP_EQ);
 // std::cout << cv::countNonZero(dst) << "\n";

//  for (int i = 0; i < images_on_white.size(); ++i) {
//    colors[i] = cv::Scalar (dis(gen), dis(gen), dis(gen));
//  }
//  for (int i = 0; i < images_on_white.size(); ++i) {
//
//  }
  cv::imshow("W", input);
  cv::waitKey(0);

  return images_on_white.size();
}

int main() {
  cv::Mat image = cv::imread("../../images_for_study/test2.png", cv::IMREAD_COLOR);
  std::cout << countImages(image);
  return 0;
}
