//
// Created by katar on 25.07.2022.
//
#include <rdkafkacpp.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <optional>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iterator>
#include <opencv2/imgproc.hpp>
#include <random>


//// Copypaste from main.cpp//////////////
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
    cv::rectangle(input, location, cv::Point(location.x + templ.cols, location.y + templ.rows), color);
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

DSU getColorDSU(std::vector<cv::Mat>& shapes,
                double max_dist1 = 0.001,
                double max_dist2 = 0.05,
                double max_dist3 = 0.001) {
  cv::Mat compare_shape1, compare_shape2;
  DSU color_matches(shapes.size());
  for (int i = 0; i < shapes.size(); ++i) {
    for (int j = i + 1; j < shapes.size(); ++j) {
      if (shapes[i].size < shapes[j].size) {
        compare_shape1 = shapes[i];
        compare_shape2 = shapes[j];
      } else {
        compare_shape1 = shapes[j];
        compare_shape2 = shapes[i];
      }
      double width = static_cast<double>(compare_shape2.cols) / compare_shape1.cols,
          height = static_cast<double>(compare_shape2.rows) / compare_shape1.rows;
      cv::resize(compare_shape1, compare_shape1, compare_shape2.size(), width, height);
      double dist1 = cv::matchShapes(compare_shape1, compare_shape2, cv::CONTOURS_MATCH_I1, 0);
      double dist2 = cv::matchShapes(compare_shape1, compare_shape2, cv::CONTOURS_MATCH_I2, 0);
      double dist3 = cv::matchShapes(compare_shape1, compare_shape2, cv::CONTOURS_MATCH_I3, 0);
      std::cout << dist1 << " " << dist2 << " " << dist3 << " " << "\n";
      if (dist2 <= max_dist2 && dist1 <= max_dist1 ||
          dist1 <= max_dist2 && dist3 <= max_dist3 ||
          dist2 <= max_dist2 && dist3 <= max_dist3) {
        color_matches.Merge(i, j);
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
      if (intersection == identified[i] || intersection == identified[j]) {
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
  cv::cvtColor(input, src_gray, cv::COLOR_BGR2GRAY);
  cv::Mat grayscaled_copy = src_gray.clone();
  cv::threshold(src_gray, src_gray, 250, 255, cv::THRESH_BINARY);
  cv::Mat canny_out;
  cv::GaussianBlur(src_gray, src_gray, cv::Size(7, 7), 0, 0);
  cv::dilate(src_gray, src_gray, cv::MORPH_ELLIPSE);
  cv::Canny(src_gray, canny_out, 0, 400);
  cv::findContours(canny_out, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
  cv::Scalar color = cv::Scalar(255, 0, 0);
  std::vector<cv::Rect> bounds(contours.size());
  for (size_t i = 0; i < contours.size(); ++i) {
    bounds[i] = cv::boundingRect(contours[i]);
  }
  bounds.erase(std::unique(bounds.begin(), bounds.end(), [](const cv::Rect& p1, const cv::Rect& p2) {
    return cv::norm(p1.tl() - p2.tl()) + cv::norm(p1.br() - p2.br()) <= 20;
  }), bounds.end());
  bounds = getRectanglesWithoutInside(bounds);
  std::vector<cv::Mat> images_on_white(bounds.size());
  for (int i = images_on_white[0].size == input.size ? 1 : 0; i < images_on_white.size(); ++i) {
    images_on_white[i] = grayscaled_copy(bounds[i]);
  }
  DSU color_parents = getColorDSU(images_on_white);
  auto colors = getColors(color_parents);
  for (int i = 0; i < colors.size(); ++i) {
    //ensure colors really different
    std::cout << "color [" << i << "] = " << colors[i] << "\n";
  }
  matchAndDraw(images_on_white, input, grayscaled_copy, colors);
}




void fetchImage(uchar* data_start, std::size_t size, cv::Mat& dest) {
  dest = cv::imdecode(cv::Mat(1, size * sizeof(uchar), CV_8UC1, data_start), cv::IMREAD_UNCHANGED);
}


bool processConsumed(RdKafka::Message* prompt, cv::Mat& image) {
  if (prompt->err()) {
    std::cerr << "ERROR occured consuming msg " << prompt->errstr() << std::endl;
    return false;
  } else {
    std::cout << "Success!\n";
    if (prompt->headers()) {
      RdKafka::Headers::Header last_id = prompt->headers()->get_last("msg_id");
      if (last_id.err()) {
        std::cout << "No header msg_id\n";
      } else {
        std::cout << "msg_id: " << *static_cast<const int*>(last_id.value()) << "\n";
      }
    }
    std::cout << "Consumed " << prompt->len() << " bytes. Starting processing\n";
    fetchImage(static_cast<uchar*>(prompt->payload()), prompt->len(), image);
    if (image.rows == 0 || image.cols == 0) {
      std::cout << "Can't decode from jpg!\n";
      return false;
    }
    processImage(image);
    return true;
  }
  return true;

}

class myProduceCallback2 : public RdKafka::DeliveryReportCb {
  void dr_cb(RdKafka::Message& msg) override {
    std::cout << msg.len() << " - length\n";
    if (msg.err()) {
      std::cerr << "Error occured! " << msg.errstr() << std::endl;
    } else {
      std::cout << "Success!\n";

      if (msg.headers()) {
        RdKafka::Headers::Header last_id = msg.headers()->get_last("msg_id");
        if (last_id.err()) {
          std::cout << "No header msg_id\n";
        } else {
          std::cout << "msg_id: " << *static_cast<const int*>(last_id.value()) << "\n";
        }
      }
      std::cout << "Delivered " << msg.len() << " bytes" << std::endl;
    }
  }
};

//for testing yet
std::string getReturnMsg() {
 std::ifstream fin("../../images_for_study/sw_reads.jpg", std::ios::binary);
 fin >> std::noskipws;
 std::string result((std::istream_iterator<char>(fin)), std::istream_iterator<char>());

 std::cout << "result length: " << result.size() << "\n";
  return result;

}

int main() {
  RdKafka::Conf* producer_conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  RdKafka::Conf* consumer_conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  std::string err;
  myProduceCallback2 pcb;
  producer_conf->set("bootstrap.servers", "localhost:9092", err);
  producer_conf->set("dr_cb", &pcb, err);
  producer_conf->set("message.max.bytes", "1000000000", err);
//  producer_conf->set("replica.fetch.max.bytes", "1000000000", err);
//  producer_conf->set("max.request.size", "1000000000", err);
  std::cout << err << "\n";
  consumer_conf->set("bootstrap.servers", "localhost:9092", err);
  consumer_conf->set("group.id", "12324", err);
  std::vector<std::string> raw_topics = {"folder1"};
  std::string processed_topics = "folder2";
  RdKafka::Producer* producer = RdKafka::Producer::create(producer_conf, err);
  RdKafka::KafkaConsumer* consumer = RdKafka::KafkaConsumer::create(consumer_conf, err);
  consumer->subscribe(raw_topics);
  while (true) {
    RdKafka::Message* my_msg = consumer->consume(1000);
    if (my_msg->err()) {
      delete my_msg;
      continue;
    }
    cv::Mat result;
    std::cout << "Processing..." << std::endl;
    if (!processConsumed(my_msg, result)){
      result = cv::imread("../../images_for_study/gyatime.jpg");
    };
    std::vector<uchar> return_msg;
    cv::imencode(".jpg", result, return_msg);
  //  std::string return_msg = getReturnMsg();
    RdKafka::ErrorCode producer_error;
    do {
      producer->poll(0);
      producer_error = producer->produce(processed_topics, RdKafka::Topic::PARTITION_UA, RdKafka::Producer::RK_MSG_COPY,
                                         const_cast<uchar*>(return_msg.data()),
                                         return_msg.size(),
                                         nullptr,
                                         0,
                                         std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()),
                                         my_msg->headers(),
                                         nullptr);
      if (producer_error == RdKafka::ErrorCode::ERR_NO_ERROR) {
        std::cout << "Response produced successfully!" << std::endl;
      } else {
        if (producer_error != RdKafka::ErrorCode::ERR__QUEUE_FULL) {
          std::cout << "Error!" << err2str(producer_error) << "\n";
        }
      }
      producer->poll(0);
    } while (producer_error == RdKafka::ERR__QUEUE_FULL);
   // delete my_msg;


  }

  return 0;
}
