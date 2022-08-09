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

#include "opencv_functional_v2.h"


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


bool getConfigFromFile(RdKafka::Conf* configuration, const std::string& filepath, const std::string& key) {
  cv::FileStorage fs(filepath, cv::FileStorage::READ);
  if (!fs.isOpened()) {
   std::cerr << "Cannot open config file!";
   return false;
  }
  cv::FileNode producer_node = fs[key];
  std::string error;
  for (cv::FileNode n : producer_node) {
    std::string property = n.name(), value = static_cast<std::string>(producer_node[property]);
    if (configuration->set(property, value, error) != RdKafka::Conf::CONF_OK) {
      std::cerr << "Error setting configuration: " << error << "\n";
    }
    std::cout << property << ":" << value  << "\n";
  }
  return true;
}

std::unordered_map<std::string, std::string> getTopicMap(const std::string& filepath, std::vector<std::string>& raw_topics) {
  std::unordered_map<std::string, std::string> topic_map;
  cv::FileStorage fs (filepath, cv::FileStorage::READ);
  cv::FileNode topics = fs["topic_map"];
  for (cv::FileNode topic_node : topics) {
    std::string consumer_topic = topic_node.name(), producer_topic = static_cast<std::string>(topics[consumer_topic]);
    topic_map[consumer_topic] = producer_topic;
    raw_topics.emplace_back(consumer_topic);
    std::cout << consumer_topic << ":" << producer_topic << "\n";
  }
  return topic_map;
}

int main() {
  RdKafka::Conf* producer_conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  RdKafka::Conf* consumer_conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  //writeFile();
  if (!getConfigFromFile(producer_conf, "../config.yaml", "producer")) {
    std::cerr << "Error while reading producer config!\n";
    exit(1);
  }
  if (!getConfigFromFile(consumer_conf, "../config.yaml", "consumer")) {
    std::cerr << "Error while reading consumer config!\n";
    exit(1);
  }
  std::vector<std::string> raw_topics;
  auto topic_map = getTopicMap("../config.yaml", raw_topics);


  std::string err;
  myProduceCallback2 pcb;
  RdKafka::Producer* producer = RdKafka::Producer::create(producer_conf, err);
  RdKafka::KafkaConsumer* consumer = RdKafka::KafkaConsumer::create(consumer_conf, err);
  if (!producer || !consumer) {
    std::cerr << "Error creating producer and consumer! " << err << "\n";
    exit(1);
  }
//  std::vector<RdKafka::Topic*> topics(raw_topics.size());
//  RdKafka::Conf* topic_conf = RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC);
//  for (int i = 0; i < topics.size(); ++i) {
//    topics[i] = RdKafka::Topic::create(producer, raw_topics[i], topic_conf, err);
//  }
//  for (int i = 0; i < topics.size(); ++i) {
//    delete topics[i];
//  }
  consumer->subscribe(raw_topics);
  while (true) {
    RdKafka::Message* my_msg = consumer->consume(1000);
    if (my_msg->err() || my_msg->len() == 0) {
      delete my_msg;
      continue;
    }
    std::string processed_topics = topic_map[my_msg->topic_name()];
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

  }

  return 0;
}
