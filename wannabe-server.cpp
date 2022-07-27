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

void fetchImage() {

}
std::optional<std::string> processConsumed(RdKafka::Message* prompt) {
  if (prompt->err()) {
    std::cerr << "Error! " << prompt->errstr()<<"\n";
    return std::nullopt;
  }

  if (prompt->err()) {
    std::cerr << "ERROR occured consuming msg " << prompt->errstr() << std::endl;
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
  }
  return "";

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
    std::optional<std::string> result = processConsumed(my_msg);
    std::cout << "Processing..." << std::endl;
    std::string return_msg = getReturnMsg();
    RdKafka::ErrorCode producer_error;
    do {
      producer->poll(0);
      producer_error = producer->produce(processed_topics, RdKafka::Topic::PARTITION_UA, RdKafka::Producer::RK_MSG_COPY,
                                         const_cast<char*>(return_msg.c_str()),
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
