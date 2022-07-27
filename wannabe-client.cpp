//
// Created by katar on 24.07.2022.
//

#include <rdkafkacpp.h>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

void saveToFileSystem(std::vector<char>& byte_data, const std::string& path) {
  std::cout << "called!\n";
  std::cout << byte_data.size() << '\n';
  cv::Mat frombytes_mat = cv::imdecode(cv::Mat(1, byte_data.size(), CV_8UC1, byte_data.data()), cv::IMREAD_UNCHANGED);
  std::cout << frombytes_mat.size << "\n";
  cv::namedWindow("window", cv::WINDOW_NORMAL);
  cv::imshow("window", frombytes_mat);
  cv::waitKey();
  if (frombytes_mat.rows != 0) {
    cv::imwrite(path, frombytes_mat);
  }
}

class MyDeliveryCallback : public RdKafka::DeliveryReportCb {
  void dr_cb(RdKafka::Message& msg) override {
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

void processResponce(RdKafka::Message* msg) {
  if (msg->err()) {
    std::cerr << "ERROR occured consuming msg " << msg->errstr() << std::endl;
  } else {
    std::cout << "Success!\n";
    if (msg->headers()) {
      RdKafka::Headers::Header last_id = msg->headers()->get_last("msg_id");
      if (last_id.err()) {
        std::cout << "No header msg_id\n";
      } else {
        std::cout << "msg_id: " << *static_cast<const int*>(last_id.value()) << "\n";
      }
    }
    std::cout << "Consumed " << msg->len() << " bytes. Starting processing\n";
   /// std::cout << static_cast<const char*>(msg->payload());
   std::vector<char> uchared_msg(msg->len());
   char* buffer = (char*)(msg->payload());
   for (int i = 0; i < uchared_msg.size(); ++i) {
     uchared_msg[i] = buffer[i];
     std::cout << uchared_msg[i];
   }
   std::cout << "\n";
    std::string msg_timestamp = std::to_string(msg->timestamp().timestamp);
    saveToFileSystem(uchared_msg, "../img/image-" + msg_timestamp + ".jpg");
    std::cout << "Processed!" << std::endl;
  }
}

void KafkaProducerThread(bool& end, RdKafka::Conf* producer_config, const std::string& topic) {
  std::string error;
  RdKafka::Producer* producer = RdKafka::Producer::create(producer_config, error);
  delete producer_config;
  if (!producer) {
    std::cerr << "ERROR creating producer! " << error << std::endl;
    end = true;
    return;
  }
  std::string filepath;
  int msg_id = 0;

  while (!end) {
    getline(std::cin, filepath);
    //process file I don't understand how yet

    RdKafka::ErrorCode err;
    if (filepath.empty()) {
      producer->poll(0);
      continue;
    }
    int* curr_msg_id = new int(msg_id++);
    RdKafka::Headers::Header h("msg_id", curr_msg_id, sizeof(msg_id));
    RdKafka::Headers* msg_headers = RdKafka::Headers::create();
    msg_headers->add(h);
    do {
      err = producer->produce(topic,
                              RdKafka::Topic::PARTITION_UA,
                              RdKafka::Producer::RK_MSG_COPY,
                              const_cast<char*> (filepath.c_str()),
                              filepath.size(),
                              nullptr,
                              0,
                              std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()),
                              msg_headers,
                              nullptr);
      if (err == RdKafka::ERR_NO_ERROR) {
        std::cerr << "Produced successfully!\n";
        producer->poll(0);
      } else {

        if (err != RdKafka::ERR__QUEUE_FULL) {
          std::cerr << "Error producing!\n";
          delete msg_headers;
          break;
        }
      }
      std::cerr << "Queue full! Trying to resend...\n";
      producer->poll(1000);
    } while (err == RdKafka::ERR__QUEUE_FULL);
    delete curr_msg_id;
    producer->poll(0);
  }
  delete producer;
}

void KafkaConsumerThread(bool& end, RdKafka::Conf* configure_consumer, const std::vector<std::string>& topic_names) {
  std::string error;
  RdKafka::KafkaConsumer* cons = RdKafka::KafkaConsumer::create(configure_consumer, error);
  if (!cons) {
    std::cerr << "Error creating consumer! " << error << std::endl;
    end = true;
    delete cons;
    return;
  }
  cons->subscribe(topic_names);
  while (!end) {
    RdKafka::Message* msg = cons->consume(1000);
    if (!msg->err()) {
      std::cerr << "Consumed!\n";
      processResponce(msg);
      std::cerr << static_cast<char*>(msg->payload()) << "\n";
    } else {
      if (msg->err() != RdKafka::ErrorCode::ERR__TIMED_OUT) {
        std::cerr << "Error occured! " << msg->errstr() << "\n";
      }
    }
    delete msg;

  }
  delete cons;
}
int main() {
  RdKafka::Conf* producer_config = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  RdKafka::Conf* consumer_config = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  std::string err;
  MyDeliveryCallback cb_prod;
  if (producer_config->set("bootstrap.servers", "localhost:9092", err) != RdKafka::Conf::CONF_OK ||
      producer_config->set("dr_cb", &cb_prod, err) != RdKafka::Conf::CONF_OK ||
      consumer_config->set("bootstrap.servers", "localhost:9092", err) != RdKafka::Conf::CONF_OK ||
      consumer_config->set("group.id", "12321", err) != RdKafka::Conf::CONF_OK ||
      consumer_config->set("message.max.bytes", "1000000000", err) != RdKafka::Conf::CONF_OK //||
     /* consumer_config->set("replica.fetch.message.max.bytes", "1000000000", err) != RdKafka::Conf::CONF_OK*/) {
    std::cerr << "Conf failed! " << err << "\n";
    return 1;
  }
////configure that????!!! From what?
  bool end = false;
  std::thread prod_thread(&KafkaProducerThread, std::ref(end), producer_config, "folder1");
  std::vector<std::string> topics = {"folder2"};

  std::thread cons_thread(&KafkaConsumerThread, std::ref(end), consumer_config, topics);
  cons_thread.join();
  prod_thread.join();
  delete producer_config;
  delete consumer_config;
  return 0;
}