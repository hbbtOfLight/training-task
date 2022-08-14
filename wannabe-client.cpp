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
#include <filesystem>

//#define DEBUG_RESULT
using namespace std::chrono_literals;

void saveToFileSystem(std::vector<uchar>& byte_data, const std::string& path) {
  try {
    cv::Mat frombytes_mat = cv::imdecode(cv::Mat(1, byte_data.size(), CV_8UC1, byte_data.data()), cv::IMREAD_UNCHANGED);
#ifdef DEBUG_RESULT
    cv::namedWindow("window", cv::WINDOW_NORMAL);
    cv::imshow("window", frombytes_mat);
    cv::waitKey();
#endif
      cv::imwrite(path, frombytes_mat);
  } catch (cv::Exception& e) {
    std::cerr << "Failed to decode image " << e.what() << "\n";
  } catch (...) {
    std::cerr << "Unknown exception\n";
  }
}

//std::vector<uchar> getJpegEncodedImage(cv::Mat& imgmat) {
//  std::vector<uchar> encoded;
//
//  return encoded;
//}

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

          std::cout << "msg_id: " << static_cast<const char*>(last_id.value()) << "\n";

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
        std::cout << "msg_id: " << static_cast<const char*>(last_id.value()) << "\n";
      }
    }
    std::cout << "Consumed " << msg->len() << " bytes. Starting processing\n";
    if (msg->len() == 0) {
      std::cerr << "Server returned empty msg!\n";
      return;
    }
    std::vector<uchar> uchared_msg(msg->len());
    uchar* buffer = (uchar*) (msg->payload());
    for (int i = 0; i < uchared_msg.size(); ++i) {
      uchared_msg[i] = buffer[i];
    }
    std::string msg_timestamp = std::to_string(msg->timestamp().timestamp);
    std::filesystem::path p("..");
    std::string file_name = "image-" + msg_timestamp + ".jpg";
    p /= "img";
    p /= file_name;
    saveToFileSystem(uchared_msg, p.string());
    std::cout << "Processed!" << std::endl;
  }
}

void KafkaProducerThread(bool& end, RdKafka::Producer* producer, const std::string& topic) {
  std::string error;
  std::string filepath;
  int msg_id = 0;
  while (!end) {
    getline(std::cin, filepath);
    RdKafka::ErrorCode err;
    if (filepath.empty()) {
      producer->poll(0);
      continue;
    }
    cv::Mat image_mat;
    std::vector<uchar> encoded_vector;
    try {
      image_mat = cv::imread(filepath);
      cv::imencode(".jpg", image_mat, encoded_vector);
    } catch (cv::Exception& ex) {
      std::cerr << "Exception reading file: " << ex.what() << "\n";
      continue;
    }

    std::cout << "Encoded matrix, img_size: " << encoded_vector.size() << "\n";
    std::string header = std::to_string(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())) + "-"
        + std::to_string(msg_id++);
    char* curr_msg_id = header.data();
    std::cout << curr_msg_id << "\n";;
    RdKafka::Headers::Header h("msg_id", curr_msg_id, sizeof(char) * header.size());
    RdKafka::Headers* msg_headers = RdKafka::Headers::create();
    msg_headers->add(h);
    do {
      err = producer->produce(topic,
                              RdKafka::Topic::PARTITION_UA,
                              RdKafka::Producer::RK_MSG_COPY,
                              const_cast<uchar*> (encoded_vector.data()),
                              encoded_vector.size() * sizeof(uchar),
                              nullptr,
                              0,
                              std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()),
                              msg_headers,
                              nullptr);
      if (err == RdKafka::ERR_NO_ERROR) {
        std::cerr << "Produced successfully!\n";
        producer->poll(0);
        break;
      } else {
        if (err != RdKafka::ERR__QUEUE_FULL) {
          std::cerr << "Error producing!:" << err2str(err) << "\n";
          delete msg_headers;
          break;
        }
      }
      std::cerr << "Queue full! Trying to resend...\n";
      producer->poll(1000);
    } while (err == RdKafka::ERR__QUEUE_FULL);

    producer->poll(0);
  }
  delete producer;
}

void KafkaConsumerThread(bool& end, RdKafka::Conf* configure_consumer, const std::vector<std::string>& topic_names) {
  std::string error;
  RdKafka::KafkaConsumer* cons = RdKafka::KafkaConsumer::create(configure_consumer, error);
  delete configure_consumer;
  if (!cons) {
    std::cerr << "Error creating consumer! " << error << std::endl;
    end = true;
    delete cons;
    return;
  }

  RdKafka::ErrorCode subscription_error = cons->subscribe(topic_names);
  if (subscription_error != RdKafka::ErrorCode::ERR_NO_ERROR) {
    std::cerr << "Can't subscribe on topic: " << RdKafka::err2str(subscription_error) << "\n";
    end = true;
  }

  while (!end) {
    RdKafka::Message* msg = cons->consume(1000);
    if (!msg->err()) {
      std::cerr << "Consumed!\n";
      processResponce(msg);
    } else {
      if (msg->err() != RdKafka::ErrorCode::ERR__TIMED_OUT) {
        std::cerr << "Error occured! " << msg->errstr() << "\n";
      }
    }
    delete msg;
  }
  delete cons;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Invalid call! Must specify topics for producer and consumer!";
    return -1;
  }
  std::string producer_topic_name = argv[1];
  std::string consumer_topic_name = argv[2];
  RdKafka::Conf* producer_config = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  RdKafka::Conf* consumer_config = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  std::string err;
  MyDeliveryCallback cb_prod;
  if (producer_config->set("bootstrap.servers", "localhost:9092", err) != RdKafka::Conf::CONF_OK ||
      producer_config->set("dr_cb", &cb_prod, err) != RdKafka::Conf::CONF_OK ||
      producer_config->set("message.max.bytes", "999999999", err) != RdKafka::Conf::CONF_OK ||
      consumer_config->set("bootstrap.servers", "localhost:9092", err) != RdKafka::Conf::CONF_OK ||
      consumer_config->set("group.id", "2", err) != RdKafka::Conf::CONF_OK ||
      consumer_config->set("message.max.bytes", "999999999", err) != RdKafka::Conf::CONF_OK ||
      consumer_config->set("max.partition.fetch.bytes", "999999999", err) != RdKafka::Conf::CONF_OK) {
    std::cerr << "Conf failed! " << err << "\n";
    delete producer_config;
    delete consumer_config;
    return 1;
  }
  RdKafka::Producer* prod = RdKafka::Producer::create(producer_config, err);
  if (!prod) {
    std::cerr << "Producer creation failed! " << err << "\n";
  }
  delete producer_config;
  RdKafka::Conf* topic_conf = RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC);

  RdKafka::Topic* processed_topic = RdKafka::Topic::create(prod, consumer_topic_name, topic_conf, err);
  if (!processed_topic) {
    std::cerr << err << "\n";
  }
  delete topic_conf;
  bool end = false;
  std::thread prod_thread(&KafkaProducerThread, std::ref(end), prod, producer_topic_name);

  std::vector<std::string> topics = {consumer_topic_name};
  std::thread cons_thread(&KafkaConsumerThread, std::ref(end), consumer_config, topics);
  delete processed_topic;
  cons_thread.join();
  prod_thread.join();
  return 0;
}