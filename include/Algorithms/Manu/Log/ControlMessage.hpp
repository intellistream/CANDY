//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_CONTROLMESSAGE_HPP
#define MANU_CONTROLMESSAGE_HPP

#include <string>

class ControlMessage {
public:
  std::string message;

  explicit ControlMessage(const std::string& message);
};

#endif // MANU_CONTROLMESSAGE_HPP
