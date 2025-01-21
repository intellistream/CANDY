#ifndef CANDY_ERROR_CODES_HPP
#define CANDY_ERROR_CODES_HPP

#include <string>

namespace candy {

enum class ErrorCode {
  SUCCESS,
  INVALID_ARGUMENT,
  OPERATOR_FAILURE,
  STATE_FAILURE
};

std::string error_to_string(ErrorCode code) {
  switch (code) {
  case ErrorCode::SUCCESS: return "SUCCESS";
  case ErrorCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
  case ErrorCode::OPERATOR_FAILURE: return "OPERATOR_FAILURE";
  case ErrorCode::STATE_FAILURE: return "STATE_FAILURE";
  default: return "UNKNOWN_ERROR";
  }
}

} // namespace candy

#endif // CANDY_ERROR_CODES_HPP
