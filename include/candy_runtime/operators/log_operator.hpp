#ifndef CANDY_LOG_OPERATOR_HPP
#define CANDY_LOG_OPERATOR_HPP

#include <candy_core/common/data_types.hpp>
#include <candy_core/utils/logging.hpp>
#include <candy_runtime/operators/base_operators.hpp>
#include <string>

namespace candy {

// Custom operator for logging VectorRecords
class LogOperator : public BaseOperator {
public:
  void open() override;  // Open the operator
  void close() override; // Close the operator
  void process(const std::shared_ptr<VectorRecord> &record);
};

} // namespace candy

#endif // CANDY_LOG_OPERATOR_HPP