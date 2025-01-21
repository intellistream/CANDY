#include <candy_runtime/operators/log_operator.hpp>

namespace candy {

void LogOperator::open() { INTELLI_INFO("LogOperator opened."); }

void LogOperator::close() { INTELLI_INFO("LogOperator closed."); }

void LogOperator::process(const std::shared_ptr<VectorRecord> &record) {
  if (!record || !record->data) {
    INTELLI_ERROR("Invalid VectorRecord received.");
    return;
  }
  INTELLI_INFO("Processing VectorRecord: ID=" + record->id + ", Timestamp=" +
               std::to_string(record->timestamp) + ", Data=" + [&]() {
                   std::string dataStr;
                   size_t count = 0;
                   for (const auto &val : *record->data) {
                       if (++count > 10) {
                           dataStr += "... (truncated)";
                           break;
                       }
                       dataStr += std::to_string(val) + " ";
                   }
                   return dataStr;
               }());
}


} // namespace candy
