#ifndef CANDY_BASE_OPERATORS_HPP
#define CANDY_BASE_OPERATORS_HPP

#include <candy_core/common/data_types.hpp> // Include VectorRecord definition
#include <memory>
#include <mutex>
#include <queue>

namespace candy {

// Base class for all operators
class BaseOperator {
protected:
  std::shared_ptr<BaseOperator>
      next_operator_; // Pointer to the next operator in the pipeline
  std::shared_ptr<std::queue<std::shared_ptr<VectorRecord>>>
      input_queue_;        // Queue of VectorRecords as input
  std::mutex queue_mutex_; // Mutex for thread-safe queue operations

public:
  virtual ~BaseOperator() = default;

  // Open the operator (e.g., initialization logic)
  virtual void open() {
    // Default implementation: No-op
  }

  // Close the operator (e.g., cleanup logic)
  virtual void close() {
    // Default implementation: No-op
  }

  // Process a single VectorRecord
  virtual void process(const std::shared_ptr<VectorRecord> &record);

  // Emit processed data to the next operator in the pipeline
  virtual void emit(const std::shared_ptr<VectorRecord> &record) {
    if (next_operator_) {
      next_operator_->enqueue(record);
    }
  }

  // Set the next operator in the pipeline
  void set_next_operator(const std::shared_ptr<BaseOperator> &next) {
    next_operator_ = next;
  }

  // Set the input queue for this operator
  void set_input_queue(
      const std::shared_ptr<std::queue<std::shared_ptr<VectorRecord>>> &queue) {
    input_queue_ = queue;
  }

  // Add data to the input queue
  void enqueue(const std::shared_ptr<VectorRecord> &data) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (input_queue_) {
      input_queue_->push(data);
    }
  }

  // Pull data from the input queue and process it
  virtual void process_queue() {
    while (true) {
      std::shared_ptr<VectorRecord> data;

      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (input_queue_ && !input_queue_->empty()) {
          data = input_queue_->front();
          input_queue_->pop();
        } else {
          break; // Exit if queue is empty
        }
      }

      if (data) {
        process(data); // Pass shared_ptr directly
      }
    }
  }
};

} // namespace candy

#endif // CANDY_BASE_OPERATORS_HPP
