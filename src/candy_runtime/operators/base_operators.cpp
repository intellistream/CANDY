#include <candy_runtime/operators/base_operators.hpp>
#include <iostream> // For logging and debugging

namespace candy {

void BaseOperator::process(const std::shared_ptr<VectorRecord>& record) {
    // Default implementation: Simply emit the record to the next operator
    std::cout << "[BaseOperator] Processing record ID: " << record->id << std::endl;
    emit(record);
}

} // namespace candy