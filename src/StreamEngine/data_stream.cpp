#include <StreamEngine/data_stream.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

// Constructor
DataStream::DataStream(const std::string &name, const std::string &sourcePath)
    : name(name), sourcePath(sourcePath) {
    loadData();
}

// Load data from source
void DataStream::loadData() {
    if (sourcePath.empty()) {
        throw std::runtime_error("Source path cannot be empty.");
    }

    std::ifstream file(sourcePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open source file: " + sourcePath);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        Vector vec;
        float value;
        while (stream >> value) {
            vec.push_back(value);
        }
        data.push_back(vec);
    }

    file.close();
    std::cout << "Loaded " << data.size() << " vectors from " << sourcePath << std::endl;
}

// Get the name of the stream
const std::string &DataStream::getName() const {
    return name;
}

// Filter transformation
std::shared_ptr<DataStream> DataStream::filter(std::function<bool(const Vector &)> predicate) {
    auto newStream = std::make_shared<DataStream>(name + "_filter", "");
    for (const auto &vec : data) {
        if (predicate(vec)) {
            newStream->data.push_back(vec);
        }
    }
    return newStream;
}

// Map transformation
std::shared_ptr<DataStream> DataStream::map(std::function<Vector(const Vector &)> mapper) {
    auto newStream = std::make_shared<DataStream>(name + "_map", "");
    for (const auto &vec : data) {
        newStream->data.push_back(mapper(vec));
    }
    return newStream;
}

// Top-K transformation
std::shared_ptr<DataStream> DataStream::topK(size_t k, std::function<float(const Vector &)> keyExtractor) {
    auto newStream = std::make_shared<DataStream>(name + "_topK", "");
    Operator::applyTopK(data, k, keyExtractor, newStream->data);
    return newStream;
}

// Join transformation
std::shared_ptr<DataStream> DataStream::join(const std::shared_ptr<DataStream> &other,
                                             std::function<bool(const Vector &, const Vector &)> joinPredicate) {
    auto newStream = std::make_shared<DataStream>(name + "_join", "");
    Operator::applyJoin(data, other->data, joinPredicate, newStream->data);
    return newStream;
}

// Sink: Write to output
void DataStream::writeSink(const std::string &sinkName, const std::string &outputPath) {
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open sink file: " + outputPath);
    }

    for (const auto &vec : data) {
        for (const auto &val : vec) {
            outFile << val << " ";
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "Written sink to " << outputPath << std::endl;
}

// Execute the stream
void DataStream::execute() {
    std::cout << "Executing stream: " << name << std::endl;
    // Placeholder: In a real implementation, this would trigger distributed or multi-threaded execution.
}
