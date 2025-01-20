#include <StreamEngine/operators.hpp>
#include <fstream>
#include <algorithm>
#include <queue>
#include <iostream>

namespace Operator {

// Filter operation implementation
void applyFilter(const std::vector<Vector> &input,
                 std::function<bool(const Vector &)> predicate,
                 std::vector<Vector> &output) {
    for (const auto &vec : input) {
        if (predicate(vec)) {
            output.push_back(vec);
        }
    }
}

// Top-K operation implementation
void applyTopK(const std::vector<Vector> &input,
               size_t k,
               std::function<float(const Vector &)> keyExtractor,
               std::vector<Vector> &output) {
    // Min-heap to store top K elements
    auto comp = [&](const Vector &a, const Vector &b) {
        return keyExtractor(a) < keyExtractor(b);
    };
    std::priority_queue<Vector, std::vector<Vector>, decltype(comp)> minHeap(comp);

    for (const auto &vec : input) {
        minHeap.push(vec);
        if (minHeap.size() > k) {
            minHeap.pop(); // Remove smallest if heap exceeds size K
        }
    }

    // Extract elements from heap to output
    while (!minHeap.empty()) {
        output.push_back(minHeap.top());
        minHeap.pop();
    }

    // Reverse to maintain descending order
    std::reverse(output.begin(), output.end());
}

// Join operation implementation
void applyJoin(const std::vector<Vector> &left,
               const std::vector<Vector> &right,
               std::function<bool(const Vector &, const Vector &)> joinPredicate,
               std::vector<Vector> &output) {
    for (const auto &vecLeft : left) {
        for (const auto &vecRight : right) {
            if (joinPredicate(vecLeft, vecRight)) {
                Vector combined(vecLeft);
                combined.insert(combined.end(), vecRight.begin(), vecRight.end());
                output.push_back(combined);
            }
        }
    }
}

// Sink operation implementation
void writeToSink(const std::vector<Vector> &data, const std::string &filePath) {
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return;
    }

    for (const auto &vec : data) {
        for (size_t i = 0; i < vec.size(); ++i) {
            outFile << vec[i];
            if (i < vec.size() - 1) outFile << " ";
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "Data written to file: " << filePath << std::endl;
}

} // namespace Operator
