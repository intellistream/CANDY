/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Core/vector_db.hpp>
#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <chrono>

// Mock search algorithm class for testing purposes
#include <cmath>       // For std::sqrt
#include <algorithm>   // For std::sort
#include <map>         // For std::map
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <c10/util/Logging.h>
TEST_CASE("Torch: Test basic functions") {
    auto ta = torch::rand({4, 4});
    std::cout << ta << std::endl;
    auto tb = torch::rand({2, 2});
    // Logging examples
    LOG(INFO) << "This is an INFO log message.";
    LOG(WARNING) << "This is a WARNING log message.";
    LOG(ERROR) << "This is an ERROR log message.";
    // torch::matmul(ta,tb);
    REQUIRE(ta.size(0) == 4);
}
