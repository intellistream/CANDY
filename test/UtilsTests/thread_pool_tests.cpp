/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/16
 * Description: [Provide description here]
 */
#include <Utils/thread_pool.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <vector>

using namespace std;

void test_func(int &result, int a, int b) { 
  this_thread::sleep_for(chrono::milliseconds(20));
  result = a + b;
}

TEST_CASE("ThreadPool submits and executes tasks correctly") {
  ThreadPool pool(4);  
  pool.init();

  SECTION("Submit tasks when thread pool is empty") {
    int res = 0;

    auto future = pool.submit(test_func, ref(res), 4, 6);
    future.get();

    REQUIRE(res == 10);
  }

  SECTION("Submit more tasks than threads available") { 
    vector<int> res(10, 0);

    vector<future<void>> futures;
    for (int i = 0; i < 10; i++) {
      futures.push_back(pool.submit(test_func, std::ref(res[i]), i, i));
    }

    for (auto &future : futures) { 
      future.get();
    }

    for (int i = 0; i < 10; i++) { 
      REQUIRE(res[i] == i + i);
    }
  }
    
  pool.shutdown();
}