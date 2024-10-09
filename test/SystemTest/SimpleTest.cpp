// Copyright (C) 2021 by the INTELLI team (https://github.com/intellistream)

#include <filesystem>
#include <gtest/gtest.h>
#include <Utils/Logger.hpp>

TEST(SystemTest, SimpleTest) {
  //Setup Logs.
  setupLogging("benchmark.log", LOG_DEBUG);

  //Run the test here.
  INTELLI_INFO("Nothing to test.");
}