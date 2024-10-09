// Copyright (C) 2021 by the IntelliStream team (https://github.com/intellistream)

/**
 * @brief This is the main entry point of the entire program.
 * We use this as the entry point for benchmarking.
 */
#include <Utils/Logger.hpp>

using namespace std;

int main(int argc, char **argv) {
  //Setup Logs.
  setupLogging("benchmark.log", LOG_DEBUG);

  //Run the test here.
  INTELLI_INFO("Nothing to run." << argc << argv);
}

