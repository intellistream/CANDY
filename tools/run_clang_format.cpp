
#include <argparse.hpp>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int k_max_threads = std::thread::hardware_concurrency();
namespace fs = std::filesystem;

void Run(const std::string& clang_format_binary,
         const std::vector<std::string>& souce_dirs) {
  std::vector<std::string> source_files;
  for (const auto& source_dir : souce_dirs) {
    for (const auto& entry : fs::recursive_directory_iterator(source_dir)) {
      if (entry.is_directory()) {
        continue;
      }
      if (const auto& path = entry.path();
          path.extension() == ".cpp" || path.extension() == ".c" ||
          path.extension() == ".h" || path.extension() == ".hpp") {
        source_files.push_back(path.string());
      }
    }
  }
  std::shuffle(source_files.begin(), source_files.end(),
               std::mt19937(std::random_device()()));
  std::vector<std::string> commands(k_max_threads);
  for (auto& command : commands) {
    command = clang_format_binary + " -i";
  }
  for (size_t i = 0; i < source_files.size(); i++) {
    auto& source_file = source_files[i];
    auto& command = commands[i % commands.size()];
    command += " " + source_file;
  }
  std::vector<std::future<int>> futures;
  futures.reserve(commands.size());
  for (auto& command : commands) {
    command += " > /dev/null 2>&1";
    futures.push_back(std::async(
        std::launch::async,
        [](const std::string& c) { return std::system(c.c_str()); }, command));
  }
  for (auto& future : futures) {
    future.get();
  }
}

auto main(int argc, char* argv[]) -> int {
  argparse::ArgumentParser program("run_format");
  program.add_argument("clang_format_binary")
      .help("Path to clang-format binary");
  program.add_argument("--source_dirs")
      .default_value(std::vector<std::string>{})
      .remaining()
      .help("Directories to exclude from formatting");
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << '\n';
    std::cout << program;
    return 1;
  }
  Run(program.get<std::string>("clang_format_binary"),
      program.get<std::vector<std::string>>("source_dirs"));
  return 0;
}