#include <candy_vector_db/storage_engine.hpp>
#include <fstream>
#include <sstream>

namespace candy {

StorageEngine::StorageEngine(const std::string &storagePath)
    : storagePath(storagePath) {
  load();
}

void StorageEngine::add(const std::shared_ptr<VectorRecord> &record) {
  data[record->id] = record;
  writeToDisk(record->id, *record->data);
}

void StorageEngine::remove(const std::string &id) {
  data.erase(id);
  deleteFromDisk(id);
}

void StorageEngine::load() {
  std::ifstream inFile(storagePath + "/vectors.dat");
  std::string line;
  while (std::getline(inFile, line)) {
    std::istringstream iss(line);
    std::string id;
    iss >> id;

    VectorData vec;
    float value;
    while (iss >> value) {
      vec.push_back(value);
    }

    data[id] = std::make_shared<VectorRecord>(id, vec, 0);
  }
  inFile.close();
}

void StorageEngine::persist() {
  std::ofstream outFile(storagePath + "/vectors.dat");
  for (const auto &[id, record] : data) {
    outFile << id;
    for (const auto &val : *record->data) {
      outFile << " " << val;
    }
    outFile << "\n";
  }
  outFile.close();
}

void StorageEngine::writeToDisk(const std::string &id, const VectorData &vec) {
  std::ofstream outFile(storagePath + "/vectors.dat", std::ios::app);
  outFile << id;
  for (const auto &val : vec) {
    outFile << " " << val;
  }
  outFile << "\n";
  outFile.close();
}

void StorageEngine::deleteFromDisk(const std::string &id) {
  // Placeholder: Actual implementation would rebuild the file or mark the entry
  // as deleted.
}

} // namespace candy
