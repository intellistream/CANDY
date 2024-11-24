import os
import json
import threading

class LocalRawDataStorage:
    def __init__(self, storage_file="/home/candy/CANDY/apps/Python/RawData/raw_data.json", data_dir="/home/candy/CANDY/apps/Python/RawData/raw_data_files"):
        self.storage_file = storage_file
        self.data_dir = data_dir
        self.data = {}
        self.next_id = 0
        self.lock = threading.Lock()  # 锁对象

        # 确保数据文件目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if os.path.exists(self.storage_file):
            self._load_storage()
        else:
            print(f"Storage file {self.storage_file} not found. Creating a new one.")
            self._save_storage()

    def _load_storage(self):
        """从存储文件中加载数据"""
        with open(self.storage_file, "r") as f:
            storage = json.load(f)
            self.data = storage.get("data", {})
            self.next_id = storage.get("next_id", 1)

    def _save_storage(self):
        """将当前数据保存到存储文件"""
        try:
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            with open(self.storage_file, "w") as f:
                json.dump({"data": self.data, "next_id": self.next_id}, f, indent=4)
            print(f"Data saved successfully to {self.storage_file}")
        except Exception as e:
            print(f"Error saving storage file: {e}")

    def add_text_as_rawdata(self, text):
        """
        将用户输入的文本保存为文件，并注册为 RawData。
        Args:
            text (str): 用户输入的文本
        Returns:
            raw_id (int): 分配的 RawID
        """
        with self.lock:  # 确保分配 ID 和写操作线程安全
            raw_id = self.next_id
            self.next_id += 1

        file_name = f"raw_{raw_id}.txt"
        file_path = os.path.join(self.data_dir, file_name)

        try:
            # 保存文本到文件
            with open(file_path, "w") as f:
                f.write(text)

            # 更新存储数据
            with self.lock:
                self.data[raw_id] = file_path
                self._save_storage()

            print(f"Text saved as RawData with ID {raw_id}: {file_path}")
            return raw_id
        except Exception as e:
            print(f"Error saving text as RawData: {e}")
            return None

    def get_rawdata(self, raw_id):
        """根据 RawID 获取对应文件路径"""
        with self.lock:  # 确保读取操作线程安全
            return self.data.get(raw_id, None)

if __name__ == "__main__":
    storage = LocalRawDataStorage()
    while True:
        print("\nRawDataStorage Interactive Console")
        print("1. Add Text Data as RawData")
        print("2. Get RawData by ID")
        print("3. Exit")
        choice = input("\nEnter your choice: ")

        if choice == "1":
            text = input("Enter text data: ")
            storage.add_text_as_rawdata(text)
        elif choice == "2":
            raw_id = int(input("Enter RawData ID: "))
            file_path = storage.get_rawdata(raw_id)
            if file_path:
                print(f"RawData file path: {file_path}")
            else:
                print(f"RawData with ID {raw_id} not found.")
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")