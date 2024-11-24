import json
import os
import sys
import torch

from RawDataStorage.LocalRawDataStorage import LocalRawDataStorage

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Python')))

from pycandy import VectorDatabase
from Embedding.TextPreprocessor import TextPreprocessor

# Initialize the tensor database and embedding models
db = VectorDatabase()
rawDataStorage = LocalRawDataStorage()
text_preprocessor = TextPreprocessor()


class DBClient:
    def __init__(self):
        pass

    def get_tensor(self, query_text):
        query_embedding = text_preprocessor.generate_embedding(query_text)
        tensor = torch.from_numpy(query_embedding)
        results = db.search_tensor(tensor.clone(), 1)
        url = rawDataStorage.get_rawdata(results[0])
        print(f"RawData is in: {url}")

    def add_tensor(self, tensor_data):
        rowid = rawDataStorage.add_text_as_rawdata(tensor_data)
        rawDataStorage.displayRawData()
        embedding = text_preprocessor.generate_embedding(tensor_data)
        tensor = torch.from_numpy(embedding)
        db.insert_tensor_rawid(tensor.clone(), rowid)
        print(db.displayStore())

    def delete_tensor(self, tensor_data):
        embedding = text_preprocessor.generate_embedding(tensor_data)
        tensor = torch.from_numpy(embedding)
        success = db.delete_tensor(tensor.clone(),1)
        rawDataStorage.delete_rawdata(success[0])
        rawDataStorage.displayRawData()


def main():
    client = DBClient()
    while True:
        print("\nDBClient Interactive Console")
        print("1. Add Text Data")
        print("2. Get Vector by Text Query")
        print("3. Delete Text Data")
        print("4. Exit")
        choice = input("\nEnter your choice: ")

        if choice == "1":
            tensor_data = input("Enter text data to add: ")
            client.add_tensor(tensor_data)

        elif choice == "2":
            query_text = input("Enter text to query: ")
            client.get_tensor(query_text)

        elif choice == "3":
            tensor_data = input("Enter text data to delete: ")
            client.delete_tensor(tensor_data)

        elif choice == "4":
            rawDataStorage.delete_all_data()
            print("Exiting DBClient... Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
