import json
import os
import sys
import torch

from RawDataStorage.LocalRawDataStorage import LocalRawDataStorage

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Python')))

from pycandy import VectorDB
from Embedding.TextPreprocessor import TextPreprocessor

# Initialize the tensor database and embedding models
search_algorithm = 'knnsearch'
db = VectorDB(128, search_algorithm)
rawDataStorage = LocalRawDataStorage()
text_preprocessor = TextPreprocessor()


class DBClient:
    def __init__(self):
        pass

    def get_tensor(self, query_text):
        query_embedding = text_preprocessor.generate_embedding(query_text)
        tensor = torch.from_numpy(query_embedding)
        results = db.query_nearest_tensors(tensor.clone(), k=1)
        if results:
            return results[0]
        else:
            print(f"Error fetching tensor for query: '{query_text}'")
            return None

    def add_tensor(self, tensor_data):
        rowid = rawDataStorage.add_text_as_rawdata(tensor_data)
        embedding = text_preprocessor.generate_embedding(tensor_data)
        tensor = torch.from_numpy(embedding)
        db.insert_tensor_rawid(tensor.clone(), rowid)
        print(db.displayStore())
        #db.insert_tensor(tensor.clone())

    def update_tensor(self, old_data, new_data):
        old_embedding = text_preprocessor.generate_embedding(old_data)
        new_embedding = text_preprocessor.generate_embedding(new_data)
        old_tensor = torch.from_numpy(old_embedding)
        new_tensor = torch.from_numpy(new_embedding)
        db.update_tensor(old_tensor.clone(), new_tensor.clone())

    def delete_tensor(self, tensor_data):
        embedding = text_preprocessor.generate_embedding(tensor_data)
        tensor = torch.from_numpy(embedding)
        success = db.remove_tensor(tensor.clone())
        if not success:
            print(f"Error deleting tensor for data: '{tensor_data}'")


def main():
    client = DBClient()
    while True:
        print("\nDBClient Interactive Console")
        print("1. Add Text Data")
        print("2. Get Vector by Text Query")
        print("3. Update Existing Text Data")
        print("4. Delete Text Data")
        print("5. Exit")
        choice = input("\nEnter your choice: ")

        if choice == "1":
            tensor_data = input("Enter text data to add: ")
            client.add_tensor(tensor_data)

        elif choice == "2":
            query_text = input("Enter text to query: ")
            tensor = client.get_tensor(query_text)
            print(f"tensor: {tensor}")
           

        elif choice == "3":
            old_data = input("Enter existing text data to update: ")
            new_data = input("Enter new text data: ")
            client.update_tensor(old_data, new_data)

        elif choice == "4":
            tensor_data = input("Enter text data to delete: ")
            client.delete_tensor(tensor_data)

        elif choice == "5":
            print("Exiting DBClient... Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
