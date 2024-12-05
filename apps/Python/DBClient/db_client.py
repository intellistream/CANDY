import json
import os
import sys
import torch
import logging
import signal

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Python')))

from pycandy import VectorDB
from Embedding.TextPreprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the tensor database and embedding models
search_algorithm = 'knnsearch'
db = VectorDB(128, search_algorithm)
text_preprocessor = TextPreprocessor()


class DBClient:
    def __init__(self):
        logging.info("Initializing DBClient")

    def get_tensor(self, query_text):
        try:
            query_embedding = text_preprocessor.generate_embedding(query_text)
            tensor = torch.from_numpy(query_embedding)
            results = db.query_nearest_tensors(tensor.clone(), k=1)
            if results:
                logging.info(f"Query successful for: {query_text}")
                return results[0]
            else:
                logging.warning(f"No results found for query: {query_text}")
                return None
        except Exception as e:
            logging.error(f"Error fetching tensor: {str(e)}")
            return None

    def add_tensor(self, tensor_data):
        try:
            embedding = text_preprocessor.generate_embedding(tensor_data)
            tensor = torch.from_numpy(embedding)
            db.insert_tensor(tensor.clone())
            logging.info(f"Successfully added tensor for data: {tensor_data}")
        except Exception as e:
            logging.error(f"Error adding tensor: {str(e)}")

    def update_tensor(self, old_data, new_data):
        try:
            old_embedding = text_preprocessor.generate_embedding(old_data)
            new_embedding = text_preprocessor.generate_embedding(new_data)
            old_tensor = torch.from_numpy(old_embedding)
            new_tensor = torch.from_numpy(new_embedding)
            db.update_tensor(old_tensor.clone(), new_tensor.clone())
            logging.info(f"Updated tensor: {old_data} -> {new_data}")
        except Exception as e:
            logging.error(f"Error updating tensor: {str(e)}")

    def delete_tensor(self, tensor_data):
        try:
            embedding = text_preprocessor.generate_embedding(tensor_data)
            tensor = torch.from_numpy(embedding)
            success = db.remove_tensor(tensor.clone())
            if success:
                logging.info(f"Successfully deleted tensor for data: {tensor_data}")
            else:
                logging.warning(f"Failed to delete tensor for data: {tensor_data}")
        except Exception as e:
            logging.error(f"Error deleting tensor: {str(e)}")


def signal_handler(sig, frame):
    logging.info("Exiting DBClient... Goodbye!")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
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
            print(f"Result: {tensor}")

        elif choice == "3":
            old_data = input("Enter existing text data to update: ")
            new_data = input("Enter new text data: ")
            client.update_tensor(old_data, new_data)

        elif choice == "4":
            tensor_data = input("Enter text data to delete: ")
            client.delete_tensor(tensor_data)

        elif choice == "5":
            logging.info("Exiting DBClient... Goodbye!")
            break

        else:
            logging.warning("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
