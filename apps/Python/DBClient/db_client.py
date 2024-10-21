import json
import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Python')))

from pycandy import VectorDB
from Embedding.TextPreprocessor import TextPreprocessor

# Initialize the vector database and embedding models
search_algorithm = 'knnsearch'
db = VectorDB(128, search_algorithm)
text_preprocessor = TextPreprocessor()


class DBClient:
    def __init__(self):
        pass

    def get_vector(self, query_text):
        query_embedding = text_preprocessor.generate_embedding(query_text)
        results = db.query_nearest_vectors(query_embedding, k=1)
        if results:
            return results[0]
        else:
            print(f"Error fetching vector for query: '{query_text}'")
            return None

    def add_vector(self, vector_data):
        embedding = text_preprocessor.generate_embedding(vector_data)
        db.insert_vector(embedding)
        print(f"Vector added successfully. Data: {vector_data}")

    def update_vector(self, old_data, new_data):
        old_embedding = text_preprocessor.generate_embedding(old_data)
        results = db.query_nearest_vectors(old_embedding, k=1)
        if results:
            vector_id = results[0]['id']
            new_embedding = text_preprocessor.generate_embedding(new_data)
            success = db.update_vector(vector_id, new_embedding)
            if success:
                print(f"Vector updated successfully.")
            else:
                print(f"Error updating vector with ID {vector_id}")
        else:
            print(f"Error finding vector to update for data: '{old_data}'")

    def delete_vector(self, vector_data):
        embedding = text_preprocessor.generate_embedding(vector_data)
        results = db.query_nearest_vectors(embedding, k=1)
        if results:
            vector_id = results[0]['id']
            success = db.delete_vector(vector_id)
            if success:
                print(f"Vector with ID {vector_id} deleted successfully.")
            else:
                print(f"Error deleting vector with ID {vector_id}")
        else:
            print(f"Error finding vector to delete for data: '{vector_data}'")


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
            vector_data = input("Enter text data to add: ")
            client.add_vector(vector_data)

        elif choice == "2":
            query_text = input("Enter text to query: ")
            vector = client.get_vector(query_text)
            if vector:
                print(json.dumps(vector, indent=2))

        elif choice == "3":
            old_data = input("Enter existing text data to update: ")
            new_data = input("Enter new text data: ")
            client.update_vector(old_data, new_data)

        elif choice == "4":
            vector_data = input("Enter text data to delete: ")
            client.delete_vector(vector_data)

        elif choice == "5":
            print("Exiting DBClient... Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
