# interactive_rag.py (Python script for interactive RAG interface)
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from pycandy import VectorDB
from Embedding.TextPreprocessor import TextPreprocessor

# Initialize the vector database and embedding models
search_algorithm = 'knnsearch'
db = VectorDB(128, search_algorithm)
# Initialize the vector database and embedding models
text_preprocessor = TextPreprocessor()

# Function to allow data ingestion by a user until they decide to stop
def data_ingestion():
    user_inputs = []
    while True:
        user_input = input("\n[Data Ingestion] Enter information to store (or type 'done' to finish):\n> ")
        if user_input.lower() == "done":
            print("\n[Data Ingestion] Finished data ingestion.")
            break
        else:
            user_inputs.append(user_input)

    for user_input in user_inputs:
        # Process text data
        embedding = text_preprocessor.generate_embedding(user_input)
        # Insert the embedding into the vector database
        db.insert_vector(embedding)
        print(f"\n[Data Ingestion] Successfully stored: '{user_input}'")

# Function to allow interactive querying by a user
def interactive_query():
    while True:
        user_query = input("\n[Interactive Query] Enter your query (or type 'exit' to quit):\n> ")
        if user_query.lower() in ["exit", "quit"]:
            print("\n[Interactive Query] Exiting...")
            break
        # Convert the numpy array to a list to ensure compatibility with pybind11
        query_embedding = text_preprocessor.generate_embedding(user_query).tolist()
        if not isinstance(query_embedding, list):
            query_embedding = list(query_embedding)
        results = db.query_nearest_vectors(query_embedding, k=3)
        print(f"\n[Interactive Query Results] Retrieved Nearest Vectors for '{user_query}':\n{results}\n")

# Main function to start data ingestion and querying
def main():
    # First, allow the user to input all data for ingestion
    data_ingestion()

    # Then, start interactive querying
    interactive_query()

if __name__ == "__main__":
    main()
