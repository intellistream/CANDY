import threading

from pyvectordb import VectorDB
from Python.Embedding.TextPreprocessor import TextPreprocessor

# Initialize the vector database and embedding models
search_algorithm = 'knnsearch'
db = VectorDB(128, search_algorithm)
# Initialize the vector database and embedding models
text_preprocessor = TextPreprocessor()

# Function to allow interactive data ingestion by a user
def interactive_data_ingestion():
    while True:
        user_input = input("Enter information to store (or type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        else:
            # Process text data
            embedding = text_preprocessor.generate_embedding(user_input)

        # Insert the embedding into the vector database
        db.insert_vector(embedding)
        print(f"[Data Ingestion] Stored: {user_input}")
        print(f"The embeddings: {embedding}")

# Function to allow interactive querying by a user
def interactive_query():
    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        # Convert the numpy array to a list to ensure compatibility with pybind11
        query_embedding = text_preprocessor.generate_embedding(user_query).tolist()
        if not isinstance(query_embedding, list):
            query_embedding = list(query_embedding)
        results = db.query_nearest_vectors(query_embedding, k=3)
        print(f"[Interactive Query Results] Retrieved: {results}")

# Main function to start the interactive data ingestion and querying
def main():
    # Start the data ingestion and querying in separate threads
    ingestion_thread = threading.Thread(target=interactive_data_ingestion)
    ingestion_thread.daemon = True
    ingestion_thread.start()

    # Start interactive querying in the main thread
    interactive_query()

if __name__ == "__main__":
    main()