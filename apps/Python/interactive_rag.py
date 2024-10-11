import threading
import time
import random
from py_vector_db import VectorDB
from data_preprocessing.text_preprocessor import TextPreprocessor
from data_preprocessing.audio_preprocessor import AudioPreprocessor

# Initialize the vector database and embedding models
db = VectorDB(128, search_algorithm)
text_preprocessor = TextPreprocessor()
audio_preprocessor = AudioPreprocessor()

# Function to allow interactive data ingestion by a user
def interactive_data_ingestion():
    while True:
        user_input = input("Enter information to store (or type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        # Determine if the data is text or voice (assuming text for simplicity here)
        if user_input.startswith("Voice transcript:"):
            # Process audio data (assuming audio represented by text here for simplicity)
            embedding = audio_preprocessor.generate_embedding_from_audio(user_input)
        else:
            # Process text data
            embedding = text_preprocessor.generate_embedding(user_input)

        # Insert the embedding into the vector database
        db.insert_vector(embedding)
        print(f"[Data Ingestion] Stored: {user_input}")

# Function to allow interactive querying by a user
def interactive_query():
    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        query_embedding = text_preprocessor.generate_embedding(user_query)
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