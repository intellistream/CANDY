import threading
import time
import random

from pyvectordb import VectorDB  # This is for the pybind11 module (e.g., `pyvectordb.so`).
from pyvectordb import VectorDBStream
from data_preprocessing.text_preprocessor import TextPreprocessor

# Initialize the vector database and embedding models
db = VectorDB(128, search_algorithm)
text_preprocessor = TextPreprocessor()

# Example of structured attributes for job postings
job_postings = [
    {"title": "Data Scientist", "description": "Analyze data using machine learning", "location": "New York", "salary": 150000},
    {"title": "Backend Developer", "description": "Develop scalable server-side software", "location": "San Francisco", "salary": 130000},
    {"title": "AI Researcher", "description": "Research AI algorithms and publish papers", "location": "Boston", "salary": 140000},
    {"title": "Frontend Developer", "description": "Develop beautiful UI with React.js", "location": "New York", "salary": 120000},
    {"title": "Cloud Engineer", "description": "Work with cloud infrastructure on AWS", "location": "Seattle", "salary": 145000},
]

# Function to allow interactive data ingestion with Flink-style streaming API
def interactive_data_ingestion():
    for job in job_postings:
        # Use Flink-style API to process, transform, and store the data
        stream = VectorDBStream.from([job])

        # Chain transformations to generate embeddings, add metadata, and store in the database
        stream \
            .map(lambda job: {
            "embedding": text_preprocessor.generate_embedding(job["description"]),
            "structured_data": job  # Keep the structured attributes
        }) \
            .to_sink(lambda processed_job: db.insert_vector(processed_job["embedding"], processed_job["structured_data"]))

        print(f"[Data Ingestion] Stored job posting: {job['title']} in {job['location']} with salary ${job['salary']}")

# Function to allow interactive hybrid querying by a user
def interactive_query():
    while True:
        user_query = input("Enter your query (e.g., 'data science jobs in New York with salary > 130000'): ")
        if user_query.lower() in ["exit", "quit"]:
            break

        # Parse the query for structured filters
        query_embedding = text_preprocessor.generate_embedding(user_query)
        location_filter = "New York" if "New York" in user_query else None
        salary_filter = 130000 if "salary > 130000" in user_query else None

        # Query the vector database using hybrid approach
        results = db.query_nearest_vectors(
            embedding=query_embedding,
            k=3,
            filters={
                "location": location_filter,
                "salary": lambda x: x > salary_filter if salary_filter else True
            }
        )

        print(f"[Interactive Query Results] Retrieved: {results}")

# Main function to start the interactive data ingestion and querying
def main():
    # Start the data ingestion in a separate thread
    ingestion_thread = threading.Thread(target=interactive_data_ingestion)
    ingestion_thread.daemon = True
    ingestion_thread.start()

    # Start interactive querying in the main thread
    interactive_query()

if __name__ == "__main__":
    main()
