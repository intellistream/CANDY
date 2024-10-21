import os
import sys

sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) #./CANDY
from Embedding.MultimodalPreprocessor import TextPreprocessor
from Embedding.MultimodalPreprocessor import ImagePreprocessor
from Embedding.MultimodalPreprocessor import MultimodalPreprocessor
from pycandy import VectorDB

# Initialize the vector database and embedding models
search_algorithm = 'knnsearch'
db = VectorDB(128, search_algorithm)

# Initialize the vector database and embedding models
text_preprocessor = TextPreprocessor()
image_preprocessor=ImagePreprocessor()
multimodalpreprocessor=MultimodalPreprocessor()

def data_ingestion():
    user_inputs = []
    multimodal_inputs=[]
    while True:
        user_input = input("\n[Data Ingestion] Enter information to store 'text' or 'image' or 'multimodal' to upload (or type 'done' to finish):\n> ")

        if user_input.lower() == "done":
            print("\n[Data Ingestion] Finished data ingestion.")
            break
        elif user_input.lower() == "text":
            text = input("\n[Data Ingestion] Enter text to store:\n> ")
            user_inputs.append({"type": "text", "data": text})
        elif user_input.lower() == "image":
            image_path = input("\n[Data Ingestion] Enter the path of the image to store:\n> ")
            if os.path.exists(image_path):
                user_inputs.append({"type": "image", "data": image_path})
            else:
                print("[Data Ingestion] Invalid image path. Please try again.")
        elif user_input.lower() == "multimodal":
            text = input("\n[Data Ingestion] Enter text to store:\n> ")
            image_path = input("\n[Data Ingestion] Enter the path of the image to store:\n> ")
            if os.path.exists(image_path):
                user_inputs.append({"type": "multimodal", "text": text, "image":image_path})
            else:
                print("[Data Ingestion] Invalid image path. Please try again.")

        else:
            print("[Data Ingestion] Invalid input. Please enter 'text', 'image', 'multimodal',or 'done'.")

    for entry in user_inputs:
        if entry["type"] == "text":
            embedding = text_preprocessor.generate_embedding(entry["data"])
            db.insert_vector(embedding)
            print(f"\n[Data Ingestion] Successfully stored text: '{entry['data']}'")
        elif entry["type"] == "image":
            try:
                image_path = entry["data"]
                embedding = image_preprocessor.generate_embedding(image_path)
                db.insert_vector(embedding)
                print(f"\n[Data Ingestion] Successfully stored image from: '{entry['data']}'")
            except Exception as e:
                print(f"[Data Ingestion] Failed to process image '{entry['data']}': {e}")
        elif entry["type"]=="multimodal":
            try:
                text=entry["text"]
                image_path=entry["image"]
                multimodal_embedding=multimodalpreprocessor.generate_multimodal_embedding(text,image_path)
                db.insert_vector(multimodal_embedding)
                print(f"\n[Data Ingestion] Successfully stored multimodal input: text='{text}', image='{image_path}' ")
            except Exception as e:
                print(f"[Data Ingestion] Failed to process multimodal input: text: '{text}',image: '{image_path}',error:{e}")

# Function to allow interactive querying by a user
def interactive_query():
    while True:
        user_choice = input("\n[Interactive Query] Choose your query type('text', 'image' , 'multimodal' or type 'exit' to quit):\n> ")
        if user_choice.lower() in ["exit", "quit"]:
            print("\n[Interactive Query] Exiting...")
            break
        query_embedding=None
        if user_choice.lower()=='text':
            user_query=input("\n[Interactive Query] Enter your text query:\n>")
            # Convert the numpy array to a list to ensure compatibility with pybind11
            query_embedding = text_preprocessor.generate_embedding(user_query).tolist()
            if not isinstance(query_embedding,list):
                query_embedding=list(query_embedding)

        elif user_choice.lower()=='image':
            user_query=input("\n[Interactive Query] Enter the path of your image query:\n>")
            if os.path.exists(user_query):
                try:
                    query_embedding=image_preprocessor.generate_embedding(user_query).tolist()
                    if not isinstance(query_embedding, list):
                        query_embedding = list(query_embedding)

                except Exception as e:
                    print(f"[Interactive Query] Failed to process image '{user_query}'： {e}")
                    continue
            else:
                print(f"[Interactive Query] Invalid image path. Please try again.")
                continue
        elif user_choice.lower()=='multimodal':
            user_query = input("\n[Interactive Query] Enter your text query:\n>")
            image_path = input("\n[Interactive Query] Enter the path of your image query:\n>")

            if os.path.exists(image_path):
                try:
                    query_embedding = multimodalpreprocessor.generate_multimodal_embedding(user_query,image_path).tolist()
                    if not isinstance(query_embedding, list):
                        query_embedding = list(query_embedding)
                    user_query = user_query + " " + image_path
                except Exception as e:
                    print(f"[Interactive Query] Failed to process image '{image_path}'： {e}")
                    continue

        else:
            print("[Interactive Query] Invalid choice. Please enter 'text', 'image' , 'multimodal' or 'exit'.")
            continue

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