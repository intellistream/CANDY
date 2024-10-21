from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
import torch
import numpy as np
from PIL import Image


class TextPreprocessor:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.fixed_dimension = 128  # Fixed dimension for the embedding vectors

    def generate_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Pooling strategy (mean pooling)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        # Ensure the embedding has the correct fixed dimension
        if embedding.size(0) > self.fixed_dimension:
            embedding = embedding[:self.fixed_dimension]  # Truncate if larger
        elif embedding.size(0) < self.fixed_dimension:
            padding = torch.zeros(self.fixed_dimension - embedding.size(0))
            embedding = torch.cat((embedding, padding))  # Pad if smaller
        return embedding.numpy().astype(np.float32)  # Convert to float32


class ImagePreprocessor: #openai/clip-vit-base-patch32
    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.fixed_dimension = 128

    def generate_embedding(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        # width,height=image.size
        # print(f"width: {width}",f"height: {height}")
        # if image.size!=(224,224):
        #     image=image.resize((224,224))

        inputs = self.feature_extractor(images=image, return_tensors='pt')
        print(inputs['pixel_values'].shape)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Pooling strategy (mean pooling)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        if embedding.size(0) > self.fixed_dimension:
            embedding = embedding[:self.fixed_dimension]
        elif embedding.size(0) < self.fixed_dimension:
            padding = torch.zeros(self.fixed_dimension - embedding.size(0))
            embedding = torch.cat((embedding, padding))
        return embedding.numpy().astype(np.float32)


class MultimodalPreprocessor:
    def __init__(self):
        self.text_processor = TextPreprocessor()
        self.image_processor = ImagePreprocessor()

    def generate_multimodal_embedding(self, text: str, image_path: str):
        text_embedding = self.text_processor.generate_embedding(text)
        image_embedding = self.image_processor.generate_embedding(image_path)
        multimodal_embedding = np.add(text_embedding, image_embedding)

        return multimodal_embedding



if __name__ == "__main__":

    multimodal_processor = MultimodalPreprocessor()

    text_input = "This is an example sentence."
    image_path = "/home/kuang/Image/D77.jpg"

    multimodal_embedding = multimodal_processor.generate_multimodal_embedding(text_input, image_path)
    print("shape: ",multimodal_embedding.shape)






