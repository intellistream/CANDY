from transformers import AutoTokenizer, AutoModel
import torch

class TextPreprocessor:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def generate_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Pooling strategy (mean pooling)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding
