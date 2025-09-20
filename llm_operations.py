import os
import json
import requests
from config import Config

class LLMOperations:
    def __init__(self, llm_text):
        self.llm_text = llm_text
        self.embedding_service_url = Config.EMBEDDING_SERVICE_URL

    def generate_embeddings(self, extracted_data):
        if not self.embedding_service_url:
            raise ValueError("EMBEDDING_SERVICE_URL is not set in Config.")

        # Assuming extracted_data is a list of dictionaries, and each dictionary has a 'text' key
        texts_to_embed = [item['text'] for item in extracted_data]

        headers = {"Content-Type": "application/json"}
        payload = {"texts": texts_to_embed}

        try:
            response = requests.post(self.embedding_service_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for HTTP errors
            embeddings = response.json()

            # Assuming the embedding service returns a list of embeddings in the same order as texts_to_embed
            for i, item in enumerate(extracted_data):
                item['embedding'] = embeddings[i]

            return extracted_data
        except requests.exceptions.RequestException as e:
            print(f"Error calling embedding service: {e}")
            raise
