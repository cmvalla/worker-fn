import os
import json
import requests
import logging
from .config import Config
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai.chat_models import ChatVertexAI
import google.auth
import google.auth.transport.requests
import google.oauth2.id_token


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text in one concise sentence:"),
    ("user", """TEXT:
---
{text_chunk}
---

Summary:
""")
])

class LLMOperations:
    def __init__(self, llm_text):
        self.llm_text = llm_text
        self.embedding_service_url = Config.EMBEDDING_SERVICE_URL

    def generate_embeddings(self, data):
        """Generates embeddings for all entities and communities in batches."""
        if not self.embedding_service_url:
            raise ValueError("EMBEDDING_SERVICE_URL is not set in Config.")

        entities = data.get("entities", [])
        
        summarization_chain = SUMMARY_PROMPT | self.llm_text

        texts_to_embed_map = {} # Map entity_id to text_to_embed
        entity_id_to_index = {} # Map entity_id to its original index in entities list

        for i, entity in enumerate(entities):
            entity_type = entity.get('type', '')
            properties = entity.get('properties', {})
            entity_id = entity.get('id')

            text_to_embed = ""
            if entity_type == 'Chunk':
                text_to_embed = properties.get('summary', '')
                if not text_to_embed:
                    logging.warning(f"Chunk {entity_id} has an empty summary. Generating a new one.")
                    original_text = properties.get('original_text', '')
                    if original_text:
                        summary = summarization_chain.invoke({"text_chunk": original_text}).content
                        properties['summary'] = summary
                        text_to_embed = summary
                    else:
                        logging.warning(f"Chunk {entity_id} also has no original_text to generate a summary from.")
            elif entity_type == 'Community':
                text_to_embed = properties.get('summary', '')
                if not text_to_embed:
                    logging.warning(f"Community {entity_id} has an empty summary. No embedding will be generated.")
            else:
                text_to_embed = f"Type: {entity_type}, Properties: {json.dumps(properties)}"

            if text_to_embed:
                texts_to_embed_map[entity_id] = text_to_embed
                entity_id_to_index[entity_id] = i
            else:
                logging.warning(f"Skipping embedding for entity {entity_id} because there is no text to embed.")
                # Assign default zero embeddings if no text to embed
                entities[i]['cluster_embedding'] = [0.0] * Config.EMBEDDING_DIMENSION
                entities[i]['embedding'] = [0.0] * Config.EMBEDDING_DIMENSION

        # Batching logic
        batch_size = 50 # Define a suitable batch size
        all_entity_ids = list(texts_to_embed_map.keys())
        
        for i in range(0, len(all_entity_ids), batch_size):
            batch_entity_ids = all_entity_ids[i:i + batch_size]
            batch_texts = [texts_to_embed_map[eid] for eid in batch_entity_ids]
            
            # Explicitly get an ID token for the Cloud Run service
            auth_req = google.auth.transport.requests.Request()
            id_token = google.oauth2.id_token.fetch_id_token(auth_req, self.embedding_service_url)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {id_token}"
            }
            payload = {"texts": batch_texts}
            
            try:
                response = requests.post(self.embedding_service_url, headers=headers, data=json.dumps(payload))
                logging.info(f"Embedding service response status: {response.status_code} {response.reason}")
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                raw_response_text = response.text
                logging.info(f"Raw embedding service response text: {raw_response_text}")
                batch_embeddings_response = response.json()
                logging.info(f"Embedding service response (raw JSON): {batch_embeddings_response}")

                # Extract the dictionary of embedding types
                all_embeddings_by_type = batch_embeddings_response.get("embeddings", {})

                if not all_embeddings_by_type:
                    logging.error(f"No embeddings found in the response: {batch_embeddings_response}")
                    raise ValueError("No embeddings found in the response")

                semantic_search_embeddings = all_embeddings_by_type.get("semantic_search", [])
                clustering_embeddings = all_embeddings_by_type.get("clustering", [])

                for j, entity_id in enumerate(batch_entity_ids):
                    original_index = entity_id_to_index[entity_id]
                    
                    # Assign semantic_search embedding
                    if j < len(semantic_search_embeddings):
                        entities[original_index]['embedding'] = semantic_search_embeddings[j]
                    else:
                        logging.warning(f"Semantic search embedding not found for entity {entity_id}. Assigning zero embedding.")
                        entities[original_index]['embedding'] = [0.0] * Config.EMBEDDING_DIMENSION

                    # Assign clustering embedding
                    if j < len(clustering_embeddings):
                        entities[original_index]['cluster_embedding'] = clustering_embeddings[j]
                    else:
                        logging.warning(f"Clustering embedding not found for entity {entity_id}. Assigning zero embedding.")
                        entities[original_index]['cluster_embedding'] = [0.0] * Config.EMBEDDING_DIMENSION
            except requests.exceptions.RequestException as e:
                logging.error(f"Error calling embedding service for batch: {e}")
                # Assign default zero embeddings on error
                for entity_id in batch_entity_ids:
                    original_index = entity_id_to_index[entity_id]
                    entities[original_index]['cluster_embedding'] = [0.0] * Config.EMBEDDING_DIMENSION
                    entities[original_index]['embedding'] = [0.0] * Config.EMBEDDING_DIMENSION
                
        return data