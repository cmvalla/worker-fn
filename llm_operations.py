import json
import requests
import logging
import time
import random
from typing import Any, Dict, List, Optional
from .config import Config
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai.chat_models import ChatVertexAI # Keep this for type hinting, even if not directly instantiated here
import google.auth
import google.auth.transport.requests
import google.oauth2.id_token
import uuid


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text in one concise sentence:"),
    ("user", '''TEXT:
---
{text_chunk}
---

Summary:
''')
])

class LLMOperations:
    def __init__(self, llm_text: ChatVertexAI):
        self.llm_text: ChatVertexAI = llm_text
        self.embedding_service_url = Config.EMBEDDING_SERVICE_URL

    def generate_embeddings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generates embeddings for all entities and communities in batches."""
        if not self.embedding_service_url:
            raise ValueError("EMBEDDING_SERVICE_URL is not set in Config.")

        entities: List[Dict[str, Any]] = data.get("entities", [])
        
        summarization_chain = SUMMARY_PROMPT | self.llm_text

        texts_to_embed_map: Dict[str, str] = {} # Map entity_id to text_to_embed
        entity_id_to_index: Dict[str, int] = {} # Map entity_id to its original index in entities list

        for i, entity in enumerate(entities):
            entity_type: str = entity.get('type', '')
            properties: Dict[str, Any] = entity.get('properties', {})
            entity_id: str = entity.get('id', '')

            text_to_embed: str = ""
            if entity_type == 'Chunk':
                text_to_embed = properties.get('summary', '')
                if not text_to_embed:
                    logging.warning(f"Chunk {entity_id} has an empty summary. Attempting to generate a new one with retries.")
                    original_text: str = properties.get('original_text', '')
                    if original_text:
                        for attempt in range(5):
                            try:
                                summary_response = summarization_chain.invoke({"text_chunk": original_text})
                                summary: str = str(summary_response.content) if not isinstance(summary_response.content, list) else "".join(map(str, summary_response.content))
                                if summary:
                                    properties['summary'] = summary
                                    text_to_embed = summary
                                    logging.info(f"Successfully generated summary for Chunk {entity_id} on attempt {attempt + 1}.")
                                    break
                            except Exception as e:
                                logging.warning(f"Attempt {attempt + 1} to generate summary for Chunk {entity_id} failed: {e}")
                            
                            if attempt < 4:
                                backoff_time = min(1 * (2 ** attempt) + random.uniform(0, 1), 300)
                                logging.info(f"Retrying in {backoff_time:.2f} seconds...")
                                time.sleep(backoff_time)
                        
                        if not text_to_embed:
                            logging.error(f"Failed to generate summary for Chunk {entity_id} after 5 attempts.")
                    else:
                        logging.warning(f"Chunk {entity_id} also has no original_text to generate a summary from.")
            elif entity_type == 'Community':
                text_to_embed = properties.get('summary', '')
                if not text_to_embed:
                    logging.warning(f"Community {entity_id} has an empty summary. No embedding will be generated as there is no text to summarize in the worker.")
            else:
                text_to_embed = f"Type: {entity_type}, Properties: {json.dumps(properties)}"

            if text_to_embed:
                texts_to_embed_map[entity_id] = text_to_embed
                entity_id_to_index[entity_id] = i
            else:
                logging.warning(f"Skipping embedding for entity {entity_id} because there is no text to embed.")

        # Batching logic
        batch_size: int = 50 # Define a suitable batch size
        all_entity_ids: List[str] = list(texts_to_embed_map.keys())
        
        for i in range(0, len(all_entity_ids), batch_size):
            batch_entity_ids: List[str] = all_entity_ids[i:i + batch_size]
            batch_texts: List[str] = [texts_to_embed_map[eid] for eid in batch_entity_ids]
            
            # Explicitly get an ID token for the Cloud Run service
            auth_req = google.auth.transport.requests.Request()
            id_token_raw: Optional[str] = google.oauth2.id_token.fetch_id_token(auth_req, self.embedding_service_url)
            id_token: str = id_token_raw if id_token_raw is not None else ""

            headers: Dict[str, str] = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {id_token}"
            }
            try:
                # Make a single call for both semantic_search and clustering embeddings
                payload: Dict[str, Any] = {"texts": batch_texts, "invocation_id": f"worker-{uuid.uuid4().hex}", "embedding_types": ["semantic_search", "clustering"]}
                response = requests.post(self.embedding_service_url, headers=headers, data=json.dumps(payload))
                logging.info(f"Embedding service response status: {response.status_code} {response.reason}")
                response.raise_for_status()
                embeddings_response: Dict[str, Any] = response.json()
                
                semantic_search_embeddings: List[List[float]] = embeddings_response.get("embeddings", {}).get("semantic_search", [])
                clustering_embeddings: List[List[float]] = embeddings_response.get("embeddings", {}).get("clustering", [])

                for j, entity_id in enumerate(batch_entity_ids):
                    original_index = entity_id_to_index[entity_id]
                    
                    # Assign semantic_search embedding
                    if j < len(semantic_search_embeddings):
                        entities[original_index]['embedding'] = semantic_search_embeddings[j]
                    else:
                        logging.warning(f"Semantic search embedding not found for entity {entity_id}.")

                    # Assign clustering embedding
                    if j < len(clustering_embeddings):
                        entities[original_index]['cluster_embedding'] = clustering_embeddings[j]
                    else:
                        logging.warning(f"Clustering embedding not found for entity {entity_id}.")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error calling embedding service for batch: {e}")
                
        return data