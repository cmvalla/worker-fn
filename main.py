import os
import re
import json
import logging
import time
import random
import igraph as ig
import pickle
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from google.cloud import pubsub_v1
import google.auth
import functions_framework
from google.cloud import storage
from typing import Any, Dict, Optional
from .config import Config
from .llm_operations import LLMOperations

logging.basicConfig(level=logging.DEBUG)



# --- Prompt Templates for Langchain ---
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
From the text below, extract entities and their relationships with extreme detail and verbosity. Your goal is to create a rich and comprehensive knowledge graph.

**Entities:**
- **ID:** Create a unique, descriptive ID for each entity (e.g., 'person-john-doe', 'organization-google', 'product-cloud-spanner').
- **Type:** Assign a specific type (e.g., Person, Organization, Product, Location, Event, Concept, ProgrammingLanguage, Software, OperatingSystem, MathematicalConcept, etc.). Be as granular as possible.
- **Properties:** Extract a comprehensive set of properties. This MUST include not only obvious attributes like 'name', 'date', or 'version', but also more nuanced details.
  - **'name':** Must retain the original language from the text.
  - **'description':** Provide a very detailed, verbose description of the entity, summarizing its context and any relevant information from the text. This should be in English.
  - **'value', 'role', 'characteristics', 'purpose', etc.:** All other properties must be translated into English.
- **Be exhaustive:** Identify and extract every single potential entity mentioned in the text. Do not omit any.

**Relationships:**
- **Connectivity:** Relationships must connect two entities by their IDs in the 'source' and 'target' fields.
- **Type:** Assign a descriptive type (e.g., WORKS_FOR, INVESTED_IN, LOCATED_IN, HAS_PROPERTY, IS_A, USES, CREATED_BY, OCCURRED_ON, etc.). The 'type' property must retain the original language from the text.
- **Properties:**
  - **'confidence':** For every single relationship, you MUST provide a 'confidence' score between 0.0 and 1.0, reflecting your certainty. A higher score means higher certainty.
  - **'description':** Provide a detailed explanation of why this relationship exists, citing evidence from the text.
  - **Temporal Information:** If a relationship is valid for a specific date or time period, include it as a property (e.g., {{"type": "WORKS_FOR", "properties": {{"startDate": "YYYY-MM-DD", "endDate": "YYYY-MM-DD"}}}}).
- **Be Verbose:** Create as many relationships as possible to capture all connections between entities. It is better to have a potentially redundant relationship than to miss a connection.

**Output Format:**
- Respond ONLY with a single, valid JSON object.
- The JSON object must have two keys: "entities" and "relationships".
- Do not include any other text, explanations, or markdown formatting. The output must be a single, raw JSON object.
"""),
    ("user", """TEXT:
---
{text_chunk}
---

JSON:
""")
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text in one concise sentence:"),
    ("user", """TEXT:
---
{text_chunk}
---

Summary:
""")
])

def extract_json_from_response(text: str) -> Dict[str, Any]:
    '''
    Extracts a JSON object from the model's text response and performs basic validation.
    Ensures the JSON contains "entities" and "relationships" keys.
    '''
    # Try to find a JSON block enclosed in ```json ... ```
    match = re.search(r"```json\s*({.*})```", text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        # If not found, try to find a JSON block enclosed in ``` ... ``` (without 'json')
        match = re.search(r"```\s*({.*})```", text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
        else:
            # Fallback: assume the entire text is JSON, but strip common wrappers
            json_str = text.strip()
            # Remove common prefixes/suffixes that are not valid JSON
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
            if json_str.startswith("```"):
                json_str = json_str[3:].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-3].strip()

    # Remove "insensitive:true" from the string
    json_str = json_str.replace("insensitive:true", "")

    # Aggressively strip any leading/trailing non-JSON characters
    json_str = json_str.strip()
    if not json_str.startswith("{") or not json_str.endswith("}"):
        # Attempt to find the first and last curly brace to isolate the JSON
        first_brace = json_str.find("{")
        last_brace = json_str.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = json_str[first_brace : last_brace + 1]
        else:
            logging.error(f"Could not find a valid JSON object in the response. Raw text: '{json_str}'")
            raise ValueError("Invalid JSON format")

    if not json_str:
        logging.error(f"Extracted JSON string is empty after stripping. Raw text: '{text}'")
        raise ValueError("Empty JSON string")

    extracted_data = json.loads(json_str)

    # Basic schema validation
    if "entities" not in extracted_data or "relationships" not in extracted_data:
        logging.error(f"Model output missing 'entities' or 'relationships' key. Raw text: '{json_str}'")
        raise ValueError("Invalid JSON schema")

    return extracted_data

def invoke_llm_with_retry(text_chunk: str, llm_json: ChatVertexAI, max_retries: int = 5) -> Dict[str, Any]:
    extraction_chain = EXTRACTION_PROMPT | llm_json
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempting to call LLM and parse JSON (attempt {attempt + 1}/{max_retries})")
            llm_response = extraction_chain.invoke({"text_chunk": text_chunk})
            return extract_json_from_response(llm_response.content)
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to get valid JSON on attempt {attempt + 1}/{max_retries}: {e}.")
            if attempt < max_retries - 1:
                backoff_time = min(1 * (2 ** attempt) + random.uniform(0, 1), 300)
                logging.info(f"Retrying in {backoff_time:.2f} seconds...")
                time.sleep(backoff_time)
            else:
                logging.error(f"Failed to get valid JSON after {max_retries} attempts.")
                return {"entities": [], "relationships": []}
    return {"entities": [], "relationships": []} # Explicit return to satisfy type checker

    def generate_embeddings(self, data):
        """Generates embeddings for all entities and communities in batches."""
        entities = data.get("entities", [])
        
        summarization_chain = SUMMARY_PROMPT | self.llm

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
                        summary = summarization_chain.invoke({"text_chunk": original_text}).get("text")
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
                entities[i]['cluster_embedding'] = [0.0] * Config.EMBEDDING_DIMENSION
                entities[i]['embedding'] = [0.0] * Config.EMBEDDING_DIMENSION

        # Batching logic
        batch_size = 50 # Define a suitable batch size
        all_entity_ids = list(texts_to_embed_map.keys())
        
        for i in range(0, len(all_entity_ids), batch_size):
            batch_entity_ids = all_entity_ids[i:i + batch_size]
            batch_texts = [texts_to_embed_map[eid] for eid in batch_entity_ids]
            
            batch_embeddings = self.get_embeddings(batch_texts) # Call the new batch embedding function

            for j, entity_id in enumerate(batch_entity_ids):
                original_index = entity_id_to_index[entity_id]
                entities[original_index]['cluster_embedding'] = batch_embeddings[j]['clustering']
                entities[original_index]['embedding'] = batch_embeddings[j]['semantic_search']

        return data




def normalize_entity_ids(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder function for normalizing entity IDs. 
    This function should ensure entity IDs are consistent and unique.
    """
    logging.warning("normalize_entity_ids is a placeholder and does not perform any normalization.")
    return data

@functions_framework.http
def worker(request: Any) -> tuple[str, int]:
    # --- Environment Variables -- -
    gcp_project: Optional[str] = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location: str = os.environ.get("GCP_LOCATION", "europe-west1")
    consolidation_topic: Optional[str] = os.environ.get("CONSOLIDATION_TOPIC")
    graph_data_bucket_name: Optional[str] = os.environ.get("GRAPH_DATA_BUCKET_NAME")
    batch_id: Optional[str] = None # Initialize batch_id to None

    if not graph_data_bucket_name:
        logging.critical("FATAL: GRAPH_DATA_BUCKET_NAME environment variable not set.")
        return "ERROR: Configuration missing", 500

    storage_client: storage.Client = storage.Client()

    logging.info(f"Initializing worker for project '{gcp_project}' in location '{location}'")

    # Log the service account being used
    try:
        credentials, project = google.auth.default()
        service_account_email: str = credentials.service_account_email
        logging.info(f"Worker running with service account: {service_account_email}")
    except Exception as e:
        logging.warning(f"Could not determine service account: {e}")

    # --- Langchain Model Initialization ---
    LLM_MODEL_NAME: str = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-flash") # Default to gemini-2.5-flash if not set

    llm_json: ChatVertexAI = ChatVertexAI(
        project=Config.GCP_PROJECT,
        location=os.environ.get("GCP_LOCATION", "europe-west1"), # Explicitly get from env with default
        model_name=LLM_MODEL_NAME,
        max_output_tokens=65536
    )
    llm_text: ChatVertexAI = ChatVertexAI(
        project=Config.GCP_PROJECT,
        location=os.environ.get("GCP_LOCATION", "europe-west1"), # Explicitly get from env with default
        model_name=LLM_MODEL_NAME,
        temperature=0.5,
        max_output_tokens=65536
    )
    llm_ops: LLMOperations = LLMOperations(llm_text)

    try:
        # 1. Parse the incoming request
        # The request body is a base64-encoded JSON string. The framework decodes base64,
        # but we need to manually parse the JSON string.
        data_str: str = request.get_data(as_text=True)
        if not data_str:
            logging.error("Request body is empty.")
            return "Bad Request: Empty body", 400
        
        try:
            request_json: Dict[str, Any] = json.loads(data_str)
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from request body: {data_str}")
            return "Bad Request: Invalid JSON", 400

        logging.info(f"Worker received request_json: {request_json}")
        
        text_chunk: Optional[str] = request_json.get("chunk")
        batch_id: Optional[str] = request_json.get("batch_id")
        total_chunks: Optional[int] = request_json.get("total_chunks")
        chunk_number: int = request_json.get("chunk_number", 0) # Assuming chunk_number is passed from orchestrator

        if not all([text_chunk, batch_id, total_chunks]):
            logging.error("Missing 'chunk', 'batch_id', or 'total_chunks' in the request.")
            return "Bad Request: Missing required fields", 400

        if text_chunk is None:
            logging.error("text_chunk is None, cannot proceed with LLM invocation.")
            return "Internal Server Error: text_chunk is missing", 500

        logging.info(f"Worker received chunk for batch_id '{batch_id}'. Total chunks expected: {total_chunks}")

        # 2. Generate summary for the chunk
        summary_chain = SUMMARY_PROMPT | llm_text
        summary_response = summary_chain.invoke({"text_chunk": text_chunk})
        summary_response_content: Any = summary_response.content
        chunk_summary: str = str(summary_response_content) if not isinstance(summary_response_content, list) else "".join(map(str, summary_response_content))
        logging.info(f"Generated summary for chunk {chunk_number}: {chunk_summary}")

        # 3. Create the "Chunk" entity
        chunk_entity_id: str = f"chunk-{batch_id}-{chunk_number}"
        chunk_entity: Dict[str, Any] = {
            "id": chunk_entity_id,
            "type": "Chunk",
            "properties": {
                "original_text": text_chunk,
                "summary": chunk_summary,
                "chunk_number": chunk_number
            }
        }

        # 4. Call the model to extract knowledge from the original text_chunk
        extracted_data: Dict[str, Any] = invoke_llm_with_retry(text_chunk, llm_json)
        logging.debug(f"Raw extracted data from LLM: {json.dumps(extracted_data)}")

        # 3. Normalize entity IDs
        extracted_data = normalize_entity_ids(extracted_data)
        logging.debug(f"Data after entity ID normalization: {json.dumps(extracted_data)}")

        # 4. Assign weights to relationships
        for rel in extracted_data.get("relationships", []):
            if "properties" not in rel:
                rel["properties"] = {}
            confidence: Optional[float] = rel["properties"].get("confidence")
            if confidence is not None and 0.0 <= confidence <= 1.0:
                rel["properties"]["weight"] = float(confidence)
            else:
                rel["properties"]["weight"] = 1.0 # Default weight if confidence is not provided or invalid
        logging.debug(f"Data after relationship weight assignment: {json.dumps(extracted_data)}")

        logging.debug(f"Before adding Chunk entity and community: {json.dumps(extracted_data)}")
        # 5. Add the "Chunk" entity to the extracted entities
        extracted_data["entities"].append(chunk_entity)

        # 6. Add the "Community" entity for the chunk
        chunk_community_entity: Dict[str, Any] = {
            "id": f"community-{batch_id}",
            "type": "Community",
            "properties": {
                "name": f"Community for batch {batch_id}",
                "description": f"Community entity representing the batch {batch_id} of processed chunks."
            }
        }
        extracted_data["entities"].append(chunk_community_entity)
        logging.debug(f"After adding Chunk entity and community: {json.dumps(extracted_data)}")

        # Generate embeddings for the extracted data
        extracted_data = llm_ops.generate_embeddings(extracted_data)
        logging.debug(f"Data after embedding generation: {json.dumps(extracted_data)}")

        # 7. Create relationships from extracted entities to the "Chunk" entity
        logging.debug(f"Before creating relationships to Chunk entity: {json.dumps(extracted_data)}")
        for entity in extracted_data["entities"]:
            # Avoid creating a relationship from the chunk entity to itself
            if entity["id"] != chunk_entity["id"]:
                extracted_data["relationships"].append({
                    "source": entity["id"],
                    "target": chunk_entity["id"],
                    "type": "PART_OF",
                    "properties": {"weight": 1, "description": "Indicates that an extracted entity is part of a specific text chunk."}
                })
        logging.debug(f"After creating relationships to Chunk entity: {json.dumps(extracted_data)}")
        # 8. Create igraph, serialize, and store in GCS
        graph: ig.Graph = create_igraph_from_extracted_data(extracted_data)
        serialized_graph: bytes = pickle.dumps(graph)

        # Generate a unique object name for the GCS blob
        timestamp: int = int(time.time())
        gcs_object_name: str = f"graph_data/{batch_id}/{chunk_number}_{timestamp}.pkl"
        gcs_path: str = f"gs://{graph_data_bucket_name}/{gcs_object_name}"

        # Upload to GCS
        bucket: storage.Bucket = storage_client.bucket(graph_data_bucket_name)
        blob: storage.Blob = bucket.blob(gcs_object_name)
        blob.upload_from_string(serialized_graph)
        logging.info(f"Uploaded serialized igraph for batch {batch_id}, chunk {chunk_number} to {gcs_path}")

        # 9. Publish message to consolidation topic
        publisher: pubsub_v1.PublisherClient = pubsub_v1.PublisherClient()
        
        if gcp_project is None:
            logging.error("GCP_PROJECT environment variable is not set.")
            return "Internal Server Error: GCP project not configured", 500
        if consolidation_topic is None:
            logging.error("CONSOLIDATION_TOPIC environment variable is not set.")
            return "Internal Server Error: Consolidation topic not configured", 500

        topic_path: str = publisher.topic_path(gcp_project, consolidation_topic)

        message_data: bytes = json.dumps({"batch_id": batch_id, "gcs_paths": [gcs_path], "chunk_number": chunk_number, "total_chunks": total_chunks}).encode("utf-8")
        future = publisher.publish(topic_path, data=message_data)
        _ = future.result() # Discarding result as message_id is not used

        logging.info(f"Published message with GCS path for batch {batch_id}, chunk {chunk_number} to topic {topic_path}.")

        # Final logging of counts
        num_entities = len(extracted_data.get("entities", []))
        num_relationships = len(extracted_data.get("relationships", []))
        
        num_clustering_embeddings = 0
        num_semantic_embeddings = 0
        for entity in extracted_data.get("entities", []):
            if entity.get("cluster_embedding") and any(e != 0.0 for e in entity["cluster_embedding"]):
                num_clustering_embeddings += 1
            if entity.get("embedding") and any(e != 0.0 for e in entity["embedding"]):
                num_semantic_embeddings += 1

        logging.info(f"Worker final counts for batch {batch_id}: Entities={num_entities}, Relationships={num_relationships}, ClusteringEmbeddings={num_clustering_embeddings}, SemanticEmbeddings={num_semantic_embeddings}")

        return "OK", 200

    except Exception as e:
        logging.error(f"An unexpected error occurred in worker for batch '{batch_id}': {e}", exc_info=True)
        return "Internal Server Error", 500

def create_igraph_from_extracted_data(extracted_data: Dict[str, Any]) -> ig.Graph:
    graph: ig.Graph = ig.Graph()
    
    # Add vertices
    entity_id_to_vertex_index: Dict[str, int] = {}
    for entity in extracted_data.get("entities", []):
        vertex = graph.add_vertex(name=entity["id"])
        for prop_key, prop_value in entity.get("properties", {}).items():
            prop_key: str = prop_key
            prop_value: Any = prop_value
            vertex[prop_key] = prop_value
        vertex["type"] = entity["type"] # Store entity type as a vertex attribute
        if "embedding" in entity:
            vertex["embedding"] = entity["embedding"]
        if "cluster_embedding" in entity:
            vertex["cluster_embedding"] = entity["cluster_embedding"]
        entity_id_to_vertex_index[entity["id"]] = vertex.index

    # Add edges
    for rel in extracted_data.get("relationships", []):
        source_id: str = rel["source"]
        target_id: str = rel["target"]
        
        if source_id in entity_id_to_vertex_index and target_id in entity_id_to_vertex_index:
            edge = graph.add_edge(entity_id_to_vertex_index[source_id], entity_id_to_vertex_index[target_id])
            for prop_key, prop_value in rel.get("properties", {}).items():
                prop_key: str = prop_key
                prop_value: Any = prop_value
                edge[prop_key] = prop_value
            edge["type"] = rel["type"] # Store relationship type as an edge attribute
        else:
            logging.warning(f"Skipping relationship due to missing source or target entity: {rel}")
            
    return graph