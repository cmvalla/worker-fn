import os
import re
import json
import logging
import redis
import litellm
from google.cloud import pubsub_v1
import functions_framework
from google.cloud import logging as cloud_logging
import google.cloud.secretmanager as secretmanager



# --- Boilerplate and Configuration -- -

# Setup structured logging
logging_client = cloud_logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# --- Environment Variables -- -
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "europe-west1")
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
# Retrieve Redis password from Secret Manager
REDIS_PASSWORD = secretmanager.SecretManagerServiceClient().access_secret_version(request={"name": f"projects/{GCP_PROJECT}/secrets/redis-password/versions/latest"}).payload.data.decode("UTF-8")
CONSOLIDATION_TOPIC = os.environ.get("CONSOLIDATION_TOPIC")

logging.info(f"Initializing worker for project '{GCP_PROJECT}' in location '{LOCATION}'")
if not all([REDIS_HOST, REDIS_PASSWORD, CONSOLIDATION_TOPIC]):
    logging.critical("FATAL: Missing one or more required environment variables: REDIS_HOST, REDIS_PASSWORD, CONSOLIDATION_TOPIC")

# --- Global Clients -- -
redis_client = None





try:
    logging.info(f"Initializing Redis client for host '{REDIS_HOST}'...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, socket_connect_timeout=10, ssl=False)
    redis_client.ping()
    logging.info("Redis client initialized and connected successfully.")
except redis.exceptions.ConnectionError as e:
    logging.critical(f"FATAL: Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}. Please check the host, port, and firewall settings. Error: {e}", exc_info=True)
except Exception as e:
    logging.critical(f"FATAL: Failed to initialize Redis client: {e}", exc_info=True)

generation_model = None

# --- Prompt Template for Knowledge Extraction -- -



# --- Prompt Template for Knowledge Extraction -- -
EXTRACTION_PROMPT = """
From the text below, extract entities and their relationships. The entities should have a unique ID, a type (e.g., Person, Organization, Product, Location, Event, Concept, ProgrammingLanguage, Software, OperatingSystem, MathematicalConcept, etc.), and a set of properties (e.g., name, description, value, date, version, role, characteristics, purpose, etc.).
Relationships should connect two entities by their IDs and have a type (e.g., WORKS_FOR, INVESTED_IN, LOCATED_IN, HAS_PROPERTY, IS_A, USES, CREATED_BY, OCCURRED_ON, etc.).
IMPORTANT: If a relationship has a specific date or time period of application, include it as a property of the relationship (e.g., {{"type": "WORKS_FOR", "properties": {{"startDate": "YYYY-MM-DD", "endDate": "YYYY-MM-DD"}}}}).

Respond ONLY with a single, valid JSON object containing two keys: "entities" and "relationships". Do not include any other text or explanations.

TEXT:
---
{text_chunk}
---

JSON:
```json
{{
  "entities": [
    {{
      "id": "unique_id_1",
      "type": "EntityType",
      "properties": {{
        "name": "Entity Name",
        "description": "Entity Description"
      }}
    }}
  ],
  "relationships": [
    {{
      "source": "unique_id_1",
      "target": "unique_id_2",
      "type": "RELATIONSHIP_TYPE",
      "properties": {{
        "startDate": "YYYY-MM-DD"
      }}
    }}
  ]
}}
```
"""

# --- Prompt Template for Summarization ---
SUMMARY_PROMPT = """
Summarize the following text in one concise sentence:
---
{text_chunk}
---

Summary:
"""

def extract_json_from_response(text):
    """
    Extracts a JSON object from the model's text response and performs basic validation.
    Ensures the JSON contains "entities" and "relationships" keys.
    """
    match = re.search(r"```(json)?(.*)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(2).strip()
    else:
        json_str = text.strip()

    try:
        extracted_data = json.loads(json_str)
        
        # Basic schema validation
        if "entities" not in extracted_data or "relationships" not in extracted_data:
            logging.error(f"Model output missing 'entities' or 'relationships' key. Raw text: '{json_str}'")
            # Return a default valid structure to prevent downstream errors
            return {{"entities": [], "relationships": []}}
            
        return extracted_data
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from model output: {e}. Raw text: '{json_str}'")
        # Return a default valid structure to prevent downstream errors
        return {{"entities": [], "relationships": []}}
    except Exception as e:
        logging.error(f"An unexpected error occurred during JSON extraction/validation: {e}. Raw text: '{json_str}'")
        return {{"entities": [], "relationships": []}}

def clean_text(text):
    """
    Cleans the input text by removing HTML tags, special characters,
    and converting it to lowercase.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

@functions_framework.http
def worker(request):
    """
    Processes a text chunk, extracts knowledge, stores it in Redis,
    and triggers the final consolidation if it's the last chunk.
    """
    if not redis_client:
        logging.critical("FATAL: Redis client is not initialized.")
        return "ERROR: Client initialization failed", 500

    try:
        # 1. Parse the incoming request
        request_json = request.get_json(silent=True)
        if not request_json:
            logging.error("Request body is not valid JSON.")
            return "Bad Request: Invalid JSON", 400

        text_chunk = request_json.get("chunk", {}).get("page_content")
        batch_id = request_json.get("batch_id")
        total_chunks = request_json.get("total_chunks")
        chunk_number = request_json.get("chunk_number", 0) # Assuming chunk_number is passed from orchestrator

        if not all([text_chunk, batch_id, total_chunks]):
            logging.error("Missing 'chunk', 'batch_id', or 'total_chunks' in the request.")
            return "Bad Request: Missing required fields", 400

        logging.info(f"Worker received chunk for batch_id '{batch_id}'. Total chunks expected: {total_chunks}")
        
        # 2. Generate summary for the chunk
        summary_prompt = SUMMARY_PROMPT.format(text_chunk=text_chunk)
        summary_response = litellm.completion(
            model="gemini/gemini-pro",
            messages=[{"role": "user", "content": summary_prompt}],
            response_format={"type": "json_object"},
        )
        chunk_summary = summary_response.choices[0].message.content.strip()
        logging.info(f"Generated summary for chunk {chunk_number}: {chunk_summary}")

        # 3. Create the "Chunk" entity
        chunk_entity_id = f"chunk-{batch_id}-{chunk_number}"
        chunk_entity = {
            "id": chunk_entity_id,
            "type": "Chunk",
            "properties": {
                "original_text": text_chunk,
                "summary": chunk_summary,
                "chunk_number": chunk_number
            }
        }

        # 4. Call the model to extract knowledge from the original text_chunk
        prompt = EXTRACTION_PROMPT.format(text_chunk=text_chunk) # Use original text_chunk for extraction
        response = litellm.completion(
            model="gemini/gemini-pro",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        extracted_data = extract_json_from_response(response.choices[0].message.content)
        logging.info(f"Successfully parsed JSON from model output for batch '{batch_id}'.")
        logging.info(f"Extracted data: {json.dumps(extracted_data)}")
        logging.info(f"Extracted data before appending chunk entity: {json.dumps(extracted_data)}")

        # 5. Add the "Chunk" entity to the extracted entities
        extracted_data["entities"].append(chunk_entity)

        # 6. Create a community for the chunk and link it
        chunk_community_id = f"community-{chunk_entity_id}"
        chunk_community_entity = {
            "id": chunk_community_id,
            "type": "Community",
            "properties": {
                "name": f"Community for Chunk {chunk_number}",
                "summary": chunk_summary # Community summary can be chunk summary
            }
        }
        extracted_data["entities"].append(chunk_community_entity)

        # Relationship: Chunk BELONGS_TO ChunkCommunity
        extracted_data["relationships"].append({
            "source": chunk_entity_id,
            "target": chunk_community_id,
            "type": "BELONGS_TO_COMMUNITY" # New relationship type
        })

        # 7. Create relationships from extracted entities to the "Chunk" entity
        for entity in extracted_data["entities"]:
            # Avoid linking the chunk entity to itself or the community entity to itself
            if entity["id"] != chunk_entity_id and entity["id"] != chunk_community_id:
                extracted_data["relationships"].append({
                    "source": entity["id"],
                    "target": chunk_entity_id,
                    "type": "ARE_PART_OF_CHUNK" # New relationship type
                })

        # 8. Store the combined data in Redis
        # The consolidator will need to be updated to handle this new structure
        redis_value = {
            "batch_id": batch_id,
            "chunk_number": chunk_number,
            "extracted_graph_data": extracted_data # Store the combined entities and relationships
        }
        results_key = f"batch:{batch_id}:results"
        redis_client.rpush(results_key, json.dumps(redis_value))
        
        # Increment the counter for the entire chunk, not per sentence
        counter_key = f"batch:{batch_id}:counter" # Ensure counter_key is defined
        current_count = redis_client.incr(counter_key)
        logging.info(f"Stored result for batch '{batch_id}'. Progress: {current_count}/{total_chunks}.")

        # 5. If all chunks are processed, trigger consolidation
        if current_count >= total_chunks:
            logging.info(f"All chunks received for batch '{batch_id}'. Triggering consolidation.")
            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path(GCP_PROJECT, CONSOLIDATION_TOPIC)
            
            message_data = json.dumps({"batch_id": batch_id}).encode("utf-8")
            future = publisher.publish(topic_path, data=message_data)
            message_id = future.result()
            
            logging.info(f"Successfully published consolidation trigger message {message_id} to topic {topic_path}.")

        return "OK", 200

    except json.JSONDecodeError:
        return "Error processing model output", 500
    except Exception as e:
        logging.error(f"An unexpected error occurred in worker for batch '{batch_id}': {e}", exc_info=True)
        return "Internal Server Error", 500