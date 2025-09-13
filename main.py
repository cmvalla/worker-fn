import os
import re
import json
import logging
import time
import random
import redis
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from google.cloud import pubsub_v1
import functions_framework
from google.cloud import logging as cloud_logging
import google.cloud.secretmanager as secretmanager



# --- Boilerplate and Configuration -- -

# Setup structured logging
logging_client = cloud_logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)


def get_redis_password(gcp_project):
    """Retrieves the Redis password from Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{gcp_project}/secrets/redis-password/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logging.critical(f"Failed to retrieve Redis password from Secret Manager: {e}", exc_info=True)
        return None

def get_redis_client(redis_host, redis_port, gcp_project):
    """Initializes and returns a Redis client."""
    redis_password = get_redis_password(gcp_project)
    if not all([redis_host, redis_password]):
        logging.critical("FATAL: Missing one or more required environment variables: REDIS_HOST, REDIS_PASSWORD")
        return None

    try:
        logging.info(f"Initializing Redis client for host '{redis_host}'...")
        redis_client = redis.Redis(host=redis_host, port=redis_port, password=redis_password, socket_connect_timeout=10, ssl=False)
        redis_client.ping()
        logging.info("Redis client initialized and connected successfully.")
        return redis_client
    except redis.exceptions.ConnectionError as e:
        logging.critical(f"FATAL: Could not connect to Redis at {redis_host}:{redis_port}. Please check the host, port, and firewall settings. Error: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.critical(f"FATAL: Failed to initialize Redis client: {e}", exc_info=True)
        return None

# --- Prompt Templates for Langchain ---
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
From the text below, extract entities and their relationships. The entities should have a unique ID, a type (e.g., Person, Organization, Product, Location, Event, Concept, ProgrammingLanguage, Software, OperatingSystem, MathematicalConcept, etc.), and a set of properties (e.g., name, description, value, date, version, role, characteristics, purpose, etc.).
For each entity, ensure its 'name' property retains the original language from the text. All other properties, such as 'description', 'value', 'role', etc., must be translated into English.
Relationships should connect two entities by their IDs. The relationship object MUST have 'source' and 'target' fields, which are the IDs of the connected entities. It should also have a 'type' (e.g., WORKS_FOR, INVESTED_IN, LOCATED_IN, HAS_PROPERTY, IS_A, USES, CREATED_BY, OCCURRED_ON, etc.). For relationships, the 'type' property must also retain the original language from the text. For every single relationship you extract, you MUST provide a 'confidence' score between 0.0 and 1.0 in its properties. The confidence score should reflect how certain you are that the relationship is correctly stated in the text. A higher score means higher certainty.
IMPORTANT: If a relationship has a specific date or time period of application, include it as a property of the relationship (e.g., {{"type": "WORKS_FOR", "properties": {{"startDate": "YYYY-MM-DD", "endDate": "YYYY-MM-DD"}}}}).

Respond ONLY with a single, valid JSON object containing two keys: "entities" and "relationships". Do not include any other text or explanations.
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

def extract_json_from_response(text):
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

    extracted_data = json.loads(json_str)

    # Basic schema validation
    if "entities" not in extracted_data or "relationships" not in extracted_data:
        logging.error(f"Model output missing 'entities' or 'relationships' key. Raw text: '{json_str}'")
        raise ValueError("Invalid JSON schema")

    return extracted_data

def invoke_llm_with_retry(text_chunk, llm_json, max_retries=5):
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

def clean_text(text):
    '''
    Cleans the input text by removing HTML tags, special characters,
    and converting it to lowercase.
    '''
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

@functions_framework.http
def worker(request):
    # --- Environment Variables -- -
    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GCP_LOCATION", "europe-west1")
    redis_host = os.environ.get("REDIS_HOST")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))
    consolidation_topic = os.environ.get("CONSOLIDATION_TOPIC")

    logging.info(f"Initializing worker for project '{gcp_project}' in location '{location}'")

    # --- Langchain Model Initialization ---
    llm_json = ChatVertexAI(
        project=gcp_project,
        location=location,
        model_name="gemini-2.5-flash",
        response_mime_type="application/json",
    )
    llm_text = ChatVertexAI(
        project=gcp_project,
        location=location,
        model_name="gemini-2.5-flash",
    )
    json_parser = JsonOutputParser()

    redis_client = get_redis_client(redis_host, redis_port, gcp_project)
    if not redis_client:
        logging.critical("FATAL: Redis client is not initialized.")
        return "ERROR: Client initialization failed", 500

    try:
        # 1. Parse the incoming request
        # The request body is a base64-encoded JSON string. The framework decodes base64,
        # but we need to manually parse the JSON string.
        data_str = request.get_data(as_text=True)
        if not data_str:
            logging.error("Request body is empty.")
            return "Bad Request: Empty body", 400
        
        try:
            request_json = json.loads(data_str)
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from request body: {data_str}")
            return "Bad Request: Invalid JSON", 400

        logging.info(f"Worker received request_json: {request_json}")
        
        text_chunk = request_json.get("chunk")
        batch_id = request_json.get("batch_id")
        total_chunks = request_json.get("total_chunks")
        chunk_number = request_json.get("chunk_number", 0) # Assuming chunk_number is passed from orchestrator

        if not all([text_chunk, batch_id, total_chunks]):
            logging.error("Missing 'chunk', 'batch_id', or 'total_chunks' in the request.")
            return "Bad Request: Missing required fields", 400

        logging.info(f"Worker received chunk for batch_id '{batch_id}'. Total chunks expected: {total_chunks}")

        # 2. Generate summary for the chunk
        summary_chain = SUMMARY_PROMPT | llm_text
        summary_response = summary_chain.invoke({"text_chunk": text_chunk})
        chunk_summary = summary_response.content.strip()
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
        extracted_data = invoke_llm_with_retry(text_chunk, llm_json)
        logging.info(f"Extracted data from LLM: {json.dumps(extracted_data)}")

        # Normalize the entity ID key from '_id' to 'id' to handle LLM inconsistencies
        for entity in extracted_data.get("entities", []):
            if "_id" in entity:
                entity["id"] = entity.pop("_id")

        # Add weight to LLM extracted relationships based on confidence, or default to 1
        for rel in extracted_data.get("relationships", []):
            if "properties" not in rel:
                rel["properties"] = {}
            confidence = rel["properties"].get("confidence")
            if isinstance(confidence, (int, float)):
                rel["properties"]["weight"] = float(confidence)
            else:
                rel["properties"]["weight"] = 1.0 # Default weight if confidence is not provided or invalid
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
                "summary": chunk_summary, # Community summary can be chunk summary
                "community_type": "contextual"
            }
        }
        extracted_data["entities"].append(chunk_community_entity)

        # Relationship: Chunk BELONGS_TO ChunkCommunity
        extracted_data["relationships"].append({
            "source": chunk_entity_id,
            "target": chunk_community_id,
            "type": "BELONGS_TO_COMMUNITY", # New relationship type
            "properties": {"weight": 0.5, "description": "Indicates that a text chunk belongs to a specific community."}
        })

        # 7. Create relationships from extracted entities to the "Chunk" entity
        for entity in extracted_data["entities"]:
            # Avoid linking the chunk entity to itself or the community entity to itself
            if entity["id"] != chunk_entity_id and entity["id"] != chunk_community_id:
                extracted_data["relationships"].append({
                    "source": entity["id"],
                    "target": chunk_entity_id,
                    "type": "ARE_PART_OF_CHUNK", # New relationship type
                    "properties": {"weight": 1, "description": "Indicates that an extracted entity is part of a specific text chunk."}
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
            topic_path = publisher.topic_path(gcp_project, consolidation_topic)

            message_data = json.dumps({"batch_id": batch_id}).encode("utf-8")
            future = publisher.publish(topic_path, data=message_data)
            message_id = future.result()

            logging.info(f"Successfully published consolidation trigger message {message_id} to topic {topic_path}.")

        return "OK", 200

    except Exception as e:
        logging.error(f"An unexpected error occurred in worker for batch '{batch_id}': {e}", exc_info=True)
        return "Internal Server Error", 500