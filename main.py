import os
import re
import json
import logging
import redis
import google.generativeai as genai
from google.cloud import pubsub_v1
import functions_framework
from google.cloud import logging as cloud_logging
import google.cloud.secretmanager as secretmanager
import nltk
import hashlib

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
generation_model = None
redis_client = None

try:
    logging.info("Initializing Generative AI client...")
    genai.configure(transport="rest")
    generation_model = genai.GenerativeModel("gemini-2.5-flash")
    logging.info("Generative AI client initialized successfully.")
except Exception as e:
    logging.critical(f"FATAL: Failed to initialize Generative AI client: {e}", exc_info=True)

try:
    logging.info(f"Initializing Redis client for host '{REDIS_HOST}'...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, socket_connect_timeout=10, ssl=False)
    redis_client.ping()
    logging.info("Redis client initialized and connected successfully.")
except redis.exceptions.ConnectionError as e:
    logging.critical(f"FATAL: Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}. Please check the host, port, and firewall settings. Error: {e}", exc_info=True)
except Exception as e:
    logging.critical(f"FATAL: Failed to initialize Redis client: {e}", exc_info=True)

# --- Prompt Template for Knowledge Extraction -- -
EXTRACTION_PROMPT = """
From the text below, extract entities and their relationships. The entities should have a unique ID, a type (e.g., Person, Organization, Product), and a set of properties. The relationships should connect two entities by their IDs and have a type (e.g., WORKS_FOR, INVESTED_IN).

Respond ONLY with a single, valid JSON object containing two keys: "entities" and "relationships". Do not include any other text or explanations.

TEXT:
---
{text_chunk}
---

JSON:
"""

def extract_json_from_response(text):
    """Extracts a JSON object from the model's text response."""
    match = re.search(r"```(json)?(.*)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(2).strip()
    else:
        json_str = text.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}. Raw text: '{json_str}'")
        raise

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
    if not generation_model or not redis_client:
        logging.critical("FATAL: A required client (GenAI or Redis) is not initialized.")
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

        if not all([text_chunk, batch_id, total_chunks]):
            logging.error("Missing 'chunk', 'batch_id', or 'total_chunks' in the request.")
            return "Bad Request: Missing required fields", 400

        logging.info(f"Worker received chunk for batch_id '{batch_id}'. Total chunks expected: {total_chunks}")
        
        # 2. Split the chunk into sentences and process each
        sentences = nltk.sent_tokenize(text_chunk)
        
        for sentence in sentences:
            if not sentence.strip(): # Skip empty sentences
                continue

            cleaned_sentence = clean_text(sentence) # Clean each sentence
            logging.info(f"Processing cleaned sentence: {cleaned_sentence}")
            
            # 3. Call the model to extract knowledge for each sentence
            prompt = EXTRACTION_PROMPT.format(text_chunk=cleaned_sentence)
            response = generation_model.generate_content(prompt)
            extracted_json = extract_json_from_response(response.text)
            logging.info(f"Successfully parsed JSON from model output for sentence in batch '{batch_id}'.")
            logging.info(f"Extracted data for sentence: {json.dumps(extracted_json)}")
            
            # 4. Store result and sentence in Redis
            # The key for Redis will be batch_id:sentence_hash to avoid duplicates
            sentence_hash = hashlib.md5(sentence.encode('utf-8')).hexdigest()
            redis_key = f"batch:{batch_id}:sentence:{sentence_hash}"
            
            # Store a dictionary containing the extracted JSON and the original sentence
            redis_value = {
                "sentence": sentence,
                "extracted_data": extracted_json
            }
            redis_client.set(redis_key, json.dumps(redis_value)) # Use SET instead of LPUSH for unique sentences
            
            logging.info(f"Processed sentence and stored in Redis for batch '{batch_id}'.")
        
        # Increment the counter for the entire chunk, not per sentence
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