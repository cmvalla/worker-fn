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

# --- Environment Variables -- -
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "europe-west1")
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
# Retrieve Redis password from Secret Manager
REDIS_PASSWORD = secretmanager.SecretManagerServiceClient().access_secret_version(request={"name": f"projects/{GCP_PROJECT}/secrets/redis-password/versions/latest"}).payload.data.decode("UTF-8")
CONSOLIDATION_TOPIC = os.environ.get("CONSOLIDATION_TOPIC")
# The worker function will use the service account credentials for authentication
# The GOOGLE_APPLICATION_CREDENTIALS environment variable should be set to the path of the service account key file


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

# --- Langchain Model Initialization ---
llm_json = ChatVertexAI(
    project=GCP_PROJECT,
    location=LOCATION,
    model_name="gemini-2.5-flash",
    response_mime_type="application/json",
)
llm_text = ChatVertexAI(
    project=GCP_PROJECT,
    location=LOCATION,
    model_name="gemini-2.5-flash",
)
json_parser = JsonOutputParser()

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

def invoke_llm_with_retry(text_chunk, max_retries=5):
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
    '''
    Processes a text chunk, extracts knowledge, stores it in Redis,
    and triggers the final consolidation if it's the last chunk.
    '''
    if not redis_client:
        logging.critical("FATAL: Redis client is not initialized.")
        return "ERROR: Client initialization failed", 500

    try:
        # 1. Parse the incoming request
        request_json = request.get_json(silent=True)
        logging.info(f"Worker received request_json: {request_json}") # Added logging
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