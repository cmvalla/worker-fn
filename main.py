import os
import json
import re
import functions_framework
import google.cloud.logging
import logging
import requests
import google.generativeai as genai
import google.auth
import google.auth.transport.requests
import google.oauth2.id_token
from google.cloud import pubsub_v1

# --- Boilerplate and Configuration ---

# Setup structured logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# --- Environment Variables ---
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "europe-west1")
logging.info(f"Initializing worker for project '{GCP_PROJECT}' in location '{LOCATION}'")

# --- Global Clients ---
generation_model = None
try:
    logging.info("Initializing Generative AI client...")
    # ADC will be used by default
    genai.configure(transport="rest")
    generation_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    logging.info("Generative AI client initialized successfully.")
except Exception as e:
    logging.critical(f"FATAL: Failed to initialize Generative AI client: {e}", exc_info=True)

# --- Prompt Template for Knowledge Extraction ---
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
    # Use a regex to find the JSON block, even with markdown backticks
    match = re.search(r"```(json)?(.*)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(2).strip()
    else:
        # If no markdown, assume the whole text is the JSON
        json_str = text.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}. Raw text: '{json_str}'")
        raise

@functions_framework.http
def worker(request):
    """
    This function receives a chunk of text, extracts entities and relationships
    using a Generative AI model, and sends the result to a Pub/Sub topic.
    """
    if not generation_model:
        logging.critical("Generative AI client not initialized. Aborting function.")
        return "ERROR: Client initialization failed", 500

    try:
        creds, project_id = google.auth.default()
        logging.info(f"Worker function is running with service account: {creds.service_account_email}")
    except Exception as e:
        logging.error(f"Could not retrieve service account credentials: {e}")

    try:
        # 1. Parse the incoming request
        request_json = request.get_json(silent=True)
        if not request_json:
            logging.error("Request body is not valid JSON.")
            return "Bad Request: Invalid JSON", 400

        text_chunk = request_json.get("chunk", {}).get("page_content")
        results_topic = request_json.get("results_topic")

        if not text_chunk or not results_topic:
            logging.error("Missing 'text_chunk' or 'results_topic' in the request.")
            return "Bad Request: Missing required fields", 400

        logging.info(f"Worker received chunk. Results topic: {results_topic}")

        # 2. Call the model to extract knowledge
        prompt = EXTRACTION_PROMPT.format(text_chunk=text_chunk)
        response = generation_model.generate_content(prompt)
        logging.info(f"AI Response: {response.text}")
        
        extracted_json = extract_json_from_response(response.text)
        logging.info("Successfully parsed JSON from model output.")

        # 3. Send the result to the Pub/Sub topic
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(GCP_PROJECT, results_topic)
        future = publisher.publish(topic_path, data=json.dumps(extracted_json).encode("utf-8"))
        message_id = future.result()
        logging.info(f"Successfully published message {message_id} to topic {topic_path}.")

        return "OK", 200

    except json.JSONDecodeError:
        return "Error processing model output", 500
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        return "Internal Server Error", 500
