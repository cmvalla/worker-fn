import os
import json
import functions_framework
import google.cloud.logging
import logging
import requests
import vertexai
from vertexai.language_models import TextGenerationModel

# --- Boilerplate and Configuration ---

# Setup structured logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.DEBUG) # Set root logger to DEBUG

# --- Environment Variables ---
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
logging.info(f"Initializing worker for project '{GCP_PROJECT}' in location '{LOCATION}'")

# --- Global Clients ---
generation_model = None
try:
    logging.info("Initializing Vertex AI client...")
    vertexai.init(project=GCP_PROJECT, location=LOCATION)
    generation_model = TextGenerationModel.from_pretrained("text-bison")
    logging.info("Vertex AI client initialized successfully.")
except Exception as e:
    logging.critical(f"FATAL: Failed to initialize Vertex AI client: {e}", exc_info=True)
    # The function will fail later if generation_model is None, but we log the critical error here.

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

@functions_framework.http
def worker(request):
    """This function receives a chunk of text, extracts entities and relationships
    using a Vertex AI model, and sends the result to a callback URL.
    """
    logging.debug("Worker function started.")
    if not generation_model:
        logging.critical("Vertex AI client not initialized. Aborting function.")
        return "ERROR: Client initialization failed", 500

    extracted_text = "" # Initialize to ensure it's available in the except block
    try:
        # 1. Parse the incoming request from Cloud Tasks
        logging.debug("Parsing incoming request...")
        request_json = request.get_json(silent=True)
        if not request_json:
            logging.error("Request body is not valid JSON.")
            return "Bad Request: Invalid JSON", 400
            
        logging.debug(f"Request JSON payload: {json.dumps(request_json, indent=2)}")

        chunk_data = request_json.get("chunk", {})
        text_chunk = chunk_data.get("page_content", "")
        callback_url = request_json.get("callback_url")

        if not text_chunk or not callback_url:
            logging.error(f"Missing 'chunk' or 'callback_url' in the request. Found chunk: {'yes' if text_chunk else 'no'}, Found callback: {'yes' if callback_url else 'no'}")
            return "Bad Request: Missing chunk or callback_url", 400

        logging.info(f"Worker received chunk to process. Callback URL: {callback_url}")

        # 2. Call Vertex AI to extract knowledge
        prompt = EXTRACTION_PROMPT.format(text_chunk=text_chunk)
        logging.debug(f"Generated prompt for Vertex AI:\n---\n{prompt}\n---")
        
        logging.info("Calling Vertex AI model...")
        response = generation_model.predict(prompt, max_output_tokens=2048) # Increased token limit
        logging.info("Received response from Vertex AI.")
        
        extracted_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        logging.debug(f"Raw response text from Vertex AI: {extracted_text}")
        
        extracted_json = json.loads(extracted_text)
        logging.info("Successfully parsed JSON from model output.")
        logging.debug(f"Extracted JSON: {json.dumps(extracted_json, indent=2)}")


        # 3. Send the result to the callback URL
        headers = {'Content-Type': 'application/json'}
        logging.info(f"Sending results to callback URL: {callback_url}")
        try:
            callback_response = requests.post(callback_url, data=json.dumps(extracted_json), headers=headers, timeout=60)
            logging.info(f"Callback response status code: {callback_response.status_code}")
            logging.debug(f"Callback response headers: {callback_response.headers}")
            logging.debug(f"Callback response body: {callback_response.text}")
            callback_response.raise_for_status()
            logging.info(f"Successfully sent results to callback URL.")
        except requests.exceptions.Timeout:
            logging.error(f"Request to callback URL timed out: {callback_url}", exc_info=True)
            return "Callback timeout", 504
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send result to callback URL: {e}", exc_info=True)
            return "Error calling callback URL", 500

        return "OK", 200

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from model output: {e}. Raw text: {extracted_text}", exc_info=True)
        return "Error processing model output", 500
    except Exception as e:
        logging.error(f"An unexpected error occurred in the worker function: {e}", exc_info=True)
        return "Internal Server Error", 500
