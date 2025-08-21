import os
import json
import functions_framework
import google.cloud.logging
import logging
import requests
import google.generativeai as genai
import google.auth
import google.auth.transport.requests

# --- Boilerplate and Configuration ---

# Setup structured logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.DEBUG) # Set root logger to DEBUG
genai.configure(api_key="AIzaSyC0qz3A96tG_ngg5fax1PDH9TwXqPMuQqc")

# --- Environment Variables ---
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "europe-west1")
logging.info(f"Initializing worker for project '{GCP_PROJECT}' in location '{LOCATION}'")

# --- Global Clients ---
generation_model = None
try:
    logging.info("Initializing Generative AI client...")
    genai.configure(transport="rest") # Use REST transport for simplicity in this environment
    generation_model = genai.GenerativeModel("gemini-2.5-flash-lite")
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

def get_google_id_token(audience):
    """Fetches a Google-signed ID token for the specified audience."""
    logging.debug(f"Fetching ID token for audience: {audience}")
    creds, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    id_token = creds.id_token
    logging.debug("Successfully fetched ID token.")
    return id_token

@functions_framework.http
def worker(request):
    """This function receives a chunk of text, extracts entities and relationships
    using a Generative AI model, and sends the result to a callback URL.
    """
    try:
        creds, project_id = google.auth.default()
        logging.info(f"Worker function is running with service account: {creds.service_account_email}")
    except Exception as e:
        logging.error(f"Could not retrieve service account credentials: {e}")

    logging.debug("Worker function started.")
    if not generation_model:
        logging.critical("Generative AI client not initialized. Aborting function.")
        return "ERROR: Client initialization failed", 500

    extracted_text = ""
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
            logging.error(f"Missing 'chunk' or 'callback_url' in the request.")
            return "Bad Request: Missing chunk or callback_url", 400

        logging.info(f"Worker received chunk to process. Callback URL: {callback_url}")

        # 2. Call the model to extract knowledge
        prompt = EXTRACTION_PROMPT.format(text_chunk=text_chunk)
        logging.info("Calling Generative AI model...")
        response = generation_model.generate_content(prompt)
        logging.info("Received response from Generative AI.")
        
        extracted_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        logging.debug(f"Raw response text from model: {extracted_text}")
        
        extracted_json = json.loads(extracted_text)
        logging.info(f"Successfully parsed JSON from model output.")

        # 3. Send the result to the callback URL with authentication
        logging.info(f"Sending results to callback URL: {callback_url}")
        try:
            # Fetch an ID token for the callback URL
            id_token = get_google_id_token(audience=callback_url)
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {id_token}'
            }
            
            callback_response = requests.post(callback_url, data=json.dumps(extracted_json), headers=headers, timeout=60)
            logging.info(f"Callback response status code: {callback_response.status_code}")
            callback_response.raise_for_status()
            logging.info(f"Successfully sent results to callback URL.")
        except Exception as e:
            logging.error(f"Failed to send result to callback URL: {e}", exc_info=True)
            return "Error calling callback URL", 500

        return "OK", 200

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from model output: {e}. Raw text: {extracted_text}", exc_info=True)
        return "Error processing model output", 500
    except Exception as e:
        logging.error(f"An unexpected error occurred in the worker function: {e}", exc_info=True)
        return "Internal Server Error", 500