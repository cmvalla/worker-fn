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

def get_id_token(audience):
    """Fetches a Google-signed ID token for the specified audience."""
    try:
        auth_req = google.auth.transport.requests.Request()
        id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience)
        return id_token
    except Exception as e:
        logging.error(f"Failed to fetch ID token for audience {audience}: {e}", exc_info=True)
        return None

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
    using a Generative AI model, and sends the result to a callback URL.
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
        callback_url = request_json.get("callback_url")

        if not text_chunk or not callback_url:
            logging.error("Missing 'text_chunk' or 'callback_url' in the request.")
            return "Bad Request: Missing required fields", 400

        logging.info(f"Worker received chunk. Callback URL: {callback_url}")

        # 2. Call the model to extract knowledge
        prompt = EXTRACTION_PROMPT.format(text_chunk=text_chunk)
        response = generation_model.generate_content(prompt)
        logging.info(f"AI Response: {response.text}")
        
        extracted_json = extract_json_from_response(response.text)
        logging.info("Successfully parsed JSON from model output.")

        # 3. Send the result to the callback URL
        headers = {
            'Content-Type': 'application/json',
        }
        
        callback_response = requests.post(callback_url, data=json.dumps(extracted_json), headers=headers, timeout=60)
        callback_response.raise_for_status()
        logging.info(f"Successfully sent results to callback URL. Status: {callback_response.status_code}")
        logging.info(f"Callback response: {callback_response.text}")

        return "OK", 200

    except json.JSONDecodeError:
        return "Error processing model output", 500
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send result to callback URL: {e}", exc_info=True)
        return "Error calling callback URL", 500
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        return "Internal Server Error", 500
