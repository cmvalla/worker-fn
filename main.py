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

# --- Environment Variables ---
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

# --- Global Clients ---
try:
    vertexai.init(project=GCP_PROJECT, location=LOCATION)
    generation_model = TextGenerationModel.from_pretrained("text-bison@001")
except Exception as e:
    logging.error(f"Failed to initialize Vertex AI client: {e}")
    generation_model = None

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
    if not generation_model:
        logging.critical("Vertex AI client not initialized. Aborting function.")
        return "ERROR: Client initialization failed", 500

    extracted_text = "" # Initialize to ensure it's available in the except block
    try:
        # 1. Parse the incoming request from Cloud Tasks
        request_json = request.get_json(silent=True)
        chunk_data = request_json.get("chunk", {})
        text_chunk = chunk_data.get("page_content", "")
        callback_url = request_json.get("callback_url")

        if not text_chunk or not callback_url:
            logging.error("Missing 'chunk' or 'callback_url' in the request.")
            return "Bad Request", 400

        logging.info(f"Worker received chunk to process. Callback URL: {callback_url}")

        # 2. Call Vertex AI to extract knowledge
        prompt = EXTRACTION_PROMPT.format(text_chunk=text_chunk)
        response = generation_model.predict(prompt, max_output_tokens=1024)
        
        # The model output might contain markdown fences (```json ... ```) - remove them
        extracted_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        logging.info(f"Raw response from Vertex AI: {extracted_text}")
        extracted_json = json.loads(extracted_text)

        logging.info(f"Successfully extracted {len(extracted_json.get('entities',[]))} entities and {len(extracted_json.get('relationships',[]))} relationships.")

        # 3. Send the result to the callback URL
        headers = {'Content-Type': 'application/json'}
        try:
            callback_response = requests.post(callback_url, data=json.dumps(extracted_json), headers=headers, timeout=60)
            callback_response.raise_for_status() # Raise an exception for bad status codes
            logging.info(f"Successfully sent results to callback URL. Status: {callback_response.status_code}")
        except requests.exceptions.Timeout:
            logging.error(f"Request to callback URL timed out: {callback_url}")
            return "Callback timeout", 504
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send result to callback URL: {e}")
            return "Error calling callback URL", 500


        return "OK", 200

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from model output: {e}. Raw text: {extracted_text}")
        return "Error processing model output", 500
    except Exception as e:
        logging.error(f"An error occurred in the worker function: {e}", exc_info=True)
        return "Internal Server Error", 500