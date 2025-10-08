import os
import json
import base64
from unittest.mock import Mock
from main import worker # Import the worker function from your main.py
from config import Config # Import Config for environment variables

# --- Configuration for local testing ---
# Set environment variables for local testing
# Replace with your actual project ID and other details
os.environ["GOOGLE_CLOUD_PROJECT"] = "spanner-demo-kid" # Replace with your project ID
os.environ["GCP_LOCATION"] = "europe-west1"
os.environ["CONSOLIDATION_TOPIC"] = "consolidation-topic-kid"
os.environ["GRAPH_DATA_BUCKET_NAME"] = "spanner-demo-graph-data-kid"
os.environ["LLM_MODEL_NAME"] = "gemini-2.5-flash"
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY" # Replace with your actual Gemini API Key

# Mock Config values if they are used directly from Config class
Config.GCP_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
Config.GCP_LOCATION = os.environ["GCP_LOCATION"]
Config.CONSOLIDATION_TOPIC = os.environ["CONSOLIDATION_TOPIC"]
Config.GRAPH_DATA_BUCKET_NAME = os.environ["GRAPH_DATA_BUCKET_NAME"]
Config.LLM_MODEL_NAME = os.environ["LLM_MODEL_NAME"]
Config.GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]


def create_mock_request(text_chunk: str, batch_id: str, total_chunks: int, chunk_number: int):
    """Creates a mock Flask request object for local testing."""
    data = {
        "chunk": text_chunk,
        "batch_id": batch_id,
        "total_chunks": total_chunks,
        "chunk_number": chunk_number
    }
    
    # The worker expects a base64 encoded JSON string in the 'data' field of the Pub/Sub message
    # However, functions_framework.http directly passes the request body.
    # So, we'll simulate the direct JSON body.
    mock_request = Mock()
    mock_request.get_data.return_value = json.dumps(data)
    return mock_request

if __name__ == "__main__":
    # Example text chunk for testing
    example_text_chunk = "Google Cloud Spanner è un database distribuito globalmente. È stato lanciato nel 2017. Gemini è un modello di linguaggio di grandi dimensioni sviluppato da Google."
    example_batch_id = "test-batch-123"
    example_total_chunks = 1
    example_chunk_number = 0

    print("--- Starting local worker test ---")

    # Create a mock request
    mock_request = create_mock_request(
        text_chunk=example_text_chunk,
        batch_id=example_batch_id,
        total_chunks=example_total_chunks,
        chunk_number=example_chunk_number
    )

    # Call the worker function
    response, status_code = worker(mock_request)

    print(f"Worker Response: {response}")
    print(f"Worker Status Code: {status_code}")
    print("--- Local worker test finished ---")
