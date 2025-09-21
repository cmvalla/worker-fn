import os
import json
import base64
import pytest
from unittest.mock import patch, MagicMock
from flask import Request # Import Request from flask

# Set environment variables for local testing
# Ensure GOOGLE_APPLICATION_CREDENTIALS points to the service account key file
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carlovalla/Desktop/Progetti/000-GoReply/cloud-spanner-demo/functions/worker/tests/credentials.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "spanner-demo-kid"
os.environ["GCP_LOCATION"] = "europe-west1"
os.environ["SPANNER_INSTANCE_ID"] = "demo-instance-kid"
os.environ["SPANNER_DATABASE_ID"] = "spanner-graphrag-db"
os.environ["CONSOLIDATION_TOPIC"] = "consolidation-topic-kid"
os.environ["LLM_MODEL_NAME"] = "gemini-2.5-pro"
os.environ["EMBEDDING_SERVICE_URL"] = "https://graphrag-embedding-kg7odfkvta-ew.a.run.app"
os.environ["GEMINI_API_KEY"] = "AIzaSyCLvNTVrJAF-ekT8qeMJaILw0TwBN0RA1g" # Actual key
os.environ["GRAPH_DATA_BUCKET_NAME"] = "spanner-demo-graph-data-kid"

# Import the worker function after setting environment variables
# This import needs to happen after environment variables are set
# to ensure the Config object in main.py picks them up correctly.
from functions.worker.main import worker

@pytest.fixture(scope="module")
def setup_env():
    # The credentials.json file is now created by a separate step.
    # This fixture will ensure the environment is set up before tests run.
    yield
    # No cleanup of credentials.json here, as it's managed externally.
    pass

@patch('google.cloud.pubsub_v1.PublisherClient')
def test_worker_function_integration(mock_pubsub_client, setup_env):
    # Mock the Pub/Sub message structure
    data = {
        "batch_id": "test-batch-123",
        "chunk": "This is a test document for the worker function. It talks about machine learning and cloud computing.",
        "total_chunks": 1,
        "chunk_number": 0
    }
    message_data = json.dumps(data).encode("utf-8") # No base64 encoding here, as the worker expects raw JSON in the request body

    # Create a mock Flask request object
    mock_request = MagicMock(spec=Request)
    mock_request.get_data.return_value = message_data
    mock_request.args = {} # No query parameters for this test

    # Call the worker function with the mock request
    # This will now use the actual Spanner, GCS, LLM, and Embedding clients
    worker(mock_request)

    # Assertions
    # We only mock PubSubClient, so we assert its call.
    # For Spanner, GCS, LLM, and Embedding, we assume they are called internally
    # and their successful execution is part of the integration.
    mock_pubsub_client.return_value.publish_message.assert_called_once()

    # You might want to add more sophisticated assertions here,
    # e.g., checking if data was actually written to Spanner or GCS
    # by querying them after the worker function call.
    # However, for a basic integration test, asserting that the worker
    # completes without errors and attempts to publish a message is a good start.
