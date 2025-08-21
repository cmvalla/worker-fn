import os
import json
import pytest
from unittest.mock import Mock, patch
from main import worker
import google.auth
from google.oauth2 import service_account

# --- Test Configuration ---
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "credentials.json")

@pytest.fixture(scope="module")
def direct_credentials_fixture():
    """Fixture to load credentials directly from the service account key file."""
    if not os.path.exists(CREDENTIALS_FILE):
        pytest.fail(
            f"Service account key file not found at '{CREDENTIALS_FILE}'. "
            "Please create the file as per the instructions."
        )
    
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    project_id = creds.project_id
    
    yield creds, project_id

@patch("requests.post")
@patch("main.get_google_id_token", return_value="mock_id_token")
def test_worker_local_integration(MockGetIdToken, MockPost, direct_credentials_fixture):
    """
    Tests the worker function locally with a real call to the Generative AI service.
    """
    direct_creds, test_project_id = direct_credentials_fixture

    # --- Mock the HTTP Post Request ---
    mock_post_response = Mock()
    mock_post_response.status_code = 200
    MockPost.return_value = mock_post_response

    # --- Simulate a Cloud Tasks Request ---
    mock_request = Mock()
    mock_request.get_json.return_value = {
        "chunk": {
            "page_content": "Bill Gates is the co-founder of Microsoft."
        },
        "callback_url": "https://example.com/callback"
    }

    # --- Patch google.auth.default to use the direct credentials ---
    with patch("google.auth.default", return_value=(direct_creds, test_project_id)):
        # --- Invoke the Worker Function ---
        response, status_code = worker(mock_request)

    # --- Assertions ---
    assert status_code == 200
    assert response == "OK"
    MockGetIdToken.assert_called_once_with(audience="https://example.com/callback")
    MockPost.assert_called_once()

    # Optional: Add an assertion to check the content of the POST call
    # This is more advanced and requires inspecting the call arguments
    # For now, we are just ensuring the call was made.

if __name__ == "__main__":
    pytest.main([__file__])
