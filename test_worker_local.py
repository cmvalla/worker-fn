import os
import json
import pytest
from unittest.mock import Mock, patch
from main import worker

# --- Test Configuration ---
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "credentials.json")

@pytest.fixture(scope="module", autouse=True)
def set_google_credentials():
    """Fixture to set the GOOGLE_APPLICATION_CREDENTIALS environment variable."""
    if not os.path.exists(CREDENTIALS_FILE):
        pytest.fail(
            f"Service account key file not found at '{CREDENTIALS_FILE}'. "
            "Please create the file as per the instructions."
        )
    
    original_value = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
    
    yield
    
    # Teardown: Restore original environment variable
    if original_value is None:
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    else:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = original_value

@patch("requests.post")
@patch("main.get_google_id_token", return_value="mock_id_token")
def test_worker_local_integration(MockGetIdToken, MockPost):
    """
    Tests the worker function locally using credentials from the key file.
    """
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

    # --- Invoke the Worker Function ---
    # No need to patch google.auth.default, the library will pick up the env var
    response, status_code = worker(mock_request)

    # --- Assertions ---
    assert status_code == 200
    assert response == "OK"
    MockGetIdToken.assert_called_once_with(audience="https://example.com/callback")
    MockPost.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])
