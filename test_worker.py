import os
import json
import pytest
from unittest.mock import Mock, patch

@patch("google.cloud.secretmanager.SecretManagerServiceClient")
def test_worker(MockSecretManager):
    """
    Tests the worker function.
    """
    # --- Mock the SecretManagerServiceClient ---
    MockSecretManager.return_value.access_secret_version.return_value.payload.data = b"test_password"

    # --- Import the worker function ---
    from main import worker

    # --- Mock the external clients ---
    with patch("main.redis.Redis") as MockRedis, \
         patch("main.llm") as mock_llm, \
         patch("main.pubsub_v1.PublisherClient") as MockPublisherClient:

        mock_redis_client = Mock()
        MockRedis.return_value = mock_redis_client

        mock_publisher_client = Mock()
        MockPublisherClient.return_value = mock_publisher_client
        mock_publisher_client.topic_path.return_value = "projects/test-project/topics/test-topic"
        mock_publisher_client.publish.return_value.result.return_value = "test_message_id"


        # --- Mock the input data ---
        mock_request = Mock()
        mock_request.get_json.return_value = {
            "chunk": {
                "page_content": "Bill Gates is the co-founder of Microsoft."
            },
            "batch_id": "test_batch_123",
            "total_chunks": 1,
            "chunk_number": 0
        }

        # --- Mock the LLM responses ---
        mock_summary_response = Mock()
        mock_summary_response.content = "Bill Gates co-founded Microsoft."
        mock_extraction_response = Mock()
        mock_extraction_response.content = json.dumps({
            "entities": [
                {"id": "1", "type": "Person", "properties": {"name": "Bill Gates"}},
                {"id": "2", "type": "Organization", "properties": {"name": "Microsoft"}}
            ],
            "relationships": [
                {"source": "1", "target": "2", "type": "FOUNDER"}
            ]
        })
        
        mock_llm.invoke.side_effect = [
                mock_summary_response,
                mock_extraction_response 
                ]
        

        # --- Invoke the Worker Function ---
        response, status_code = worker(mock_request)

        # --- Assertions ---
        assert status_code == 200
        assert response == "OK"

        # Assert that Redis was called correctly
        mock_redis_client.rpush.assert_called_once()
        mock_redis_client.incr.assert_called_once_with("batch:test_batch_123:counter")

if __name__ == "__main__":
    pytest.main([__file__])