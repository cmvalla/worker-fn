import os
import json
import pytest
from unittest.mock import Mock, patch

@patch("main.get_redis_password", return_value="test_password")
@patch("main.get_redis_client")
@patch("langchain_google_vertexai.chat_models.ChatVertexAI")
def test_worker(MockChatVertexAI, MockGetRedisClient, MockGetRedisPassword):
    """
    Tests the worker function.
    """
    # Set dummy environment variables for the test
    os.environ["REDIS_HOST"] = "mock_redis_host"
    os.environ["CONSOLIDATION_TOPIC"] = "mock_consolidation_topic"

    # --- Mock the Redis Client ---
    mock_redis_instance = Mock()
    mock_redis_instance.ping.return_value = True
    mock_redis_instance.rpush.return_value = 1
    mock_redis_instance.incr.return_value = 1
    MockGetRedisClient.return_value = mock_redis_instance

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

    # --- Import the worker function ---
    from main import worker

    # --- Mock the external clients ---
    with patch("main.pubsub_v1.PublisherClient") as MockPublisherClient:

        mock_publisher_client = Mock()
        MockPublisherClient.return_value = mock_publisher_client
        mock_publisher_client.topic_path.return_value = "projects/test-project/topics/test-topic"
        mock_publisher_client.publish.return_value.result.return_value = "test_message_id"

        # --- Mock the LLM responses ---
        mock_summary_response_content = "Bill Gates co-founded Microsoft."
        mock_extraction_response_content = json.dumps({
            "entities": [
                {"id": "1", "type": "Person", "properties": {"name": "Bill Gates"}},
                {"id": "2", "type": "Organization", "properties": {"name": "Microsoft"}}
            ],
            "relationships": [
                {"source": "1", "target": "2", "type": "FOUNDER"}
            ]
        })
        
        # Configure the mock ChatVertexAI instance for llm_text
        mock_llm_text_response_obj = mock_summary_response.content
        mock_llm_json_response_obj = mock_extraction_response.content
        MockChatVertexAI.return_value.invoke.side_effect = [mock_llm_text_response_obj, mock_llm_json_response_obj]
        

        # --- Invoke the Worker Function ---
        response, status_code = worker(mock_request)

        # --- Assertions ---
        assert status_code == 200
        assert response == "OK"

        # Assert that Redis was called correctly
        mock_redis_instance.rpush.assert_called_once()
        mock_redis_instance.incr.assert_called_once_with("batch:test_batch_123:counter")

if __name__ == "__main__":
    pytest.main([__file__])
