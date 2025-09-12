import os
import json
import pytest
from unittest.mock import Mock, patch


@patch.dict(os.environ, {"REDIS_HOST": "mock_redis_host", "CONSOLIDATION_TOPIC": "mock_consolidation_topic"})
@patch("main.get_redis_password", return_value="mock_redis_password")
@patch("main.redis.Redis")
@patch("langchain_google_vertexai.chat_models.ChatVertexAI")
@patch("google.cloud.pubsub_v1.PublisherClient")
def test_worker_local_integration(MockPublisherClient, MockChatVertexAI, MockRedis, MockGetRedisPassword):
    """
    Tests the worker function locally using mocked Redis and Secret Manager.
    """
    from main import worker, get_redis_client, get_redis_password

    # --- Mock Redis Client ---
    mock_redis_instance = Mock()
    mock_redis_instance.ping.return_value = True
    mock_redis_instance.rpush.return_value = 1
    mock_redis_instance.incr.return_value = 1
    MockRedis.return_value = mock_redis_instance

    # --- Mock LLM responses ---
    mock_summary_response_content = "Mocked summary."
    mock_extraction_response_content = json.dumps({
        "entities": [],
        "relationships": []
    })

    # Create separate mock instances for llm_text and llm_json
    # Create separate mock instances for llm_text and llm_json
    mock_llm_text_instance = Mock()
    mock_llm_text_instance.invoke.return_value = mock_summary_response.content

    mock_llm_json_instance = Mock()
    mock_llm_json_instance.invoke.return_value = mock_extraction_response.content

    # Configure MockChatVertexAI to return these instances in the correct order
    MockChatVertexAI.side_effect = [mock_llm_json_instance, mock_llm_text_instance]

    # --- Mock Publisher Client ---
    mock_publisher_instance = Mock()
    mock_publisher_instance.topic_path.return_value = "projects/test-project/topics/test-topic"
    mock_publisher_instance.publish.return_value.result.return_value = "test_message_id"
    MockPublisherClient.return_value = mock_publisher_instance

    # --- Simulate a Cloud Tasks Request ---
    mock_request = Mock()
    mock_request.get_json.return_value = {
        "chunk": {
            "page_content": "Bill Gates is the co-founder of Microsoft."
        },
        "batch_id": "test_batch_id",
        "total_chunks": 1,
        "chunk_number": 0
    }

    # --- Invoke the Worker Function ---
    response, status_code = worker(mock_request)

    # --- Assertions ---
    assert status_code == 200
    assert response == "OK"
    MockGetRedisPassword.assert_called_once()
    MockRedis.assert_called_once_with(
        host=os.environ.get("REDIS_HOST"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        password="mock_redis_password",
        socket_connect_timeout=10,
        ssl=False
    )
    mock_redis_instance.rpush.assert_called_once()
    mock_redis_instance.incr.assert_called_once()
    MockChatVertexAI.return_value.invoke.assert_called_once()
    MockPublisherClient.assert_called_once()