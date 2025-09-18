import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock


@patch.dict(os.environ, {"CONSOLIDATION_TOPIC": "mock_consolidation_topic", "GRAPH_DATA_BUCKET_NAME": "mock_graph_data_bucket"})
@patch("langchain_google_vertexai.chat_models.ChatVertexAI")
@patch("google.cloud.pubsub_v1.PublisherClient")
@patch("google.cloud.storage.Client")
def test_worker_local_integration(MockStorageClient, MockPublisherClient, MockChatVertexAI):
    """
    Tests the worker function locally using mocked Redis and Secret Manager.
    """
    from main import worker

    # --- Mock LLM responses ---
    mock_summary_response_content = "Mocked summary."
    mock_extraction_response_content = json.dumps({
        "entities": [],
        "relationships": []
    })

    # Create separate mock instances for llm_text and llm_json
    mock_llm_text_instance = Mock()
    mock_llm_text_instance.invoke.return_value = Mock(content=mock_summary_response_content)

    mock_llm_json_instance = Mock()
    mock_llm_json_instance.invoke.return_value = Mock(content=mock_extraction_response_content)

    # Configure MockChatVertexAI to return these instances in the correct order
    MockChatVertexAI.return_value.invoke.side_effect = [mock_llm_text_instance.invoke.return_value, mock_llm_json_instance.invoke.return_value]

    # --- Mock Publisher Client ---
    mock_publisher_instance = Mock()
    mock_publisher_instance.topic_path.return_value = "projects/test-project/topics/test-topic"
    mock_publisher_instance.publish.return_value.result.return_value = "test_message_id"
    MockPublisherClient.return_value = mock_publisher_instance

    # --- Mock Storage Client ---
    mock_storage_instance = Mock()
    MockStorageClient.return_value = mock_storage_instance
    mock_bucket = Mock()
    mock_storage_instance.bucket.return_value = mock_bucket
    mock_blob = Mock()
    mock_bucket.blob.return_value = mock_blob

    # --- Simulate a Cloud Tasks Request ---
    mock_request = Mock()
    mock_request.get_data.return_value = json.dumps({
        "chunk": "Bill Gates is the co-founder of Microsoft.",
        "batch_id": "test_batch_id",
        "total_chunks": 1,
        "chunk_number": 0
    }).encode("utf-8")

    # --- Invoke the Worker Function ---
    response, status_code = worker(mock_request)

    # --- Assertions ---
    assert status_code == 200
    assert response == "OK"

    # Assert GCS upload
    mock_bucket.blob.assert_called_once()
    mock_blob.upload_from_string.assert_called_once()

    # Assert Pub/Sub message
    mock_publisher_instance.publish.assert_called_once()
    published_message_data = json.loads(mock_publisher_instance.publish.call_args[1]["data"].decode("utf-8"))
    assert published_message_data["batch_id"] == "test_batch_id"
    assert "gcs_path" in published_message_data
    assert published_message_data["chunk_number"] == 0
    assert published_message_data["total_chunks"] == 1