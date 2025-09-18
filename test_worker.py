import os
import json
import pytest
from unittest.mock import Mock, patch

@patch("langchain_google_vertexai.chat_models.ChatVertexAI")
def test_worker(MockChatVertexAI):
    """
    Tests the worker function.
    """
    # Set dummy environment variables for the test
    os.environ["CONSOLIDATION_TOPIC"] = "mock_consolidation_topic"
    os.environ["GRAPH_DATA_BUCKET_NAME"] = "mock_graph_data_bucket"

    # --- Mock the input data ---
    mock_request = Mock()
    mock_request.get_data.return_value = json.dumps({
            "chunk": "Bill Gates is the co-founder of Microsoft.",
            "batch_id": "test_batch_123",
            "total_chunks": 1,
            "chunk_number": 0
        }).encode("utf-8")

    # --- Import the worker function ---
    from main import worker

    # --- Mock the external clients ---
    with patch("main.pubsub_v1.PublisherClient") as MockPublisherClient:
        with patch("main.storage.Client") as MockStorageClient:

            mock_publisher_client = Mock()
            MockPublisherClient.return_value = mock_publisher_client
            mock_publisher_client.topic_path.return_value = "projects/test-project/topics/test-topic"
            mock_publisher_client.publish.return_value.result.return_value = "test_message_id"

            mock_storage_client = Mock()
            MockStorageClient.return_value = mock_storage_client
            mock_bucket = Mock()
            mock_storage_client.bucket.return_value = mock_bucket
            mock_blob = Mock()
            mock_bucket.blob.return_value = mock_blob

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
            
            mock_summary_response = MagicMock()
            mock_summary_response.content = mock_summary_response_content
            mock_extraction_response = MagicMock()
            mock_extraction_response.content = mock_extraction_response_content
            
            # Configure the mock ChatVertexAI instance for llm_text
            MockChatVertexAI.return_value.invoke.side_effect = [mock_summary_response, mock_extraction_response]
            

            # --- Invoke the Worker Function ---
            response, status_code = worker(mock_request)

            # --- Assertions ---
            assert status_code == 200
            assert response == "OK"

            # Assert GCS upload
            mock_bucket.blob.assert_called_once()
            mock_blob.upload_from_string.assert_called_once()

            # Assert Pub/Sub message
            mock_publisher_client.publish.assert_called_once()
            published_message_data = json.loads(mock_publisher_client.publish.call_args[1]["data"].decode("utf-8"))
            assert published_message_data["batch_id"] == "test_batch_123"
            assert "gcs_path" in published_message_data
            assert published_message_data["chunk_number"] == 0
            assert published_message_data["total_chunks"] == 1

if __name__ == "__main__":
    pytest.main([__file__])
