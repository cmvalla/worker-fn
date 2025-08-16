import functions_framework
import google.cloud.logging
import logging

# Setup structured logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()

@functions_framework.http
def worker(request):
    """This function is a stub for the worker.

    It logs the received request and returns a successful response.
    """
    request_json = request.get_json(silent=True)
    logging.info(f"Received request: {request_json}")
    return "OK", 200