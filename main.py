
import os
import re
import json
import logging
import time
import random
import igraph as ig
import pickle
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from google.cloud import pubsub_v1, storage
import google.auth
import functions_framework
from typing import Any, Dict, Optional, List, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END

from .config import Config
from .llm_operations import LLMOperations

# --- Basic Setup ---
# Setup structured logging for Google Cloud Logging
if os.environ.get("LOCAL_DEBUG", "false").lower() == "true":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("LOCAL_DEBUG is true. Logging to console.")
else:
    client = google.cloud.logging.Client()
    client.setup_logging(log_level=logging.DEBUG)
    logging.info("LOCAL_DEBUG is false. Logging to Cloud Logging.")

# --- Prompts ---
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
From the text below, extract entities and their relationships with extreme detail and verbosity. Your goal is to create a rich and comprehensive knowledge graph.

{previous_context}

**CRITICAL INSTRUCTIONS:**
1.  **Descriptive IDs:** You MUST create descriptive, human-readable IDs for all entities and relationships. For example, `person-john-doe`. You MUST NOT use generic, sequential IDs like `e1`, `e2`.
2.  **JSON Output:** Respond ONLY with a single, valid JSON object with two keys: "entities" and "relationships".

**Instructions for Extraction:**

**entities:**
- **id:** CRITICAL: Create a unique, descriptive, human-readable id. DO NOT use generic IDs like `e1`, `e2`.
- **type:** Assign a specific type (e.g., Person, Organization, Product).
- **properties:** Extract a comprehensive set of key-value pairs.
  - 'name': Must retain the original language from the text.
  - 'description': Provide a detailed description in English.

**relationships:**
- **id:** CRITICAL: Create a unique, descriptive id (e.g., 'person-john-doe-WORKS_FOR-organization-google'). DO NOT use generic IDs.
- **source:** The `id` of the source entity.
- **target:** The `id` of the target entity.
- **type:** Assign a descriptive type (e.g., WORKS_FOR, INVESTED_IN).
- **properties:**
  - 'confidence': A float score between 0.0 and 1.0.
  - 'description': A detailed explanation of the relationship.
"""),
    ("user",
     """TEXT:
---
{text_chunk}
---

JSON:
""")
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text in one concise sentence:"),
    ("user", """TEXT:
---
{text_chunk}
---

Summary:
""")
])

# --- Utility Functions ---
def extract_json_from_response(text: str) -> Dict[str, Any]:
    '''
    Extracts a JSON object from the model's text response, handling markdown and other prefixes.
    '''
    # Try to find markdown blocks and get the content inside
    match = re.search(r"```(json)?\s*({.*})```", text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(2).strip()
    else:
        # If no markdown, use the whole text
        json_str = text

    # Find the first '{' and the last '}' to isolate the JSON object
    start_index = json_str.find('{')
    end_index = json_str.rfind('}')

    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_str = json_str[start_index : end_index + 1]
    else:
        logging.error(f"Could not find a valid JSON object in the response. Raw text: '{text}'")
        raise ValueError("No JSON object found in response")

    try:
        extracted_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON. Raw text: '{json_str}'")
        raise ValueError("Invalid JSON format") from e

    if "entities" not in extracted_data or "relationships" not in extracted_data:
        raise ValueError("Invalid JSON schema: missing 'entities' or 'relationships' key")

    return extracted_data

def create_igraph_from_extracted_data(extracted_data: Dict[str, Any]) -> ig.Graph:
    graph = ig.Graph()
    entity_id_to_vertex_index: Dict[str, int] = {}
    for entity in extracted_data.get("entities", []):
        vertex = graph.add_vertex(name=entity["id"])
        for prop_key, prop_value in entity.get("properties", {}).items():
            vertex[prop_key] = prop_value
        vertex["type"] = entity["type"]
        if "embedding" in entity:
            vertex["embedding"] = entity["embedding"]
        if "cluster_embedding" in entity:
            vertex["cluster_embedding"] = entity["cluster_embedding"]
        entity_id_to_vertex_index[entity["id"]] = vertex.index

    for rel in extracted_data.get("relationships", []):
        source_id = rel.get("source")
        target_id = rel.get("target")
        if source_id in entity_id_to_vertex_index and target_id in entity_id_to_vertex_index:
            edge = graph.add_edge(entity_id_to_vertex_index[source_id], entity_id_to_vertex_index[target_id])
            for prop_key, prop_value in rel.get("properties", {}).items():
                edge[prop_key] = prop_value
            edge["type"] = rel["type"]
        else:
            logging.warning(f"Skipping relationship due to missing source or target entity: {rel}")
    return graph

# --- LangGraph State Definition ---

class GraphState(TypedDict):
    request_json: Dict[str, Any]
    text_chunk: str
    batch_id: str
    chunk_number: int
    total_chunks: int
    
    sentences: List[str]
    sentence_results: List[Dict[str, Any]]
    merged_data: Dict[str, Any]
    gcs_path: str
    
    # Clients and Config
    llm_json: ChatVertexAI
    llm_text: ChatVertexAI
    llm_ops: LLMOperations
    storage_client: storage.Client
    publisher_client: pubsub_v1.PublisherClient
    gcp_project: str
    consolidation_topic: str
    graph_data_bucket_name: str

# --- LangGraph Node Definitions ---

def split_text_node(state: GraphState) -> Dict[str, Any]:
    """Splits the input text chunk into sentences."""
    logging.info("Node: split_text_node")
    text_chunk = state["text_chunk"]
    sentences = re.split(r'(?<=[.!?])\s+', text_chunk)
    sentences = [s.strip() for s in sentences if s.strip()]
    logging.info(f"Split text into {len(sentences)} sentences.")
    return {"sentences": sentences}

def extraction_mapper(sentence_and_state: tuple[str, GraphState]) -> Dict[str, Any]:
    """
    A mapper function that calls the LLM for a single sentence.
    This is designed to be run in parallel for all sentences.
    """
    sentence, state = sentence_and_state
    logging.info(f"Processing sentence: {sentence[:80]}...")
    
    prompt_input = {"text_chunk": sentence, "previous_context": ""} # Context handling simplified for parallel runs
    llm_json = state["llm_json"]
    max_retries = 5
    
    current_prompt_template = EXTRACTION_PROMPT
    
    for attempt in range(max_retries):
        try:
            extraction_chain = current_prompt_template | llm_json
            llm_response = extraction_chain.invoke(prompt_input)
            return extract_json_from_response(llm_response.content)
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to get valid JSON for sentence on attempt {attempt + 1}/{max_retries}: {e}.")
            if attempt < max_retries - 1:
                invalid_output = llm_response.content if 'llm_response' in locals() else "N/A"
                sanitized_error = str(e).replace("{", "").replace("}", "")
                sanitized_output = str(invalid_output).replace("{", "").replace("}", "")
                
                correction_message = (
                    "The previous JSON output was invalid. "
                    "Please regenerate the JSON, ensuring it is valid and adheres to the specified schema."
                )
                
                correction_prompt_template = ChatPromptTemplate.from_messages([
                    ("system", correction_message),
                    ("user", "TEXT:\n---\n{text_chunk}\n---\n\nJSON:\n")
                ])
                current_prompt_template = correction_prompt_template
                time.sleep(1 * (2 ** attempt))
            else:
                logging.error(f"Failed to get valid JSON for sentence after {max_retries} attempts.")
                return {"entities": [], "relationships": []}
    return {"entities": [], "relationships": []}

def parallel_extraction_node(state: GraphState) -> Dict[str, Any]:
    """
    Processes all sentences in parallel using the extraction_mapper.
    """
    logging.info("Node: parallel_extraction_node")
    
    sentences = state["sentences"]
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create a list of tuples, each containing a sentence and the state
        tasks = [(sentence, state) for sentence in sentences]
        
        future_to_sentence = {executor.submit(extraction_mapper, task): task[0] for task in tasks}
        for future in as_completed(future_to_sentence):
            sentence = future_to_sentence[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                logging.error(f'Sentence "{sentence}" generated an exception: {exc}')
                results.append({"entities": [], "relationships": []}) 
                
    logging.info(f"Finished parallel extraction for {len(sentences)} sentences.")
    return {"sentence_results": results}

def merge_results_node(state: GraphState) -> Dict[str, Any]:
    logging.info("Node: merge_results_node")
    
    merged_data = {"entities": [], "relationships": []}
    entity_map = {} # Map entity_id to the actual entity object

    for result in state["sentence_results"]:
        for entity in result.get("entities", []):
            entity_id = entity.get("id")
            if not entity_id:
                logging.warning(f"Skipping entity due to missing 'id' key during merge: {entity}")
                continue

            if entity_id in entity_map:
                # Duplicate ID found, merge properties
                existing_entity = entity_map[entity_id]
                existing_entity["properties"].update(entity.get("properties", {}))
                # Optionally, merge embeddings or other fields if needed
                # For now, we'll just update properties
            else:
                # New entity, add it
                entity_map[entity_id] = entity
                merged_data["entities"].append(entity)
        
        # Relationships are simply extended, assuming they are unique by (source, target, type)
        # If relationships can have duplicate IDs, a similar merge logic would be needed.
        merged_data["relationships"].extend(result.get("relationships", []))
        
    state["merged_data"] = merged_data
    logging.info(f"Merged results: {len(merged_data['entities'])} entities, {len(merged_data['relationships'])} relationships.")
    return {"merged_data": merged_data}

def final_processing_node(state: GraphState) -> Dict[str, Any]:
    """
    Performs the final steps after extraction and merging.
    """
    logging.info("Node: final_processing_node")
    
    extracted_data = state["merged_data"]
    llm_text = state["llm_text"]
    llm_ops = state["llm_ops"]
    batch_id = state["batch_id"]
    chunk_number = state["chunk_number"]
    text_chunk = state["text_chunk"]
    graph_data_bucket_name = state["graph_data_bucket_name"]
    storage_client = state["storage_client"]
    publisher_client = state["publisher_client"]
    gcp_project = state["gcp_project"]
    consolidation_topic = state["consolidation_topic"]

    # Generate summary for the chunk
    summary_chain = SUMMARY_PROMPT | llm_text
    summary_response = summary_chain.invoke({"text_chunk": text_chunk})
    chunk_summary = str(summary_response.content)

    # Create "Chunk" and "Community" entities
    chunk_entity_id = f"chunk-{batch_id}-{chunk_number}"
    extracted_data["entities"].append({
        "id": chunk_entity_id,
        "type": "Chunk",
        "properties": {
            "original_text": text_chunk,
            "summary": chunk_summary,
            "chunk_number": chunk_number
        }
    })
    extracted_data["entities"].append({
        "id": f"community-{batch_id}",
        "type": "Community",
        "properties": {
            "name": f"Community for batch {batch_id}",
            "description": f"Community entity for batch {batch_id}"
        }
    })

    # Generate embeddings
    extracted_data = llm_ops.generate_embeddings(extracted_data)

    if os.environ.get("LOCAL_DEBUG", "false").lower() == "true":
        logging.info("--- Entities (first 5) ---")
        for i, entity in enumerate(extracted_data["entities"][:5]):
            logging.info(f"  {i+1}. ID: {entity.get('id')}, Type: {entity.get('type')}, Properties: {entity.get('properties')}")
        logging.info("--- Relationships (first 5) ---")
        for i, rel in enumerate(extracted_data["relationships"][:5]):
            logging.info(f"  {i+1}. Source: {rel.get('source')}, Target: {rel.get('target')}, Type: {rel.get('type')}, Properties: {rel.get('properties')}")

    # Create relationships to Chunk entity
    for entity in extracted_data["entities"]:
        if entity["id"] != chunk_entity_id:
            extracted_data["relationships"].append({
                "source": entity["id"],
                "target": chunk_entity_id,
                "type": "PART_OF",
                "properties": {"weight": 1.0, "description": "Entity is part of this text chunk."}
            })

    # Create and save igraph
    graph = create_igraph_from_extracted_data(extracted_data)
    serialized_graph = pickle.dumps(graph)
    gcs_object_name = f"graph_data/{batch_id}/{chunk_number}_{int(time.time())}.pkl"
    gcs_path = f"gs://{graph_data_bucket_name}/{gcs_object_name}"
    bucket = storage_client.bucket(graph_data_bucket_name)
    blob = bucket.blob(gcs_object_name)
    blob.upload_from_string(serialized_graph)
    logging.info(f"Uploaded serialized igraph to {gcs_path}")

    # Publish to Pub/Sub
    topic_path = publisher_client.topic_path(gcp_project, consolidation_topic)
    message_data = json.dumps({
        "batch_id": batch_id, 
        "gcs_paths": [gcs_path], 
        "chunk_number": chunk_number, 
        "total_chunks": state["total_chunks"]
    }).encode("utf-8")
    future = publisher_client.publish(topic_path, data=message_data)
    future.result()
    logging.info(f"Published message for chunk {chunk_number} to {topic_path}.")

    return {"gcs_path": gcs_path}


# --- Graph Definition ---
workflow = StateGraph(GraphState)
workflow.add_node("split_text", split_text_node)
workflow.add_node("parallel_extraction", parallel_extraction_node)
workflow.add_node("merge_results", merge_results_node)
workflow.add_node("final_processing", final_processing_node)

workflow.set_entry_point("split_text")
workflow.add_edge("split_text", "parallel_extraction")
workflow.add_edge("parallel_extraction", "merge_results")
workflow.add_edge("merge_results", "final_processing")
workflow.add_edge("final_processing", END)

app = workflow.compile()

# --- Main Worker Function ---

@functions_framework.http
def worker(request: Any) -> tuple[str, int]:
    try:
        # 1. Parse request
        data_str = request.get_data(as_text=True)
        if not data_str:
            logging.error("Request body is empty.")
            return "Bad Request: Empty body", 400
        request_json = json.loads(data_str)

        text_chunk = request_json.get("chunk")
        batch_id = request_json.get("batch_id")
        total_chunks = request_json.get("total_chunks")
        chunk_number = request_json.get("chunk_number", 0)

        if not all([text_chunk, batch_id, total_chunks is not None]):
            return "Bad Request: Missing required fields", 400

        # 2. Initialize clients and config
        gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GCP_LOCATION", "europe-west1")
        consolidation_topic = os.environ.get("CONSOLIDATION_TOPIC")
        graph_data_bucket_name = os.environ.get("GRAPH_DATA_BUCKET_NAME")
        llm_model_name = os.environ.get("LLM_MODEL_NAME", "gemini-1.5-flash")

        if not all([gcp_project, consolidation_topic, graph_data_bucket_name]):
            logging.critical("FATAL: Missing critical environment variables.")
            return "ERROR: Configuration missing", 500

        llm_json = ChatVertexAI(
            project=Config.GCP_PROJECT,
            location=location,
            model_name=llm_model_name,
            max_output_tokens=8192,
            generation_config={"response_mime_type": "application/json"}
        )
        llm_text = ChatVertexAI(
            project=Config.GCP_PROJECT,
            location=location,
            model_name=llm_model_name,
            temperature=0.5,
            max_output_tokens=2048
        )
        llm_ops = LLMOperations(llm_text)
        storage_client = storage.Client()
        publisher_client = pubsub_v1.PublisherClient()

        # 3. Prepare initial state
        initial_state = {
            "request_json": request_json,
            "text_chunk": text_chunk,
            "batch_id": batch_id,
            "chunk_number": chunk_number,
            "total_chunks": total_chunks,
            "llm_json": llm_json,
            "llm_text": llm_text,
            "llm_ops": llm_ops,
            "storage_client": storage_client,
            "publisher_client": publisher_client,
            "gcp_project": gcp_project,
            "consolidation_topic": consolidation_topic,
            "graph_data_bucket_name": graph_data_bucket_name,
        }

        # 4. Invoke the graph
        final_state = app.invoke(initial_state)

        logging.info(f"Worker finished successfully for batch {batch_id}, chunk {chunk_number}. Final GCS path: {final_state.get('gcs_path')}")
        return "OK", 200

    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from request body: {e}")
        return "Bad Request: Invalid JSON", 400
    except Exception as e:
        logging.error(f"An unexpected error occurred in worker: {e}", exc_info=True)
        return "Internal Server Error", 500
