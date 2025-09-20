variable "project_id" {
  description = "The Google Cloud project ID."
  type        = string
}

variable "location" {
  description = "The GCP region for the Cloud Run service."
  type        = string
}

variable "region" {
  description = "The GCP region for the Cloud Run service."
  type        = string
}

variable "worker_sa_email" {
  description = "Service account email for the worker function."
  type        = string
}

variable "image_url" {
  description = "The URL of the Docker image for the worker function."
  type        = string
}

variable "image_tag" {
  description = "The tag for the Docker image."
  type        = string
}

variable "consolidation_topic" {
  description = "The name of the Pub/Sub topic for consolidation."
  type        = string
}

variable "worker_sa_key_secret_id" {
  description = "The Secret Manager ID for the worker service account key."
  type        = string
}

variable "vpc_connector" {
  description = "The VPC Access connector to use for the Cloud Run service."
  type        = string
}

variable "spanner_instance_id" {
  description = "The Spanner instance ID."
  type        = string
}

variable "spanner_database_id" {
  description = "The Spanner database ID."
  type        = string
}

variable "gemini_api_key" {
  description = "The Gemini API key."
  type        = string
  sensitive   = true
}

variable "llm_model_name" {
  description = "The name of the LLM model to use for the worker function."
  type        = string
}

variable "graph_data_bucket_name" {
  description = "The name of the GCS bucket for graph data."
  type        = string
}

variable "embedding_service_url" {
  description = "The URL of the embedding service."
  type        = string
}