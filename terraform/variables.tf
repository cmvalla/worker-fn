variable "project_id" {
  description = "The Google Cloud project ID."
  type        = string
}

variable "location" {
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

variable "redis_host" {
  description = "The Redis host."
  type        = string
}

variable "redis_port" {
  description = "The Redis port."
  type        = number
}

variable "consolidation_topic" {
  description = "The name of the Pub/Sub topic for consolidation."
  type        = string
}

variable "llm_model_name" {
  description = "The name of the LLM model to use for the worker function."
  type        = string
}