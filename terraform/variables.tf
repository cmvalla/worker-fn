variable "project_id" {
  type        = string
  description = "The GCP project ID."
}

variable "region" {
  type        = string
  description = "The GCP region."
}

variable "location" {
  type        = string
  description = "The GCP location."
}

variable "repository_id" {
  type        = string
  description = "The Artifact Registry repository ID."
}

variable "image_name" {
  type        = string
  description = "The name of the Docker image."
}

variable "image_tag" {
  type        = string
  description = "The tag of the Docker image, typically the Build ID."
}

variable "image_url" {
  type        = string
  description = "The full URL of the Docker image to deploy."
}

variable "worker_sa_email" {
  type        = string
  description = "The email of the service account for the worker function."
}

variable "redis_host" {
  type        = string
  description = "The Redis host IP address."
}

variable "redis_port" {
  type        = string
  description = "The Redis port."
}

variable "spanner_instance" {
  type        = string
  description = "The Spanner instance ID."
}

variable "spanner_database" {
  type        = string
  description = "The Spanner database ID."
}

variable "consolidation_topic" {
  type        = string
  description = "The Pub/Sub topic for consolidation."
}

variable "vpc_connector" {
  type        = string
  description = "The Serverless VPC Access connector name."
}

variable "worker_sa_key_secret_id" {
  type        = string
  description = "The full resource name of the Secret Manager secret containing the worker service account key."
}
