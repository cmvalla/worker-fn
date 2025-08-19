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

variable "worker_sa_roles" {
  type = list(string)
  description = "Project-level IAM roles for the worker service account"
  default = [
    "roles/run.invoker",
    "roles/serviceusage.serviceUsageConsumer",
    "roles/cloudtrace.agent",
    "roles/spanner.databaseUser"
  ]
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
