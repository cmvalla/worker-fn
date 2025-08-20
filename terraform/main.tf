resource "google_cloud_run_v2_service" "worker" {
  project  = var.project_id
  name     = "worker-fn"
  location = var.location
  deletion_protection = false
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = var.worker_sa_email
    timeout         = "900s" # 15 minutes
    scaling {
      min_instance_count = 1
      max_instance_count = 10 # Increased for parallel processing
    }
    containers {
      image = "${var.location}-docker.pkg.dev/${var.project_id}/${var.repository_id}/${var.image_name}:${var.image_tag}"
      ports {
        container_port = 8080
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "REDIS_HOST"
        value = var.redis_host
      }
      env {
        name  = "REDIS_PORT"
        value = var.redis_port
      }
      env {
        name  = "SPANNER_INSTANCE"
        value = var.spanner_instance
      }
      env {
        name  = "SPANNER_DATABASE"
        value = var.spanner_database
      }
    }
  }
}