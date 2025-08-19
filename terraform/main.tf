resource "google_project_service" "cloudresourcemanager" {
  project = var.project_id
  service = "cloudresourcemanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_service_account" "worker_sa" {
  project      = var.project_id
  account_id   = "worker-sa"
  display_name = "Service Account for GraphRAG Worker Function"
}

resource "google_project_iam_member" "worker_sa_roles" {
  count   = length(var.worker_sa_roles)
  project = var.project_id
  role    = var.worker_sa_roles[count.index]
  member  = "serviceAccount:${google_service_account.worker_sa.email}"
  depends_on = [google_project_service.cloudresourcemanager]
}

resource "google_cloud_run_v2_service" "worker" {
  project  = var.project_id
  name     = "worker-fn"
  location = var.location
  ingress  = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  template {
    service_account = google_service_account.worker_sa.email
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
