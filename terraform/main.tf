data "google_service_account" "worker_sa" {
  account_id = "worker-sa"
  project    = var.project_id
}

resource "google_cloud_run_v2_service" "worker" {
  project  = var.project_id
  name     = "worker-fn"
  location = var.location
  deletion_protection = false
  ingress  = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  template {
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
    
    volumes {
      name = "secret-volume"
      secret {
        secret = "worker-sa-key"
        items {
          path    = "credentials.json"
          version = "latest"
        }
      }
    }

    containers {
      image = "${var.location}-docker.pkg.dev/${var.project_id}/${var.repository_id}/${var.image_name}:${var.image_tag}"
      
      ports {
        container_port = 8080
      }

      volume_mounts {
        name       = "secret-volume"
        mount_path = "/app/creds"
      }

      env {
        name  = "GOOGLE_APPLICATION_CREDENTIALS"
        value = "/app/creds/credentials.json"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name = "GCP_LOCATION"
        value = var.location
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
      env {
        name = "REDIS_PASSWORD"
        value = var.redis_password
      }
      env {
        name = "CONSOLIDATION_TOPIC"
        value = var.consolidation_topic
      }
    }
  }
}
