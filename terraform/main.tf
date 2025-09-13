data "google_service_account" "worker_sa" {
  account_id = "worker-sa"
  project    = var.project_id
}

data "google_secret_manager_secret_version" "worker_sa_key" {
  project = var.project_id
  secret  = var.worker_sa_key_secret_id
}



resource "google_cloud_run_v2_service" "worker" {
  name     = "worker-fn"
  location = var.location
  deletion_protection = false
  ingress  = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  timeouts {
    create = "3m"
    update = "3m"
  }

  traffic {
    type = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
  template {
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
    vpc_access {
      connector = var.vpc_connector
      egress = "ALL_TRAFFIC"
    }
    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }
    max_instance_request_concurrency = 50
    volumes {
      name = "secret-volume"
      secret {
        secret = data.google_secret_manager_secret_version.worker_sa_key.secret
        items {
          path    = "credentials.json"
          version = "latest"
        }
      }
    }

    containers {
      image = "${var.image_url}:${var.image_tag}"

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
        name  = "SPANNER_INSTANCE_ID"
        value = var.spanner_instance_id
      }
      env {
        name  = "SPANNER_DATABASE_ID"
        value = var.spanner_database_id
      }
      env {
        name = "CONSOLIDATION_TOPIC"
        value = var.consolidation_topic
      }
      env {
        name  = "LLM_MODEL_NAME"
        value = var.llm_model_name
      }
    }
  }
}
