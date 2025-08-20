terraform {
  backend "gcs" {
    bucket = "terraform-goreply-devops-tools"
    prefix = "cloud-spanner-demo/worker/${terraform.workspace}"
  }
}