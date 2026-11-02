# Manages the app on Docker Desktop Kubernetes. It's separate from the Azure setup in infra/terraform.
terraform {
  required_version = ">= 1.6"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.35"
    }
  }
}

provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "docker-desktop"
}

locals {
  labels = { app = "gesture-captions-tf" }
}

resource "kubernetes_namespace_v1" "app" {
  metadata {
    name = "gesture-captions-tf"
  }
}

resource "kubernetes_deployment_v1" "app" {
  metadata {
    name      = "gesture-captions"
    namespace = kubernetes_namespace_v1.app.metadata[0].name
    labels    = local.labels
  }

  spec {
    replicas = 1

    selector {
      match_labels = local.labels
    }

    template {
      metadata {
        labels = local.labels
      }

      spec {
        security_context {
          run_as_non_root = true
          run_as_user     = "10001"
        }

        container {
          name              = "app"
          image             = "gesture-captions:local"
          image_pull_policy = "Never"

          env {
            name  = "MODEL_VERSION"
            value = "terraform-local"
          }

          port {
            name           = "http"
            container_port = 8501
          }

          port {
            name           = "metrics"
            container_port = 9100
          }

          readiness_probe {
            http_get {
              path = "/_stcore/health"
              port = 8501
            }
            initial_delay_seconds = 5
            period_seconds        = 10
          }

          resources {
            requests = {
              cpu    = "500m"
              memory = "512Mi"
            }
            limits = {
              memory = "1536Mi"
            }
          }
        }
      }
    }
  }
}

resource "kubernetes_service_v1" "app" {
  metadata {
    name      = "gesture-captions"
    namespace = kubernetes_namespace_v1.app.metadata[0].name
    labels    = local.labels
  }

  spec {
    selector = local.labels

    port {
      name        = "http"
      port        = 8501
      target_port = 8501
    }

    port {
      name        = "metrics"
      port        = 9100
      target_port = 9100
    }
  }
}

output "open_app" {
  value = "kubectl port-forward -n ${kubernetes_namespace_v1.app.metadata[0].name} svc/${kubernetes_service_v1.app.metadata[0].name} 8502:8501"
}
