variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "monopoly-deal-agent"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "my-cluster-1"
}