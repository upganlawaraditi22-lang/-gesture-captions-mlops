variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "github_repository" {
  description = "GitHub repository allowed to deploy, as owner/name"
  type        = string

  validation {
    condition     = can(regex("^[^/]+/[^/]+$", var.github_repository))
    error_message = "Use the format owner/name."
  }
}

variable "project_name" {
  description = "Prefix for Azure resource names"
  type        = string
  default     = "gesture"
}

variable "location" {
  description = "Azure region. Student subscriptions may allow only certain regions."
  type        = string
  default     = "centralindia"
}

variable "node_vm_size" {
  description = "VM size for the cluster node"
  type        = string
  default     = "Standard_B2ms"
}

variable "node_count" {
  description = "Number of cluster nodes"
  type        = number
  default     = 1
}
