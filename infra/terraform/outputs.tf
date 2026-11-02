output "resource_group" {
  value = azurerm_resource_group.main.name
}

output "aks_name" {
  value = azurerm_kubernetes_cluster.main.name
}

output "app_url" {
  value = "https://${local.dns_label}.${azurerm_resource_group.main.location}.cloudapp.azure.com"
}

output "github_settings" {
  description = "Values to add under GitHub repository Settings > Secrets and variables > Actions"
  value = {
    secrets = {
      AZURE_CLIENT_ID       = azurerm_user_assigned_identity.github.client_id
      AZURE_TENANT_ID       = azurerm_user_assigned_identity.github.tenant_id
      AZURE_SUBSCRIPTION_ID = var.subscription_id
    }
    variables = {
      ACR_NAME       = azurerm_container_registry.main.name
      AKS_NAME       = azurerm_kubernetes_cluster.main.name
      RESOURCE_GROUP = azurerm_resource_group.main.name
      APP_DNS_LABEL  = local.dns_label
      AZURE_REGION   = azurerm_resource_group.main.location
    }
  }
}
