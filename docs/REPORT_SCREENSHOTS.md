# Screenshots for the report

| # | Stage | What to capture |
|---|---|---|
| 1 | GitHub | Repository page and commit history |
| 2 | CI | Passing `CI` run showing the test, docker, and infrastructure jobs |
| 3 | Terraform | `terraform plan` summary and `terraform apply` complete message |
| 4 | Azure | Resource group showing the registry and Kubernetes cluster |
| 5 | Registry | `gesture-captions` image in Azure Container Registry with the commit tag |
| 6 | CD | Passing `Deploy to Azure` run |
| 7 | Kubernetes | `kubectl get pods,svc -n gesture-captions` |
| 8 | Self-healing | Delete the pod, then show its replacement with `kubectl get pods -n gesture-captions -w` |
| 9 | Prometheus | Status > Targets showing `gesture-captions` as UP |
| 10 | Grafana | Gesture Captions dashboard with live traffic |
| 11 | App | Public HTTPS address showing captions from your trained model |
