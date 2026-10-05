# FleetSense Cloud Deployment Guide

This guide details how to deploy the **FleetSense ML Platform** to **Microsoft Azure** or **Amazon Web Services (AWS)**.

---

## 1. Local Testing with Docker Compose

Before deploying to the cloud, you can test the full container stack (API, Prometheus, Grafana) locally:

```bash
# Build and run containers
docker-compose up --build -d

# Check running services
docker-compose ps

# Verify endpoints:
# - API Health:  http://localhost:8000/health
# - API Swagger: http://localhost:8000/docs
# - Prometheus:  http://localhost:9090
# - Grafana:     http://localhost:3000 (User/Password: admin/admin)
```

To stop the services:
```bash
docker-compose down
```

---

## 2. Deploying to Microsoft Azure (Recommended)

### Architecture
- **Compute**: Azure Container Apps (serverless, auto-scales from 1 to 5 replicas)
- **Container Registry**: Azure Container Registry (ACR)
- **Storage**: Azure Blob Storage (for raw TLC Parquet files and DVC remote)
- **Monitoring**: Azure Log Analytics Workspace + Prometheus `/metrics` endpoint

### Option A: Automated CLI Script
```bash
# Authenticate with Azure
az login

# Set custom parameters (optional)
export AZURE_RESOURCE_GROUP="fleetsense-rg"
export AZURE_LOCATION="eastus"

# Run automated deployment
chmod +x deploy/azure/deploy.sh
./deploy/azure/deploy.sh
```

### Option B: Infrastructure-as-Code (Bicep)
```bash
az group create --name fleetsense-rg --location eastus

az deployment group create \
  --resource-group fleetsense-rg \
  --template-file deploy/azure/main.bicep \
  --parameters prefix=fleetsense
```

---

## 3. Deploying to Amazon Web Services (AWS)

### Architecture
- **Compute**: AWS App Runner (fully managed container runner with automatic HTTPS & scaling)
- **Registry**: AWS Elastic Container Registry (ECR)
- **Storage**: Amazon S3 (for DVC remote & dataset storage)

### Automated Deployment
```bash
# Configure AWS CLI
aws configure

# Run automated deployment
chmod +x deploy/aws/deploy.sh
./deploy/aws/deploy.sh
```

---

## 4. DVC Remote Configuration in Cloud

To push and pull datasets from cloud remotes:

### Azure Blob Storage Remote
```bash
dvc remote add -d azure_remote azure://fleetsense-data/dvc
dvc remote modify azure_remote connection_string "$AZURE_STORAGE_CONNECTION_STRING"
dvc push
```

### AWS S3 Remote
```bash
dvc remote add -d s3_remote s3://fleetsense-data-bucket/dvc
dvc push
```
