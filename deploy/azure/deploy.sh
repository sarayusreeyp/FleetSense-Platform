#!/usr/bin/env bash
# ==============================================================================
# FleetSense - Azure Container Apps Automated Deployment Script
# Builds and deploys:
#   1. fleetsense-api (ML Inference Service)
#   2. fleetsense-frontend (Web UI)
#   3. fleetsense-prometheus (Internal Metrics Scraper)
#   4. fleetsense-grafana (Metrics Dashboard)
# ==============================================================================
set -euo pipefail

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-fleetsense-rg}"
LOCATION="${AZURE_LOCATION:-centralindia}"
PREFIX="${FLEETSENSE_PREFIX:-fleetsense}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d%H%M%S)}"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-FleetSense2026!}"
AZURE_STORAGE_CONNECTION_STRING="${AZURE_STORAGE_CONNECTION_STRING:-}"
AZURE_CONTAINER_NAME="${AZURE_CONTAINER_NAME:-fleetsense-production}"

# Auto-load connection string from .env if not explicitly passed
if [ -z "${AZURE_STORAGE_CONNECTION_STRING}" ] && [ -f ".env" ]; then
    AZURE_STORAGE_CONNECTION_STRING=$(grep -E '^AZURE_STORAGE_CONNECTION_STRING=' .env | cut -d '=' -f 2- | tr -d '\r\n')
fi

echo "=========================================================="
echo " Starting FleetSense 4-Tier Azure Deployment"
echo " Resource Group: ${RESOURCE_GROUP} (${LOCATION})"
echo "=========================================================="

# 1. Create Resource Group
echo "[1/4] Ensuring Azure Resource Group exists..."
az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" --output table

# 2. Deploy Core Infrastructure (Storage, ACR, ACA Environment) via Bicep
echo "[2/4] Deploying core infrastructure (Storage, ACR, Environment)..."
az deployment group create \
    --resource-group "${RESOURCE_GROUP}" \
    --template-file deploy/azure/infra.bicep \
    --parameters prefix="${PREFIX}" \
    --output none

ACR_SERVER=$(az deployment group show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "infra" \
    --query "properties.outputs.acrLoginServer.value" \
    --output tsv)

ACR_NAME=$(az deployment group show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "infra" \
    --query "properties.outputs.acrName.value" \
    --output tsv)

echo "ACR Server: ${ACR_SERVER}"

# 3. Build & push all 4 container images to ACR
echo "[3/4] Building and pushing Docker images to ${ACR_NAME}..."

# Check if local Docker daemon is running
if docker info >/dev/null 2>&1; then
    echo "Using local Docker to build and push images directly to ${ACR_SERVER}..."
    az acr login --name "${ACR_NAME}"

    echo "-> Building & pushing fleetsense-api..."
    docker build -t "${ACR_SERVER}/fleetsense-api:${IMAGE_TAG}" -f Dockerfile .
    docker push "${ACR_SERVER}/fleetsense-api:${IMAGE_TAG}"

    echo "-> Building & pushing fleetsense-frontend..."
    docker build -t "${ACR_SERVER}/fleetsense-frontend:${IMAGE_TAG}" -f Dockerfile.frontend .
    docker push "${ACR_SERVER}/fleetsense-frontend:${IMAGE_TAG}"

    echo "-> Building & pushing fleetsense-prometheus..."
    docker build -t "${ACR_SERVER}/fleetsense-prometheus:${IMAGE_TAG}" -f deploy/monitoring/Dockerfile.prometheus .
    docker push "${ACR_SERVER}/fleetsense-prometheus:${IMAGE_TAG}"

    echo "-> Building & pushing fleetsense-grafana..."
    docker build -t "${ACR_SERVER}/fleetsense-grafana:${IMAGE_TAG}" -f deploy/monitoring/Dockerfile.grafana .
    docker push "${ACR_SERVER}/fleetsense-grafana:${IMAGE_TAG}"
else
    echo "Local Docker not detected; attempting ACR Cloud Build..."
    echo "-> Building fleetsense-api..."
    az acr build --registry "${ACR_NAME}" --image fleetsense-api:${IMAGE_TAG} --image fleetsense-api:latest --file Dockerfile --no-cache .

    echo "-> Building fleetsense-frontend..."
    az acr build --registry "${ACR_NAME}" --image fleetsense-frontend:${IMAGE_TAG} --image fleetsense-frontend:latest --file Dockerfile.frontend .

    echo "-> Building fleetsense-prometheus..."
    az acr build --registry "${ACR_NAME}" --image fleetsense-prometheus:${IMAGE_TAG} --image fleetsense-prometheus:latest --file deploy/monitoring/Dockerfile.prometheus .

    echo "-> Building fleetsense-grafana..."
    az acr build --registry "${ACR_NAME}" --image fleetsense-grafana:${IMAGE_TAG} --image fleetsense-grafana:latest --file deploy/monitoring/Dockerfile.grafana .
fi

# 4. Deploy all 4 Container Apps via main.bicep (images exist now in ACR)
echo "[4/4] Deploying 4 Container Apps..."
az deployment group create \
    --resource-group "${RESOURCE_GROUP}" \
    --template-file deploy/azure/main.bicep \
    --parameters prefix="${PREFIX}" imageTag="${IMAGE_TAG}" grafanaAdminPassword="${GRAFANA_ADMIN_PASSWORD}" azureContainerName="${AZURE_CONTAINER_NAME}" azureStorageConnectionString="${AZURE_STORAGE_CONNECTION_STRING}" \
    --output none

# Retrieve FQDNs
API_FQDN=$(az containerapp show --name "${PREFIX}-api" --resource-group "${RESOURCE_GROUP}" --query "properties.configuration.ingress.fqdn" --output tsv)
FRONTEND_FQDN=$(az containerapp show --name "${PREFIX}-frontend" --resource-group "${RESOURCE_GROUP}" --query "properties.configuration.ingress.fqdn" --output tsv)
GRAFANA_FQDN=$(az containerapp show --name "${PREFIX}-grafana" --resource-group "${RESOURCE_GROUP}" --query "properties.configuration.ingress.fqdn" --output tsv)

echo "=========================================================="
echo " FleetSense Deployment Complete!"
echo " Frontend Web App:   https://${FRONTEND_FQDN}"
echo " API Docs:           https://${API_FQDN}/docs"
echo " Grafana Dashboard:  https://${GRAFANA_FQDN}"
echo " Default Grafana PW: ${GRAFANA_ADMIN_PASSWORD}"
echo "=========================================================="
