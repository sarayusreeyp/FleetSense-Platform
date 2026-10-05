#!/usr/bin/env bash
# ==============================================================================
# FleetSense - Azure Container Apps Automated Deployment Script
# ==============================================================================
set -euo pipefail

# Configuration defaults
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-fleetsense-rg}"
LOCATION="${AZURE_LOCATION:-eastus}"
STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-fleetsensedata$RANDOM}"
CONTAINER_NAME="${AZURE_CONTAINER_NAME:-tlc-raw}"
ACR_NAME="${AZURE_ACR_NAME:-fleetsenseacr$RANDOM}"
ENVIRONMENT_NAME="${AZURE_ENV_NAME:-fleetsense-env}"
APP_NAME="${AZURE_APP_NAME:-fleetsense-api}"

echo "=========================================================="
echo " Starting FleetSense Deployment to Azure Container Apps"
echo " Resource Group: ${RESOURCE_GROUP} (${LOCATION})"
echo "=========================================================="

# 1. Create Resource Group
echo "[1/6] Creating Azure Resource Group..."
az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" --output table

# 2. Create Storage Account for TLC Parquet Data & DVC
echo "[2/6] Provisioning Azure Storage Account for dataset versioning..."
az storage account create \
    --name "${STORAGE_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --sku Standard_LRS \
    --output table

STORAGE_KEY=$(az storage account keys list \
    --account-name "${STORAGE_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query "[0].value" --output tsv)

STORAGE_CONN_STR="DefaultEndpointsProtocol=https;AccountName=${STORAGE_ACCOUNT};AccountKey=${STORAGE_KEY};EndpointSuffix=core.windows.net"

az storage container create \
    --account-name "${STORAGE_ACCOUNT}" \
    --account-key "${STORAGE_KEY}" \
    --name "${CONTAINER_NAME}" \
    --output table

# 3. Create Azure Container Registry (ACR)
echo "[3/6] Provisioning Azure Container Registry..."
az acr create \
    --name "${ACR_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --sku Basic \
    --admin-enabled true \
    --output table

# 4. Build and push container image using ACR Cloud Build
echo "[4/6] Building & pushing Docker image via ACR Build..."
az acr build \
    --registry "${ACR_NAME}" \
    --image fleetsense-api:latest \
    .

# 5. Create Azure Container Apps Managed Environment
echo "[5/6] Creating Container Apps Environment..."
az containerapp env create \
    --name "${ENVIRONMENT_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --output table

# 6. Deploy FleetSense API Container App
echo "[6/6] Deploying FleetSense Container App..."
ACR_SERVER="${ACR_NAME}.azurecr.io"
ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --query "passwords[0].value" --output tsv)

az containerapp create \
    --name "${APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --environment "${ENVIRONMENT_NAME}" \
    --image "${ACR_SERVER}/fleetsense-api:latest" \
    --target-port 8000 \
    --ingress external \
    --registry-server "${ACR_SERVER}" \
    --registry-username "${ACR_NAME}" \
    --registry-password "${ACR_PASSWORD}" \
    --min-replicas 1 \
    --max-replicas 5 \
    --cpu 1.0 \
    --memory 2.0Gi \
    --env-vars \
        FLEETSENSE_APP_ENV="production" \
        FLEETSENSE_STORAGE_BACKEND="azure" \
        AZURE_STORAGE_CONNECTION_STRING="${STORAGE_CONN_STR}" \
        AZURE_STORAGE_CONTAINER_NAME="${CONTAINER_NAME}" \
    --output table

FQDN=$(az containerapp show --name "${APP_NAME}" --resource-group "${RESOURCE_GROUP}" --query "properties.configuration.ingress.fqdn" --output tsv)

echo "=========================================================="
echo " FleetSense Deployment Complete!"
echo " Public Endpoint: https://${FQDN}"
echo " API Docs:        https://${FQDN}/docs"
echo " Health Status:   https://${FQDN}/health"
echo " Metrics:         https://${FQDN}/metrics"
echo "=========================================================="
