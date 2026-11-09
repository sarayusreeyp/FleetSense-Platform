# ==============================================================================
# FleetSense - Azure Container Apps Automated Deployment Script (PowerShell)
# Builds and deploys:
#   1. fleetsense-api (ML Inference Service)
#   2. fleetsense-frontend (Web UI)
#   3. fleetsense-prometheus (Internal Metrics Scraper)
#   4. fleetsense-grafana (Metrics Dashboard)
# ==============================================================================
param (
    [string]$ResourceGroup = "fleetsense-rg",
    [string]$Location = "centralindia",
    [string]$Prefix = "fleetsense",
    [string]$ImageTag = $(Get-Date -Format "yyyyMMddHHmmss"),
    [string]$GrafanaAdminPassword = "FleetSense2026!",
    [string]$AzureContainerName = "fleetsense-production",
    [string]$AzureStorageConnectionString = ""
)

$ErrorActionPreference = "Stop"

# Auto-load connection string from environment or .env if not explicitly passed
if (-not $AzureStorageConnectionString) {
    if ($env:AZURE_STORAGE_CONNECTION_STRING) {
        $AzureStorageConnectionString = $env:AZURE_STORAGE_CONNECTION_STRING
    } elseif (Test-Path ".env") {
        Get-Content ".env" | ForEach-Object {
            if ($_ -match '^\s*AZURE_STORAGE_CONNECTION_STRING\s*=\s*(.+)$') {
                $AzureStorageConnectionString = $matches[1].Trim()
            }
        }
    }
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting FleetSense 4-Tier Azure Deployment" -ForegroundColor Cyan
Write-Host " Resource Group: $ResourceGroup ($Location)" -ForegroundColor Cyan
Write-Host " Storage Container: $AzureContainerName" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Create Resource Group
Write-Host "`n[1/4] Ensuring Azure Resource Group exists..." -ForegroundColor Yellow
az group create --name $ResourceGroup --location $Location --output table

# 2. Deploy Core Infrastructure (Storage, ACR, ACA Environment) via Bicep
Write-Host "`n[2/4] Deploying core infrastructure (Storage, ACR, Environment)..." -ForegroundColor Yellow
az deployment group create `
    --resource-group $ResourceGroup `
    --template-file deploy/azure/infra.bicep `
    --parameters prefix=$Prefix `
    --output none

$acrServer = (az deployment group show `
    --resource-group $ResourceGroup `
    --name "infra" `
    --query "properties.outputs.acrLoginServer.value" `
    --output tsv).Trim()

$acrName = (az deployment group show `
    --resource-group $ResourceGroup `
    --name "infra" `
    --query "properties.outputs.acrName.value" `
    --output tsv).Trim()

Write-Host "ACR Login Server: $acrServer" -ForegroundColor Green

# 3. Build & push container images
Write-Host "`n[3/4] Building and pushing Docker images to $acrName..." -ForegroundColor Yellow

$dockerRunning = $false
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -eq 0) { $dockerRunning = $true }
} catch {
    $dockerRunning = $false
}

if ($dockerRunning) {
    Write-Host "Using local Docker to build and push images..." -ForegroundColor Green
    az acr login --name $acrName

    Write-Host "-> Building & pushing fleetsense-api..." -ForegroundColor Gray
    docker build -t "$acrServer/fleetsense-api:$ImageTag" -f Dockerfile .
    docker push "$acrServer/fleetsense-api:$ImageTag"

    Write-Host "-> Building & pushing fleetsense-frontend..." -ForegroundColor Gray
    docker build -t "$acrServer/fleetsense-frontend:$ImageTag" -f Dockerfile.frontend .
    docker push "$acrServer/fleetsense-frontend:$ImageTag"

    Write-Host "-> Building & pushing fleetsense-prometheus..." -ForegroundColor Gray
    docker build -t "$acrServer/fleetsense-prometheus:$ImageTag" -f deploy/monitoring/Dockerfile.prometheus .
    docker push "$acrServer/fleetsense-prometheus:$ImageTag"

    Write-Host "-> Building & pushing fleetsense-grafana..." -ForegroundColor Gray
    docker build -t "$acrServer/fleetsense-grafana:$ImageTag" -f deploy/monitoring/Dockerfile.grafana .
    docker push "$acrServer/fleetsense-grafana:$ImageTag"
} else {
    Write-Host "Local Docker daemon not detected. Using Azure ACR Cloud Build (no local Docker required)..." -ForegroundColor Magenta

    Write-Host "-> Building fleetsense-api via ACR..." -ForegroundColor Gray
    az acr build --registry $acrName --image "fleetsense-api:$ImageTag" --image "fleetsense-api:latest" --file Dockerfile --no-cache .

    Write-Host "-> Building fleetsense-frontend via ACR..." -ForegroundColor Gray
    az acr build --registry $acrName --image "fleetsense-frontend:$ImageTag" --image "fleetsense-frontend:latest" --file Dockerfile.frontend .

    Write-Host "-> Building fleetsense-prometheus via ACR..." -ForegroundColor Gray
    az acr build --registry $acrName --image "fleetsense-prometheus:$ImageTag" --image "fleetsense-prometheus:latest" --file deploy/monitoring/Dockerfile.prometheus .

    Write-Host "-> Building fleetsense-grafana via ACR..." -ForegroundColor Gray
    az acr build --registry $acrName --image "fleetsense-grafana:$ImageTag" --image "fleetsense-grafana:latest" --file deploy/monitoring/Dockerfile.grafana .
}

# 4. Deploy all 4 Container Apps via main.bicep
Write-Host "`n[4/4] Deploying 4 Container Apps..." -ForegroundColor Yellow
$deployParams = @(
    "prefix=$Prefix",
    "imageTag=$ImageTag",
    "grafanaAdminPassword=$GrafanaAdminPassword",
    "azureContainerName=$AzureContainerName"
)
if ($AzureStorageConnectionString) {
    $deployParams += "azureStorageConnectionString=$AzureStorageConnectionString"
}

az deployment group create `
    --resource-group $ResourceGroup `
    --template-file deploy/azure/main.bicep `
    --parameters @deployParams `
    --output none

# 5. Retrieve FQDNs
Write-Host "`nRetrieving deployment endpoints..." -ForegroundColor Yellow
$apiFqdn = (az containerapp show --name "$Prefix-api" --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" --output tsv).Trim()
$frontendFqdn = (az containerapp show --name "$Prefix-frontend" --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" --output tsv).Trim()
$grafanaFqdn = (az containerapp show --name "$Prefix-grafana" --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" --output tsv).Trim()

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " FleetSense Production Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Web Dashboard:   https://$frontendFqdn" -ForegroundColor Cyan
Write-Host "  FastAPI Docs:    https://$apiFqdn/docs" -ForegroundColor Cyan
Write-Host "  Health Status:   https://$apiFqdn/health" -ForegroundColor Cyan
Write-Host "  Grafana Portal:  https://$grafanaFqdn (user: admin, pass: $GrafanaAdminPassword)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
