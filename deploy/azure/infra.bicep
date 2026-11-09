// ==============================================================================
// FleetSense - Core Infrastructure (Storage, ACR, Log Analytics, ACA Environment)
// ==============================================================================

@description('Deployment location for all resources')
param location string = resourceGroup().location

@description('Prefix for resource naming')
param prefix string = 'fleetsense'

var uniqueSuffix = uniqueString(resourceGroup().id)
var storageAccountName = '${prefix}data${take(uniqueSuffix, 8)}'
var acrName = '${prefix}acr${take(uniqueSuffix, 8)}'
var logAnalyticsName = '${prefix}-logs-${uniqueSuffix}'
var envName = '${prefix}-env-${uniqueSuffix}'

// 1. Storage Account for TLC Datasets, Ingestion Metadata & Models
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource rawContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'tlc-raw'
}

resource modelsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'fleetsense-data'
}

// 2. Azure Container Registry
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// 3. Log Analytics Workspace
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// 4. Container Apps Managed Environment
resource containerEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// Outputs
output acrLoginServer string = acr.properties.loginServer
output acrName string = acr.name
output storageAccountName string = storageAccount.name
output containerEnvId string = containerEnv.id
