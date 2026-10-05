// ==============================================================================
// FleetSense - Azure Infrastructure as Code (Bicep)
// ==============================================================================

@description('Deployment location for all resources')
param location string = resourceGroup().location

@description('Prefix for resource naming')
param prefix string = 'fleetsense'

@description('Docker image tag to deploy')
param imageTag string = 'latest'

var uniqueSuffix = uniqueString(resourceGroup().id)
var storageAccountName = '${prefix}data${take(uniqueSuffix, 8)}'
var acrName = '${prefix}acr${take(uniqueSuffix, 8)}'
var logAnalyticsName = '${prefix}-logs-${uniqueSuffix}'
var envName = '${prefix}-env-${uniqueSuffix}'
var appName = '${prefix}-api'

// 1. Storage Account for TLC Datasets & DVC Remote
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

// 5. Container App (FleetSense ML API)
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
      }
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
      ]
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.name
          passwordSecretRef: 'acr-password'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'fleetsense-api'
          image: '${acr.properties.loginServer}/fleetsense-api:${imageTag}'
          resources: {
            cpu: json('1.0')
            memory: '2.0Gi'
          }
          env: [
            {
              name: 'FLEETSENSE_APP_ENV'
              value: 'production'
            }
            {
              name: 'FLEETSENSE_STORAGE_BACKEND'
              value: 'azure'
            }
            {
              name: 'AZURE_STORAGE_CONTAINER_NAME'
              value: 'tlc-raw'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}

output apiFqdn string = containerApp.properties.configuration.ingress.fqdn
output storageAccountName string = storageAccount.name
output acrLoginServer string = acr.properties.loginServer
