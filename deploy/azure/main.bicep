// ==============================================================================
// FleetSense - Azure Infrastructure as Code (Bicep)
// Provisions: Storage, ACR, Log Analytics, ACA Managed Environment,
//             and 4 Container Apps: API, Frontend, Prometheus, Grafana
// ==============================================================================

@description('Deployment location for all resources')
param location string = resourceGroup().location

@description('Prefix for resource naming')
param prefix string = 'fleetsense'

@description('Docker image tag to deploy')
param imageTag string = 'latest'

@description('Grafana admin password')
@secure()
param grafanaAdminPassword string

@description('Existing Azure Storage Connection String. If empty, uses the provisioned storage account.')
@secure()
param azureStorageConnectionString string = ''

@description('Azure Blob Storage container name for models and datasets')
param azureContainerName string = 'fleetsense-production'

var uniqueSuffix = uniqueString(resourceGroup().id)
var storageAccountName = '${prefix}data${take(uniqueSuffix, 8)}'
var acrName = '${prefix}acr${take(uniqueSuffix, 8)}'
var logAnalyticsName = '${prefix}-logs-${uniqueSuffix}'
var envName = '${prefix}-env-${uniqueSuffix}'

// Container App Names
var apiAppName = '${prefix}-api'
var frontendAppName = '${prefix}-frontend'
var prometheusAppName = '${prefix}-prometheus'
var grafanaAppName = '${prefix}-grafana'

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

// Construct Azure Storage Connection String securely
var storageKey = storageAccount.listKeys().keys[0].value
var defaultStorageConnString = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageKey};EndpointSuffix=${environment().suffixes.storage}'
var effectiveStorageConnString = empty(azureStorageConnectionString) ? defaultStorageConnString : azureStorageConnectionString

// Shared ACR Configuration for Container Apps
var acrCredentials = acr.listCredentials()
var acrRegistryConfig = [
  {
    server: acr.properties.loginServer
    username: acrCredentials.username
    passwordSecretRef: 'acr-password'
  }
]

var acrSecret = {
  name: 'acr-password'
  value: acrCredentials.passwords[0].value
}

// 5. Container App: FleetSense ML API (Port 8000, External Ingress)
resource apiApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: apiAppName
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
        acrSecret
        {
          name: 'storage-connection-string'
          value: effectiveStorageConnString
        }
      ]
      registries: acrRegistryConfig
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
              value: azureContainerName
            }
            {
              name: 'AZURE_STORAGE_CONNECTION_STRING'
              secretRef: 'storage-connection-string'
            }
            {
              name: 'CORS_ALLOWED_ORIGINS'
              value: '*'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

// 6. Container App: Frontend (Port 80, External Ingress)
resource frontendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: frontendAppName
  location: location
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 80
        transport: 'auto'
      }
      secrets: [acrSecret]
      registries: acrRegistryConfig
    }
    template: {
      containers: [
        {
          name: 'fleetsense-frontend'
          image: '${acr.properties.loginServer}/fleetsense-frontend:${imageTag}'
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 2
      }
    }
  }
}

// 7. Container App: Prometheus (Port 9090, Internal Ingress for Security)
resource prometheusApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: prometheusAppName
  location: location
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: false // Accessible only internally by Grafana inside the ACA environment
        targetPort: 9090
        transport: 'auto'
      }
      secrets: [acrSecret]
      registries: acrRegistryConfig
    }
    template: {
      containers: [
        {
          name: 'fleetsense-prometheus'
          image: '${acr.properties.loginServer}/fleetsense-prometheus:${imageTag}'
          command: [
            '/bin/sh'
            '-c'
            'printf "global:\\n  scrape_interval: 15s\\n  evaluation_interval: 15s\\nscrape_configs:\\n  - job_name: fleetsense-api\\n    metrics_path: /metrics\\n    scheme: https\\n    tls_config:\\n      insecure_skip_verify: true\\n    static_configs:\\n      - targets: [\'${apiAppName}.${containerEnv.properties.defaultDomain}\']\\n" > /tmp/prometheus.yml && /bin/prometheus --config.file=/tmp/prometheus.yml --storage.tsdb.path=/prometheus --web.enable-lifecycle'
          ]
          resources: {
            cpu: json('0.5')
            memory: '1.0Gi'
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

// 8. Container App: Grafana (Port 3000, External Ingress)
resource grafanaApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: grafanaAppName
  location: location
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 3000
        transport: 'auto'
      }
      secrets: [
        acrSecret
        {
          name: 'grafana-admin-pwd'
          value: grafanaAdminPassword
        }
      ]
      registries: acrRegistryConfig
    }
    template: {
      containers: [
        {
          name: 'fleetsense-grafana'
          image: '${acr.properties.loginServer}/fleetsense-grafana:${imageTag}'
          resources: {
            cpu: json('0.5')
            memory: '1.0Gi'
          }
          env: [
            {
              name: 'PROMETHEUS_URL'
              value: 'https://${prometheusAppName}.${containerEnv.properties.defaultDomain}'
            }
            {
              name: 'GF_SECURITY_ADMIN_PASSWORD'
              secretRef: 'grafana-admin-pwd'
            }
            {
              name: 'GF_USERS_ALLOW_SIGN_UP'
              value: 'false'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

// Outputs
output apiFqdn string = apiApp.properties.configuration.ingress.fqdn
output frontendFqdn string = frontendApp.properties.configuration.ingress.fqdn
output grafanaFqdn string = grafanaApp.properties.configuration.ingress.fqdn
output acrLoginServer string = acr.properties.loginServer
output storageAccountName string = storageAccount.name
