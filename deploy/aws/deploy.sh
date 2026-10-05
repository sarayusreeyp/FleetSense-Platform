#!/usr/bin/env bash
# ==============================================================================
# FleetSense - AWS App Runner & ECR Deployment Script
# ==============================================================================
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO_NAME="${AWS_ECR_REPO:-fleetsense-api}"
SERVICE_NAME="${AWS_SERVICE_NAME:-fleetsense-service}"
S3_BUCKET="${AWS_S3_BUCKET:-fleetsense-data-${AWS_ACCOUNT_ID}}"

echo "=========================================================="
echo " Starting FleetSense Deployment to AWS App Runner"
echo " AWS Region: ${AWS_REGION} | Account: ${AWS_ACCOUNT_ID}"
echo "=========================================================="

# 1. Create S3 Bucket for DVC remote / Raw TLC Parquet storage
echo "[1/4] Ensuring S3 data bucket exists..."
if ! aws s3api head-bucket --bucket "${S3_BUCKET}" 2>/dev/null; then
    aws s3api create-bucket --bucket "${S3_BUCKET}" --region "${AWS_REGION}"
    echo "Created S3 bucket: s3://${S3_BUCKET}"
fi

# 2. Create ECR Repository if not exists
echo "[2/4] Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" 2>/dev/null || \
    aws ecr create-repository --repository-name "${ECR_REPO_NAME}" --region "${AWS_REGION}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

# 3. Log in to ECR, build and push image
echo "[3/4] Authenticating with ECR and pushing Docker image..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_URI}"

docker build -t "${ECR_URI}:latest" .
docker push "${ECR_URI}:latest"

# 4. Create or Update AWS App Runner Service
echo "[4/4] Deploying AWS App Runner Service..."
aws apprunner create-service \
    --service-name "${SERVICE_NAME}" \
    --source-configuration "{
        \"ImageRepository\": {
            \"ImageIdentifier\": \"${ECR_URI}:latest\",
            \"ImageConfiguration\": {
                \"Port\": \"8000\",
                \"RuntimeEnvironmentVariables\": {
                    \"FLEETSENSE_APP_ENV\": \"production\",
                    \"FLEETSENSE_STORAGE_BACKEND\": \"local\"
                }
            },
            \"ImageRepositoryType\": \"ECR\"
        },
        \"AutoDeploymentsEnabled\": true
    }" \
    --instance-configuration "{
        \"Cpu\": \"1024\",
        \"Memory\": \"2048\"
    }" \
    --region "${AWS_REGION}" \
    --output table

echo "=========================================================="
echo " FleetSense AWS App Runner Deployment Initiated!"
echo " Monitor status with: aws apprunner list-services"
echo "=========================================================="
