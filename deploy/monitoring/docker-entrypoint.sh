#!/bin/sh
set -e

# If API_SCRAPE_TARGET is provided (e.g. in Azure Container Apps), configure Prometheus for HTTPS scraping
if [ -n "$API_SCRAPE_TARGET" ]; then
    cat <<EOF > /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/alert_rules.yml"

scrape_configs:
  - job_name: "fleetsense-api"
    metrics_path: "/metrics"
    scheme: "https"
    tls_config:
      insecure_skip_verify: true
    static_configs:
      - targets: ["$API_SCRAPE_TARGET"]
        labels:
          service: "fleetsense-inference-service"
          environment: "production"
EOF
fi

exec /bin/prometheus "$@"
