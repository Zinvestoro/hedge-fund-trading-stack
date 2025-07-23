#!/bin/bash

# Trading Stack Monitoring Infrastructure Setup Script
# Sets up Prometheus, Grafana, and custom metrics exporters

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRADING_STACK_DIR="$(dirname "$SCRIPT_DIR")"
MONITORING_DIR="$TRADING_STACK_DIR/monitoring"

log "Setting up trading stack monitoring infrastructure..."
log "Monitoring directory: $MONITORING_DIR"

# Create monitoring directory structure
log "Creating monitoring directory structure..."
mkdir -p "$MONITORING_DIR"/{configs,dashboards,alerts,data}

# Activate virtual environment
cd "$TRADING_STACK_DIR"
if [[ ! -d "venv" ]]; then
    error "Virtual environment not found. Please run setup_data_ingestion.sh first."
    exit 1
fi

source venv/bin/activate

# Install monitoring dependencies
log "Installing monitoring dependencies..."
pip install prometheus-client==0.17.1
pip install grafana-api==1.0.3
pip install aiohttp==3.8.5
pip install psutil==5.9.5

# Create Prometheus configuration
log "Creating Prometheus configuration..."
cat > "$MONITORING_DIR/configs/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "trading_rules.yml"

scrape_configs:
  - job_name: 'trading-stack-metrics'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'trading-stack-health'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 30s
    metrics_path: /health

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

# Create Grafana provisioning configs
log "Creating Grafana configuration..."
mkdir -p "$MONITORING_DIR/configs/grafana"/{provisioning,dashboards}

cat > "$MONITORING_DIR/configs/grafana/provisioning/datasources.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9091
    isDefault: true
    editable: true
EOF

cat > "$MONITORING_DIR/configs/grafana/provisioning/dashboards.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'trading-dashboards'
    orgId: 1
    folder: 'Trading Stack'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Create Docker Compose for monitoring stack
log "Creating Docker Compose configuration for monitoring..."
cat > "$MONITORING_DIR/configs/docker-compose-monitoring.yml" << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.40.0
    container_name: trading-prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./trading_rules.yml:/etc/prometheus/trading_rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - trading-network

  grafana:
    image: grafana/grafana:9.3.0
    container_name: trading-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ../dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=trading123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    restart: unless-stopped
    networks:
      - trading-network

  alertmanager:
    image: prom/alertmanager:v0.25.0
    container_name: trading-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    restart: unless-stopped
    networks:
      - trading-network

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:

networks:
  trading-network:
    external: true
EOF

# Create alerting rules
log "Creating Prometheus alerting rules..."
cat > "$MONITORING_DIR/configs/trading_rules.yml" << 'EOF'
groups:
  - name: trading_alerts
    rules:
      - alert: HighPortfolioVaR
        expr: trading_portfolio_var_usd > 45000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Portfolio VaR is high"
          description: "Portfolio VaR is {{ $value }} USD, above 45k threshold"

      - alert: RiskLimitBreach
        expr: trading_risk_limit_utilization > 90
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Risk limit breach detected"
          description: "Risk limit utilization is {{ $value }}%"

      - alert: ComponentDown
        expr: trading_component_status == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Trading component is down"
          description: "Component {{ $labels.component }} is unhealthy"

      - alert: HighOrderLatency
        expr: histogram_quantile(0.95, trading_order_execution_latency_seconds) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High order execution latency"
          description: "95th percentile order latency is {{ $value }}s"

      - alert: LowStrategyWinRate
        expr: trading_strategy_win_rate < 40
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low strategy win rate"
          description: "Strategy {{ $labels.strategy }} win rate is {{ $value }}%"
EOF

# Create Alertmanager configuration
cat > "$MONITORING_DIR/configs/alertmanager.yml" << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@trading-stack.local'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://localhost:8080/alerts'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF

# Create monitoring startup script
log "Creating monitoring startup script..."
cat > "$MONITORING_DIR/start_monitoring.sh" << 'EOF'
#!/bin/bash

# Start Trading Stack Monitoring Infrastructure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/configs"

cd "$CONFIGS_DIR"

echo "Starting trading stack monitoring infrastructure..."
echo "Prometheus: http://localhost:9091"
echo "Grafana: http://localhost:3000 (admin/trading123)"
echo "Alertmanager: http://localhost:9093"
echo ""

# Start monitoring stack
docker-compose -f docker-compose-monitoring.yml up -d

echo "Monitoring infrastructure started!"
echo ""
echo "To stop: docker-compose -f docker-compose-monitoring.yml down"
EOF

chmod +x "$MONITORING_DIR/start_monitoring.sh"

# Create monitoring stop script
cat > "$MONITORING_DIR/stop_monitoring.sh" << 'EOF'
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/configs"

cd "$CONFIGS_DIR"

echo "Stopping trading stack monitoring infrastructure..."
docker-compose -f docker-compose-monitoring.yml down

echo "Monitoring infrastructure stopped!"
EOF

chmod +x "$MONITORING_DIR/stop_monitoring.sh"

# Create metrics exporter service
log "Creating metrics exporter service..."
cat > "$MONITORING_DIR/start_metrics_exporter.sh" << 'EOF'
#!/bin/bash

# Start Trading Stack Metrics Exporter

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRADING_STACK_DIR="$(dirname "$SCRIPT_DIR")"

cd "$TRADING_STACK_DIR"

# Activate virtual environment
source venv/bin/activate

echo "Starting trading stack metrics exporter..."
echo "Metrics endpoint: http://localhost:9090/metrics"
echo "Health endpoint: http://localhost:9090/health"
echo ""
echo "Press Ctrl+C to stop"

python -m monitoring.prometheus_exporter
EOF

chmod +x "$MONITORING_DIR/start_metrics_exporter.sh"

# Create sample Grafana dashboard
log "Creating sample Grafana dashboard..."
cat > "$MONITORING_DIR/dashboards/trading_overview.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Trading Stack Overview",
    "tags": ["trading", "overview"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Portfolio Value",
        "type": "stat",
        "targets": [
          {
            "expr": "trading_portfolio_value_usd{account=\"main\"}",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD"
          }
        }
      },
      {
        "id": 2,
        "title": "Portfolio P&L",
        "type": "stat",
        "targets": [
          {
            "expr": "trading_portfolio_pnl_usd{type=\"unrealized\"}",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD"
          }
        }
      },
      {
        "id": 3,
        "title": "Open Positions",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(trading_position_count)",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0}
      },
      {
        "id": 4,
        "title": "Portfolio VaR",
        "type": "stat",
        "targets": [
          {
            "expr": "trading_portfolio_var_usd{time_horizon=\"daily\"}",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD"
          }
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s",
    "schemaVersion": 30,
    "version": 1
  },
  "overwrite": true
}
EOF

success "Monitoring infrastructure setup completed!"
echo ""
echo "Next steps:"
echo "1. Start monitoring stack: ./monitoring/start_monitoring.sh"
echo "2. Start metrics exporter: ./monitoring/start_metrics_exporter.sh"
echo "3. Access Grafana at: http://localhost:3000 (admin/trading123)"
echo "4. Access Prometheus at: http://localhost:9091"
echo ""
echo "Monitoring directory: $MONITORING_DIR"

