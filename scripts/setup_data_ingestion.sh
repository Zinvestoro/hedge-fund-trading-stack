#!/bin/bash

# Trading Stack Data Ingestion Setup Script
# Automates the deployment of Kafka, QuestDB, and related infrastructure

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root"
   exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRADING_STACK_DIR="$(dirname "$SCRIPT_DIR")"
CONFIGS_DIR="$TRADING_STACK_DIR/configs"

log "Starting trading stack data ingestion setup..."
log "Trading stack directory: $TRADING_STACK_DIR"

# Create necessary directories
log "Creating directory structure..."
mkdir -p "$TRADING_STACK_DIR"/{data_ingestion,research,strategies,execution,monitoring,logs,data}

# Check if config files exist
if [[ ! -f "$CONFIGS_DIR/docker-compose.yml" ]]; then
    error "Docker Compose configuration not found at $CONFIGS_DIR/docker-compose.yml"
    exit 1
fi

# Stop any existing containers
log "Stopping existing containers..."
cd "$CONFIGS_DIR"
docker-compose down --remove-orphans || true

# Pull latest images
log "Pulling Docker images..."
docker-compose pull

# Start infrastructure services
log "Starting infrastructure services..."
docker-compose up -d

# Wait for services to be ready
log "Waiting for services to start..."

# Wait for Kafka
log "Waiting for Kafka to be ready..."
timeout=60
counter=0
while ! docker exec trading-kafka kafka-topics.sh --bootstrap-server localhost:9092 --list &>/dev/null; do
    if [ $counter -eq $timeout ]; then
        error "Kafka failed to start within $timeout seconds"
        exit 1
    fi
    sleep 2
    ((counter++))
done
success "Kafka is ready"

# Wait for QuestDB
log "Waiting for QuestDB to be ready..."
timeout=60
counter=0
while ! curl -s http://localhost:9000/status &>/dev/null; do
    if [ $counter -eq $timeout ]; then
        error "QuestDB failed to start within $timeout seconds"
        exit 1
    fi
    sleep 2
    ((counter++))
done
success "QuestDB is ready"

# Wait for PostgreSQL
log "Waiting for PostgreSQL to be ready..."
timeout=60
counter=0
while ! docker exec trading-postgres pg_isready -U trading_user -d trading &>/dev/null; do
    if [ $counter -eq $timeout ]; then
        error "PostgreSQL failed to start within $timeout seconds"
        exit 1
    fi
    sleep 2
    ((counter++))
done
success "PostgreSQL is ready"

# Create Kafka topics
log "Creating Kafka topics..."

# Function to create topic if it doesn't exist
create_topic() {
    local topic_name=$1
    local partitions=$2
    local retention_ms=$3
    
    if ! docker exec trading-kafka kafka-topics.sh --bootstrap-server localhost:9092 --list | grep -q "^${topic_name}$"; then
        log "Creating topic: $topic_name"
        docker exec trading-kafka kafka-topics.sh \
            --create \
            --topic "$topic_name" \
            --bootstrap-server localhost:9092 \
            --partitions "$partitions" \
            --replication-factor 1 \
            --config compression.type=zstd \
            --config retention.ms="$retention_ms" \
            --config segment.ms=3600000
        success "Created topic: $topic_name"
    else
        log "Topic $topic_name already exists"
    fi
}

# Create topics for different data types
create_topic "equity-ticks" 16 604800000      # 7 days retention
create_topic "equity-quotes" 16 604800000     # 7 days retention
create_topic "crypto-ticks" 8 604800000       # 7 days retention
create_topic "crypto-quotes" 8 604800000      # 7 days retention
create_topic "orderbook-snapshots" 32 86400000  # 1 day retention
create_topic "trading-signals" 4 2592000000   # 30 days retention
create_topic "strategy-orders" 4 2592000000   # 30 days retention
create_topic "risk-events" 2 7776000000       # 90 days retention

# Create QuestDB tables
log "Creating QuestDB tables..."

# Function to execute SQL in QuestDB
execute_questdb_sql() {
    local sql="$1"
    curl -s -G \
        --data-urlencode "query=$sql" \
        http://localhost:9000/exec || true
}

# Create tables for market data
log "Creating equity_ticks table..."
execute_questdb_sql "
CREATE TABLE IF NOT EXISTS equity_ticks (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 10000 CACHE,
    price DOUBLE,
    size LONG,
    exchange SYMBOL CAPACITY 100 CACHE,
    condition_codes STRING,
    sequence_number LONG
) TIMESTAMP(timestamp) PARTITION BY DAY;"

log "Creating equity_quotes table..."
execute_questdb_sql "
CREATE TABLE IF NOT EXISTS equity_quotes (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 10000 CACHE,
    bid_price DOUBLE,
    bid_size LONG,
    ask_price DOUBLE,
    ask_size LONG,
    exchange SYMBOL CAPACITY 100 CACHE,
    sequence_number LONG
) TIMESTAMP(timestamp) PARTITION BY DAY;"

log "Creating crypto_ticks table..."
execute_questdb_sql "
CREATE TABLE IF NOT EXISTS crypto_ticks (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 1000 CACHE,
    price DOUBLE,
    size DOUBLE,
    side SYMBOL CAPACITY 10 CACHE,
    trade_id STRING,
    exchange SYMBOL CAPACITY 50 CACHE
) TIMESTAMP(timestamp) PARTITION BY DAY;"

log "Creating orderbook_snapshots table..."
execute_questdb_sql "
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 10000 CACHE,
    bid_prices STRING,
    bid_sizes STRING,
    ask_prices STRING,
    ask_sizes STRING,
    exchange SYMBOL CAPACITY 100 CACHE,
    sequence_number LONG
) TIMESTAMP(timestamp) PARTITION BY HOUR;"

log "Creating trading_signals table..."
execute_questdb_sql "
CREATE TABLE IF NOT EXISTS trading_signals (
    timestamp TIMESTAMP,
    strategy_id SYMBOL CAPACITY 1000 CACHE,
    symbol SYMBOL CAPACITY 10000 CACHE,
    signal_type SYMBOL CAPACITY 100 CACHE,
    signal_strength DOUBLE,
    confidence DOUBLE,
    metadata STRING
) TIMESTAMP(timestamp) PARTITION BY DAY;"

success "QuestDB tables created successfully"

# Install Python dependencies
log "Installing Python dependencies..."
cd "$TRADING_STACK_DIR"

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    log "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
source venv/bin/activate

# Create requirements.txt if it doesn't exist
if [[ ! -f "requirements.txt" ]]; then
    log "Creating requirements.txt..."
    cat > requirements.txt << EOF
# Data ingestion and streaming
kafka-python==2.0.2
questdb==1.1.0
websockets==11.0.3
aiohttp==3.8.5

# Data processing and analysis
pandas==2.1.0
numpy==1.24.3
scipy==1.11.1

# Machine learning and AI
torch==2.0.1
torchvision==0.15.2
torchaudio==0.15.2
tensorflow==2.13.0
scikit-learn==1.3.0

# Trading and backtesting frameworks
vectorbt==0.25.2
zipline-reloaded==2.2.0
backtrader==1.9.78.123

# Risk management and portfolio optimization
cvxpy==1.3.2
pyportfolioopt==1.5.5

# Database and caching
psycopg2-binary==2.9.7
redis==4.6.0
sqlalchemy==2.0.19

# API clients and data providers
yfinance==0.2.18
alpha-vantage==2.3.1
polygon-api-client==1.12.0

# Monitoring and logging
prometheus-client==0.17.1
grafana-api==1.0.3

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
click==8.1.6
rich==13.5.2
asyncio-mqtt==0.13.0
EOF
fi

log "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create environment configuration
log "Creating environment configuration..."
if [[ ! -f ".env" ]]; then
    cat > .env << EOF
# Trading Stack Environment Configuration

# Data Sources
POLYGON_API_KEY=your_polygon_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trading
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=trading_password

# QuestDB Configuration
QUESTDB_HOST=localhost
QUESTDB_HTTP_PORT=9000
QUESTDB_ILP_PORT=9009
QUESTDB_PG_PORT=8812

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_COMPRESSION_TYPE=zstd

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Risk Management
MAX_POSITION_SIZE=10000
MAX_DAILY_LOSS=1000
RISK_CHECK_INTERVAL=60

# Logging
LOG_LEVEL=INFO
LOG_FILE=/tmp/trading_stack.log
EOF
    warning "Created .env file with default values. Please update with your actual API keys and configuration."
fi

# Create systemd service files for production deployment
log "Creating systemd service files..."
mkdir -p "$TRADING_STACK_DIR/systemd"

cat > "$TRADING_STACK_DIR/systemd/trading-data-ingestion.service" << EOF
[Unit]
Description=Trading Stack Data Ingestion Service
After=docker.service
Requires=docker.service

[Service]
Type=forking
RemainAfterExit=yes
WorkingDirectory=$CONFIGS_DIR
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Create monitoring script
log "Creating monitoring script..."
cat > "$TRADING_STACK_DIR/scripts/monitor_services.sh" << 'EOF'
#!/bin/bash

# Service monitoring script
check_service() {
    local service_name=$1
    local check_command=$2
    
    if eval "$check_command" &>/dev/null; then
        echo "✅ $service_name is running"
        return 0
    else
        echo "❌ $service_name is not responding"
        return 1
    fi
}

echo "Trading Stack Service Status Check"
echo "=================================="

check_service "Kafka" "docker exec trading-kafka kafka-topics.sh --bootstrap-server localhost:9092 --list"
check_service "QuestDB" "curl -s http://localhost:9000/status"
check_service "PostgreSQL" "docker exec trading-postgres pg_isready -U trading_user -d trading"
check_service "Redis" "docker exec trading-redis redis-cli ping"
check_service "Kafka UI" "curl -s http://localhost:8080"

echo ""
echo "Container Status:"
docker-compose -f "$(dirname "$0")/../configs/docker-compose.yml" ps
EOF

chmod +x "$TRADING_STACK_DIR/scripts/monitor_services.sh"

# Create data ingestion test script
log "Creating test script..."
cat > "$TRADING_STACK_DIR/scripts/test_data_ingestion.py" << 'EOF'
#!/usr/bin/env python3
"""
Test script for data ingestion infrastructure
"""

import asyncio
import json
import time
from kafka import KafkaProducer, KafkaConsumer
import questdb.ingress as qdb_ingress
import psycopg2
import redis

async def test_kafka():
    """Test Kafka connectivity and basic operations"""
    print("Testing Kafka...")
    
    try:
        # Test producer
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        test_message = {
            'timestamp': int(time.time() * 1000000),
            'symbol': 'TEST',
            'price': 100.0,
            'size': 100
        }
        
        producer.send('equity-ticks', test_message)
        producer.flush()
        print("✅ Kafka producer test passed")
        
        # Test consumer
        consumer = KafkaConsumer(
            'equity-ticks',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='latest',
            consumer_timeout_ms=5000,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        # Send another message to consume
        producer.send('equity-ticks', test_message)
        producer.flush()
        
        for message in consumer:
            print("✅ Kafka consumer test passed")
            break
        
        producer.close()
        consumer.close()
        
    except Exception as e:
        print(f"❌ Kafka test failed: {e}")

def test_questdb():
    """Test QuestDB connectivity"""
    print("Testing QuestDB...")
    
    try:
        # Test ingress client
        sender = qdb_ingress.Sender('localhost', 9009)
        
        sender.row(
            'test_table',
            symbols={'symbol': 'TEST'},
            columns={'price': 100.0, 'size': 100},
            at=int(time.time() * 1000000)
        )
        
        sender.flush()
        sender.close()
        print("✅ QuestDB ingress test passed")
        
    except Exception as e:
        print(f"❌ QuestDB test failed: {e}")

def test_postgresql():
    """Test PostgreSQL connectivity"""
    print("Testing PostgreSQL...")
    
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='trading',
            user='trading_user',
            password='trading_password'
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trading.instruments")
        count = cursor.fetchone()[0]
        print(f"✅ PostgreSQL test passed - {count} instruments in database")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ PostgreSQL test failed: {e}")

def test_redis():
    """Test Redis connectivity"""
    print("Testing Redis...")
    
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.set('test_key', 'test_value')
        value = r.get('test_key')
        
        if value == 'test_value':
            print("✅ Redis test passed")
        else:
            print("❌ Redis test failed - unexpected value")
            
        r.delete('test_key')
        
    except Exception as e:
        print(f"❌ Redis test failed: {e}")

async def main():
    """Run all tests"""
    print("Trading Stack Infrastructure Test")
    print("=================================")
    
    await test_kafka()
    test_questdb()
    test_postgresql()
    test_redis()
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x "$TRADING_STACK_DIR/scripts/test_data_ingestion.py"

# Final status check
log "Running final status check..."
sleep 5
"$TRADING_STACK_DIR/scripts/monitor_services.sh"

success "Data ingestion infrastructure setup completed!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your API keys"
echo "2. Run test script: python scripts/test_data_ingestion.py"
echo "3. Start data ingestion: python data_ingestion/polygon_client.py"
echo "4. Monitor services: ./scripts/monitor_services.sh"
echo ""
echo "Web interfaces:"
echo "- QuestDB Console: http://localhost:9000"
echo "- Kafka UI: http://localhost:8080"
echo ""
echo "For production deployment, copy systemd service files:"
echo "sudo cp systemd/*.service /etc/systemd/system/"
echo "sudo systemctl enable trading-data-ingestion"
echo "sudo systemctl start trading-data-ingestion"
EOF

chmod +x /home/ubuntu/trading-stack/scripts/setup_data_ingestion.sh

