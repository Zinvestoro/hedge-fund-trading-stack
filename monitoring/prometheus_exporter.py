#!/usr/bin/env python3
"""
Prometheus Metrics Exporter for Trading Stack
Comprehensive monitoring and observability for trading operations
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import redis
import psycopg2
from aiohttp import web
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    """Configuration for custom metrics"""
    name: str
    metric_type: str
    description: str
    labels: List[str] = None
    buckets: List[float] = None

class TradingStackMetricsExporter:
    """Comprehensive metrics exporter for trading stack monitoring"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.registry = CollectorRegistry()
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_stack',
            'user': 'trading_user',
            'password': 'trading_pass'
        }
        
        # Initialize all metrics
        self._setup_core_metrics()
        self._setup_trading_metrics()
        self._setup_risk_metrics()
        self._setup_performance_metrics()
        
        logger.info("Initialized TradingStackMetricsExporter")
    
    def _setup_core_metrics(self):
        """Setup core infrastructure metrics"""
        
        # Data ingestion metrics
        self.data_ingestion_rate = Counter(
            'trading_data_ingestion_total',
            'Total market data points ingested',
            ['source', 'symbol', 'data_type'],
            registry=self.registry
        )
        
        self.data_ingestion_latency = Histogram(
            'trading_data_ingestion_latency_seconds',
            'Market data ingestion latency',
            ['source', 'symbol'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        # Order execution metrics
        self.orders_submitted = Counter(
            'trading_orders_submitted_total',
            'Total orders submitted',
            ['strategy', 'symbol', 'side', 'order_type'],
            registry=self.registry
        )
        
        self.orders_filled = Counter(
            'trading_orders_filled_total',
            'Total orders filled',
            ['strategy', 'symbol', 'side'],
            registry=self.registry
        )
        
        self.order_execution_latency = Histogram(
            'trading_order_execution_latency_seconds',
            'Order execution latency',
            ['strategy', 'symbol'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        # System health metrics
        self.component_status = Gauge(
            'trading_component_status',
            'Component health status (1=healthy, 0=unhealthy)',
            ['component', 'instance'],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'trading_cpu_usage_percent',
            'CPU usage percentage',
            ['component'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'trading_memory_usage_bytes',
            'Memory usage in bytes',
            ['component', 'type'],
            registry=self.registry
        )
    
    def _setup_trading_metrics(self):
        """Setup trading-specific metrics"""
        
        # Portfolio metrics
        self.portfolio_value = Gauge(
            'trading_portfolio_value_usd',
            'Total portfolio value in USD',
            ['account', 'strategy'],
            registry=self.registry
        )
        
        self.portfolio_pnl = Gauge(
            'trading_portfolio_pnl_usd',
            'Portfolio P&L in USD',
            ['account', 'strategy', 'type'],
            registry=self.registry
        )
        
        self.position_count = Gauge(
            'trading_position_count',
            'Number of open positions',
            ['account', 'strategy', 'side'],
            registry=self.registry
        )
        
        # Strategy performance
        self.strategy_returns = Gauge(
            'trading_strategy_returns',
            'Strategy returns',
            ['strategy', 'period'],
            registry=self.registry
        )
        
        self.strategy_sharpe_ratio = Gauge(
            'trading_strategy_sharpe_ratio',
            'Strategy Sharpe ratio',
            ['strategy'],
            registry=self.registry
        )
        
        self.strategy_win_rate = Gauge(
            'trading_strategy_win_rate',
            'Strategy win rate percentage',
            ['strategy'],
            registry=self.registry
        )
    
    def _setup_risk_metrics(self):
        """Setup risk management metrics"""
        
        # VaR metrics
        self.portfolio_var = Gauge(
            'trading_portfolio_var_usd',
            'Portfolio Value at Risk in USD',
            ['account', 'confidence_level', 'time_horizon'],
            registry=self.registry
        )
        
        self.risk_limit_utilization = Gauge(
            'trading_risk_limit_utilization',
            'Risk limit utilization percentage',
            ['limit_type', 'scope'],
            registry=self.registry
        )
        
        self.risk_alerts = Counter(
            'trading_risk_alerts_total',
            'Total risk alerts',
            ['alert_type', 'severity'],
            registry=self.registry
        )
        
        # Exposure metrics
        self.gross_exposure = Gauge(
            'trading_gross_exposure_usd',
            'Gross exposure in USD',
            ['account'],
            registry=self.registry
        )
        
        self.leverage = Gauge(
            'trading_leverage_ratio',
            'Portfolio leverage ratio',
            ['account'],
            registry=self.registry
        )
    
    def _setup_performance_metrics(self):
        """Setup performance monitoring metrics"""
        
        self.market_data_latency = Histogram(
            'trading_market_data_latency_seconds',
            'Market data processing latency',
            ['source', 'symbol'],
            buckets=[0.0001, 0.001, 0.01, 0.1, 1.0],
            registry=self.registry
        )
        
        self.errors_total = Counter(
            'trading_errors_total',
            'Total errors',
            ['component', 'error_type'],
            registry=self.registry
        )
    
    async def update_metrics(self):
        """Update all metrics from data sources"""
        try:
            await asyncio.gather(
                self._update_portfolio_metrics(),
                self._update_risk_metrics(),
                self._update_system_metrics(),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _update_portfolio_metrics(self):
        """Update portfolio-related metrics"""
        try:
            # Get portfolio data from Redis
            portfolio_data = self.redis_client.get('risk_engine:positions')
            if portfolio_data:
                positions = json.loads(portfolio_data)
                
                total_value = 0
                total_unrealized_pnl = 0
                position_counts = {'LONG': 0, 'SHORT': 0}
                
                for symbol, position in positions.items():
                    market_value = position.get('market_value', 0)
                    unrealized_pnl = position.get('unrealized_pnl', 0)
                    side = position.get('side', 'LONG')
                    
                    total_value += abs(market_value)
                    total_unrealized_pnl += unrealized_pnl
                    position_counts[side] += 1
                
                # Update metrics
                self.portfolio_value.labels(account='main', strategy='all').set(total_value)
                self.portfolio_pnl.labels(account='main', strategy='all', type='unrealized').set(total_unrealized_pnl)
                
                for side, count in position_counts.items():
                    self.position_count.labels(account='main', strategy='all', side=side).set(count)
                
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    async def _update_risk_metrics(self):
        """Update risk-related metrics"""
        try:
            risk_data = self.redis_client.get('risk_engine:metrics')
            if risk_data:
                metrics = json.loads(risk_data)
                
                for metric_name, metric_info in metrics.items():
                    value = metric_info.get('value', 0)
                    utilization = metric_info.get('utilization', 0)
                    
                    if 'var' in metric_name.lower():
                        time_horizon = 'daily' if 'daily' in metric_name else 'intraday'
                        self.portfolio_var.labels(
                            account='main',
                            confidence_level='95',
                            time_horizon=time_horizon
                        ).set(value)
                    
                    # Update risk limit utilization
                    self.risk_limit_utilization.labels(
                        limit_type=metric_name,
                        scope='portfolio'
                    ).set(utilization * 100)
                
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    async def _update_system_metrics(self):
        """Update system health metrics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            self.cpu_usage.labels(component='trading_stack').set(cpu_percent)
            self.memory_usage.labels(component='trading_stack', type='used').set(memory.used)
            
            # Component health checks
            components = {
                'redis': self._check_redis_health(),
                'postgres': self._check_postgres_health()
            }
            
            for component, is_healthy in components.items():
                self.component_status.labels(component=component, instance='main').set(1 if is_healthy else 0)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _check_redis_health(self) -> bool:
        """Check Redis connectivity"""
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def _check_postgres_health(self) -> bool:
        """Check PostgreSQL connectivity"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            return True
        except:
            return False
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)
    
    async def start_metrics_server(self, port: int = 9090):
        """Start HTTP server for metrics endpoint"""
        
        async def metrics_handler(request):
            await self.update_metrics()
            metrics_output = self.get_metrics()
            return web.Response(text=metrics_output, content_type='text/plain')
        
        async def health_handler(request):
            return web.json_response({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat()
            })
        
        app = web.Application()
        app.router.add_get('/metrics', metrics_handler)
        app.router.add_get('/health', health_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        logger.info(f"Metrics server started on port {port}")
        return runner

# Main execution
async def main():
    """Start the metrics exporter"""
    exporter = TradingStackMetricsExporter()
    
    # Start metrics server
    runner = await exporter.start_metrics_server(port=9090)
    
    try:
        # Keep the server running
        while True:
            await asyncio.sleep(30)
            await exporter.update_metrics()
            
    except KeyboardInterrupt:
        logger.info("Shutting down metrics exporter...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

