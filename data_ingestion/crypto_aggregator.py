#!/usr/bin/env python3
"""
Multi-Exchange Cryptocurrency Data Aggregator
Aggregates real-time crypto market data from multiple exchanges
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Set, Optional
import websockets
from kafka import KafkaProducer
import questdb.ingress as qdb_ingress
from dataclasses import dataclass
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/crypto_aggregator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CryptoTrade:
    """Normalized cryptocurrency trade data"""
    timestamp: int
    symbol: str
    price: float
    size: float
    side: str
    trade_id: str
    exchange: str

class CryptoDataAggregator:
    """
    Multi-exchange cryptocurrency data aggregator
    Supports Binance, Coinbase Pro, and other major exchanges
    """
    
    def __init__(self, kafka_bootstrap_servers: str = "localhost:9092",
                 questdb_host: str = "localhost", questdb_port: int = 9009):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.questdb_host = questdb_host
        self.questdb_port = questdb_port
        
        # Initialize Kafka producer
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            compression_type='zstd',
            batch_size=16384,
            linger_ms=5,
            buffer_memory=33554432,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None
        )
        
        # Initialize QuestDB client
        try:
            self.qdb_client = qdb_ingress.Sender(questdb_host, questdb_port)
        except Exception as e:
            logger.warning(f"QuestDB connection failed: {e}")
            self.qdb_client = None
        
        self.connections = {}
        self.subscriptions = {}
        self.running = False
        self.stats = {
            'binance': {'trades': 0, 'errors': 0, 'last_message': 0},
            'coinbase': {'trades': 0, 'errors': 0, 'last_message': 0},
            'total_trades': 0,
            'start_time': time.time()
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def connect_binance(self, symbols: List[str]) -> bool:
        """Connect to Binance WebSocket streams"""
        try:
            # Convert symbols to Binance format (lowercase)
            binance_symbols = [symbol.lower().replace('-', '') for symbol in symbols]
            streams = [f"{symbol}@trade" for symbol in binance_symbols]
            
            # Binance allows up to 1024 streams per connection
            if len(streams) > 1024:
                logger.warning(f"Too many symbols ({len(streams)}), limiting to 1024")
                streams = streams[:1024]
            
            stream_names = "/".join(streams)
            url = f"wss://stream.binance.com:9443/ws/{stream_names}"
            
            logger.info(f"Connecting to Binance with {len(streams)} streams...")
            connection = await websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=10,
                max_size=10**7
            )
            
            self.connections["binance"] = connection
            self.subscriptions["binance"] = symbols
            
            logger.info(f"Connected to Binance with {len(symbols)} symbols")
            
            # Start message processing task
            asyncio.create_task(self._process_binance_messages())
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            return False
    
    async def connect_coinbase(self, symbols: List[str]) -> bool:
        """Connect to Coinbase Pro WebSocket feed"""
        try:
            url = "wss://ws-feed.exchange.coinbase.com"
            
            logger.info(f"Connecting to Coinbase Pro...")
            connection = await websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.connections["coinbase"] = connection
            self.subscriptions["coinbase"] = symbols
            
            # Subscribe to matches channel (trades)
            subscribe_message = {
                "type": "subscribe",
                "product_ids": symbols,
                "channels": ["matches"]
            }
            
            await connection.send(json.dumps(subscribe_message))
            
            logger.info(f"Connected to Coinbase Pro with {len(symbols)} symbols")
            
            # Start message processing task
            asyncio.create_task(self._process_coinbase_messages())
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase Pro: {e}")
            return False
    
    async def _process_binance_messages(self):
        """Process Binance WebSocket messages"""
        connection = self.connections["binance"]
        exchange = "binance"
        
        try:
            async for message in connection:
                if not self.running:
                    break
                    
                try:
                    data = json.loads(message)
                    
                    if "e" in data and data["e"] == "trade":
                        await self._process_binance_trade(data)
                        self.stats[exchange]['trades'] += 1
                        self.stats[exchange]['last_message'] = time.time()
                        self.stats['total_trades'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing Binance message: {e}")
                    self.stats[exchange]['errors'] += 1
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Binance connection closed")
        except Exception as e:
            logger.error(f"Binance message processing error: {e}")
        finally:
            if exchange in self.connections:
                del self.connections[exchange]
    
    async def _process_coinbase_messages(self):
        """Process Coinbase Pro WebSocket messages"""
        connection = self.connections["coinbase"]
        exchange = "coinbase"
        
        try:
            async for message in connection:
                if not self.running:
                    break
                    
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "match":
                        await self._process_coinbase_trade(data)
                        self.stats[exchange]['trades'] += 1
                        self.stats[exchange]['last_message'] = time.time()
                        self.stats['total_trades'] += 1
                    elif data.get("type") == "subscriptions":
                        logger.info(f"Coinbase subscription confirmed: {data}")
                        
                except Exception as e:
                    logger.error(f"Error processing Coinbase message: {e}")
                    self.stats[exchange]['errors'] += 1
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Coinbase connection closed")
        except Exception as e:
            logger.error(f"Coinbase message processing error: {e}")
        finally:
            if exchange in self.connections:
                del self.connections[exchange]
    
    async def _process_binance_trade(self, trade_data: Dict):
        """Process Binance trade data"""
        try:
            normalized_trade = {
                "timestamp": trade_data["T"] * 1000,  # Convert to microseconds
                "symbol": trade_data["s"].upper(),
                "price": float(trade_data["p"]),
                "size": float(trade_data["q"]),
                "side": "sell" if trade_data["m"] else "buy",  # m=true means buyer is market maker
                "trade_id": str(trade_data["t"]),
                "exchange": "binance"
            }
            
            await self._route_crypto_data(normalized_trade)
            
        except Exception as e:
            logger.error(f"Error processing Binance trade: {e}")
    
    async def _process_coinbase_trade(self, trade_data: Dict):
        """Process Coinbase Pro trade data"""
        try:
            # Parse timestamp
            timestamp_str = trade_data["time"]
            # Convert ISO timestamp to microseconds
            import datetime
            dt = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            timestamp_us = int(dt.timestamp() * 1000000)
            
            normalized_trade = {
                "timestamp": timestamp_us,
                "symbol": trade_data["product_id"],
                "price": float(trade_data["price"]),
                "size": float(trade_data["size"]),
                "side": trade_data["side"],
                "trade_id": str(trade_data["trade_id"]),
                "exchange": "coinbase"
            }
            
            await self._route_crypto_data(normalized_trade)
            
        except Exception as e:
            logger.error(f"Error processing Coinbase trade: {e}")
    
    async def _route_crypto_data(self, trade_data: Dict):
        """Route normalized crypto data to storage systems"""
        try:
            # Send to Kafka
            self.kafka_producer.send(
                "crypto-ticks",
                value=trade_data,
                key=f"{trade_data['exchange']}:{trade_data['symbol']}"
            )
            
            # Write to QuestDB if available
            if self.qdb_client:
                try:
                    self.qdb_client.row(
                        "crypto_ticks",
                        symbols={
                            "symbol": trade_data["symbol"],
                            "side": trade_data["side"],
                            "exchange": trade_data["exchange"]
                        },
                        columns={
                            "price": trade_data["price"],
                            "size": trade_data["size"],
                            "trade_id": trade_data["trade_id"]
                        },
                        at=trade_data["timestamp"]
                    )
                except Exception as e:
                    logger.warning(f"QuestDB write failed: {e}")
                    
        except Exception as e:
            logger.error(f"Error routing crypto data: {e}")
    
    async def start_aggregation(self, binance_symbols: List[str] = None, 
                               coinbase_symbols: List[str] = None):
        """Start data aggregation from specified exchanges"""
        self.running = True
        tasks = []
        
        # Connect to exchanges
        if binance_symbols:
            if await self.connect_binance(binance_symbols):
                logger.info("Binance connection established")
            else:
                logger.error("Failed to connect to Binance")
        
        if coinbase_symbols:
            if await self.connect_coinbase(coinbase_symbols):
                logger.info("Coinbase connection established")
            else:
                logger.error("Failed to connect to Coinbase")
        
        # Start monitoring task
        tasks.append(asyncio.create_task(self._monitor_connections()))
        
        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in aggregation tasks: {e}")
    
    async def _monitor_connections(self):
        """Monitor connection health and statistics"""
        while self.running:
            try:
                current_time = time.time()
                
                # Log statistics every 60 seconds
                if int(current_time) % 60 == 0:
                    uptime = current_time - self.stats['start_time']
                    logger.info(f"Stats - Uptime: {uptime:.0f}s, Total trades: {self.stats['total_trades']}")
                    
                    for exchange in ['binance', 'coinbase']:
                        if exchange in self.stats:
                            stats = self.stats[exchange]
                            last_msg_age = current_time - stats['last_message']
                            logger.info(f"{exchange.title()}: {stats['trades']} trades, "
                                      f"{stats['errors']} errors, last message {last_msg_age:.1f}s ago")
                
                # Check for stale connections (no messages for 5 minutes)
                for exchange in ['binance', 'coinbase']:
                    if exchange in self.stats and exchange in self.connections:
                        last_message = self.stats[exchange]['last_message']
                        if last_message > 0 and current_time - last_message > 300:
                            logger.warning(f"{exchange} connection appears stale, attempting reconnect")
                            await self._reconnect_exchange(exchange)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _reconnect_exchange(self, exchange: str):
        """Reconnect to a specific exchange"""
        try:
            # Close existing connection
            if exchange in self.connections:
                await self.connections[exchange].close()
                del self.connections[exchange]
            
            # Reconnect based on exchange
            symbols = self.subscriptions.get(exchange, [])
            if not symbols:
                return
            
            if exchange == "binance":
                await self.connect_binance(symbols)
            elif exchange == "coinbase":
                await self.connect_coinbase(symbols)
                
            logger.info(f"Successfully reconnected to {exchange}")
            
        except Exception as e:
            logger.error(f"Failed to reconnect to {exchange}: {e}")
    
    def get_stats(self) -> Dict:
        """Get aggregator statistics"""
        current_time = time.time()
        return {
            **self.stats,
            'uptime_seconds': current_time - self.stats['start_time'],
            'active_connections': list(self.connections.keys()),
            'subscribed_symbols': {k: len(v) for k, v in self.subscriptions.items()}
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down crypto aggregator...")
        self.running = False
        
        # Close all connections
        for exchange, connection in self.connections.items():
            try:
                await connection.close()
                logger.info(f"Closed {exchange} connection")
            except Exception as e:
                logger.warning(f"Error closing {exchange} connection: {e}")
        
        # Close Kafka producer
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
        
        # Close QuestDB client
        if self.qdb_client:
            self.qdb_client.close()
        
        logger.info("Crypto aggregator shutdown complete")

async def main():
    """Main entry point for standalone execution"""
    
    # Initialize aggregator
    aggregator = CryptoDataAggregator()
    
    # Define symbols for each exchange
    binance_symbols = [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
        "LTCUSDT", "BCHUSDT", "XLMUSDT", "EOSUSDT", "TRXUSDT"
    ]
    
    coinbase_symbols = [
        "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD",
        "LTC-USD", "BCH-USD", "XLM-USD", "EOS-USD"
    ]
    
    try:
        # Start aggregation
        await aggregator.start_aggregation(
            binance_symbols=binance_symbols,
            coinbase_symbols=coinbase_symbols
        )
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await aggregator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

