#!/usr/bin/env python3
"""
Polygon.io Data Client for High-Frequency Trading Stack
Provides real-time equity market data ingestion with Kafka and QuestDB integration
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set
import websockets
import aiohttp
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
        logging.FileHandler('/tmp/polygon_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradeData:
    """Normalized trade data structure"""
    timestamp: int
    symbol: str
    price: float
    size: int
    exchange: str
    condition_codes: str
    sequence_number: int

class PolygonDataClient:
    """
    High-performance Polygon.io data client with Kafka integration
    Handles real-time market data streaming with automatic reconnection
    """
    
    def __init__(self, api_key: str, kafka_bootstrap_servers: str = "localhost:9092",
                 questdb_host: str = "localhost", questdb_port: int = 9009):
        self.api_key = api_key
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.questdb_host = questdb_host
        self.questdb_port = questdb_port
        
        # Initialize Kafka producer with optimized settings
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            compression_type='zstd',
            batch_size=16384,
            linger_ms=5,
            buffer_memory=33554432,  # 32MB buffer
            max_request_size=1048576,  # 1MB max request
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None
        )
        
        # Initialize QuestDB ingress client
        try:
            self.qdb_client = qdb_ingress.Sender(questdb_host, questdb_port)
        except Exception as e:
            logger.warning(f"QuestDB connection failed: {e}. Will retry later.")
            self.qdb_client = None
        
        self.websocket_url = "wss://socket.polygon.io/stocks"
        self.subscriptions: Set[str] = set()
        self.connection: Optional[websockets.WebSocketServerProtocol] = None
        self.running = False
        self.stats = {
            'messages_received': 0,
            'trades_processed': 0,
            'quotes_processed': 0,
            'errors': 0,
            'last_message_time': 0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to Polygon.io"""
        try:
            logger.info("Connecting to Polygon.io WebSocket...")
            self.connection = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10,
                max_size=10**7,  # 10MB max message size
                compression=None  # Disable compression for lower latency
            )
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "params": self.api_key
            }
            await self.connection.send(json.dumps(auth_message))
            
            # Wait for authentication confirmation
            response = await asyncio.wait_for(self.connection.recv(), timeout=10)
            auth_response = json.loads(response)
            
            if not auth_response or auth_response[0].get("status") != "auth_success":
                raise Exception(f"Authentication failed: {auth_response}")
                
            logger.info("Successfully authenticated with Polygon.io")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Polygon.io: {e}")
            if self.connection:
                await self.connection.close()
                self.connection = None
            return False
    
    async def subscribe_to_trades(self, symbols: List[str]) -> bool:
        """Subscribe to real-time trade data for specified symbols"""
        if not self.connection:
            if not await self.connect():
                return False
            
        try:
            # Subscribe in batches to avoid message size limits
            batch_size = 100
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                subscription_message = {
                    "action": "subscribe",
                    "params": f"T.{',T.'.join(batch)}"
                }
                
                await self.connection.send(json.dumps(subscription_message))
                self.subscriptions.update(f"T.{symbol}" for symbol in batch)
                
                # Small delay between batches
                if i + batch_size < len(symbols):
                    await asyncio.sleep(0.1)
            
            logger.info(f"Subscribed to trades for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to trades: {e}")
            return False
    
    async def subscribe_to_quotes(self, symbols: List[str]) -> bool:
        """Subscribe to real-time quote data for specified symbols"""
        if not self.connection:
            if not await self.connect():
                return False
            
        try:
            batch_size = 100
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                subscription_message = {
                    "action": "subscribe",
                    "params": f"Q.{',Q.'.join(batch)}"
                }
                
                await self.connection.send(json.dumps(subscription_message))
                self.subscriptions.update(f"Q.{symbol}" for symbol in batch)
                
                if i + batch_size < len(symbols):
                    await asyncio.sleep(0.1)
            
            logger.info(f"Subscribed to quotes for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to quotes: {e}")
            return False
    
    async def process_messages(self):
        """Process incoming WebSocket messages and route to Kafka/QuestDB"""
        self.running = True
        
        while self.running:
            try:
                if not self.connection:
                    logger.warning("No connection available, attempting to reconnect...")
                    if not await self.reconnect():
                        await asyncio.sleep(5)
                        continue
                
                # Process messages with timeout
                try:
                    message = await asyncio.wait_for(self.connection.recv(), timeout=60)
                    self.stats['messages_received'] += 1
                    self.stats['last_message_time'] = time.time()
                    
                    data = json.loads(message)
                    
                    # Handle different message types
                    for item in data:
                        event_type = item.get("ev")
                        
                        if event_type == "T":  # Trade event
                            await self._process_trade(item)
                            self.stats['trades_processed'] += 1
                        elif event_type == "Q":  # Quote event
                            await self._process_quote(item)
                            self.stats['quotes_processed'] += 1
                        elif event_type == "status":
                            logger.info(f"Status message: {item}")
                        
                except asyncio.TimeoutError:
                    logger.warning("Message timeout, checking connection...")
                    await self._ping_connection()
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, attempting reconnect...")
                await self.reconnect()
            except Exception as e:
                logger.error(f"Error processing messages: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(1)
    
    async def _ping_connection(self):
        """Send ping to check connection health"""
        try:
            if self.connection:
                await self.connection.ping()
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            self.connection = None
    
    async def _process_trade(self, trade_data: Dict):
        """Process individual trade message"""
        try:
            # Normalize trade data
            normalized_trade = {
                "timestamp": trade_data["t"] * 1000,  # Convert to microseconds
                "symbol": trade_data["sym"],
                "price": float(trade_data["p"]),
                "size": int(trade_data["s"]),
                "exchange": trade_data.get("x", ""),
                "condition_codes": ",".join(trade_data.get("c", [])),
                "sequence_number": trade_data.get("q", 0)
            }
            
            # Send to Kafka for real-time processing
            self.kafka_producer.send(
                "equity-ticks", 
                value=normalized_trade,
                key=normalized_trade["symbol"]
            )
            
            # Direct write to QuestDB if available
            if self.qdb_client:
                try:
                    self.qdb_client.row(
                        "equity_ticks",
                        symbols={
                            "symbol": normalized_trade["symbol"], 
                            "exchange": normalized_trade["exchange"]
                        },
                        columns={
                            "price": normalized_trade["price"],
                            "size": normalized_trade["size"],
                            "condition_codes": normalized_trade["condition_codes"],
                            "sequence_number": normalized_trade["sequence_number"]
                        },
                        at=normalized_trade["timestamp"]
                    )
                except Exception as e:
                    logger.warning(f"QuestDB write failed: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
            self.stats['errors'] += 1
    
    async def _process_quote(self, quote_data: Dict):
        """Process individual quote message"""
        try:
            normalized_quote = {
                "timestamp": quote_data["t"] * 1000,
                "symbol": quote_data["sym"],
                "bid_price": float(quote_data.get("bp", 0)),
                "bid_size": int(quote_data.get("bs", 0)),
                "ask_price": float(quote_data.get("ap", 0)),
                "ask_size": int(quote_data.get("as", 0)),
                "exchange": quote_data.get("x", ""),
                "sequence_number": quote_data.get("q", 0)
            }
            
            # Send to Kafka
            self.kafka_producer.send(
                "equity-quotes",
                value=normalized_quote,
                key=normalized_quote["symbol"]
            )
            
        except Exception as e:
            logger.error(f"Error processing quote: {e}")
            self.stats['errors'] += 1
    
    async def reconnect(self) -> bool:
        """Implement exponential backoff reconnection strategy"""
        max_retries = 10
        base_delay = 1
        
        for attempt in range(max_retries):
            if not self.running:
                return False
                
            try:
                delay = min(base_delay * (2 ** attempt), 60)  # Max 60 second delay
                logger.info(f"Reconnection attempt {attempt + 1} in {delay} seconds")
                await asyncio.sleep(delay)
                
                if await self.connect():
                    # Re-subscribe to previous subscriptions
                    if self.subscriptions:
                        trade_symbols = [sub.replace("T.", "") for sub in self.subscriptions 
                                       if sub.startswith("T.")]
                        quote_symbols = [sub.replace("Q.", "") for sub in self.subscriptions 
                                       if sub.startswith("Q.")]
                        
                        if trade_symbols:
                            await self.subscribe_to_trades(trade_symbols)
                        if quote_symbols:
                            await self.subscribe_to_quotes(quote_symbols)
                    
                    logger.info("Successfully reconnected and resubscribed")
                    return True
                
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
                
        logger.error("Max reconnection attempts exceeded")
        return False
    
    def get_stats(self) -> Dict:
        """Get client statistics"""
        current_time = time.time()
        return {
            **self.stats,
            'uptime_seconds': current_time - self.stats.get('start_time', current_time),
            'connection_status': 'connected' if self.connection else 'disconnected'
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Polygon client...")
        self.running = False
        
        if self.connection:
            await self.connection.close()
        
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
        
        if self.qdb_client:
            self.qdb_client.close()
        
        logger.info("Polygon client shutdown complete")

async def main():
    """Main entry point for standalone execution"""
    import os
    
    # Get API key from environment
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.error("POLYGON_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize client
    client = PolygonDataClient(api_key)
    client.stats['start_time'] = time.time()
    
    # Example symbols for testing
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"]
    
    try:
        # Connect and subscribe
        if await client.connect():
            await client.subscribe_to_trades(symbols)
            await client.subscribe_to_quotes(symbols[:4])  # Subset for quotes
            
            # Start processing messages
            await client.process_messages()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await client.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

