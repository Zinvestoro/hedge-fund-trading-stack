#!/usr/bin/env python3
"""
Custom FinRL Environment for Trading Stack Integration
Provides seamless integration between FinRL and the trading stack infrastructure
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import questdb.ingress as qdb_ingress
from typing import Dict, List, Tuple, Optional
import logging
import psycopg2
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingStackEnvironment(gym.Env):
    """
    Custom FinRL environment integrated with Trading Stack infrastructure
    Supports both historical backtesting and live trading simulation
    """
    
    def __init__(self, 
                 symbols: List[str],
                 start_date: str,
                 end_date: str,
                 initial_balance: float = 100000,
                 transaction_cost: float = 0.001,
                 questdb_host: str = "localhost",
                 questdb_port: int = 8812,
                 mode: str = "backtest",
                 lookback_window: int = 30,
                 max_position_size: float = 0.1):
        
        super().__init__()
        
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.mode = mode
        self.lookback_window = lookback_window
        self.max_position_size = max_position_size
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.positions = {symbol: 0 for symbol in symbols}
        self.portfolio_value = initial_balance
        self.trades = []
        self.portfolio_history = []
        
        # Data connection
        self.questdb_host = questdb_host
        self.questdb_port = questdb_port
        self.data = None
        self.price_data = None
        
        # Action and observation spaces
        # Actions: continuous values [-1, 1] for each symbol (sell, hold, buy)
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(len(symbols),), 
            dtype=np.float32
        )
        
        # Observations: technical indicators + portfolio state
        n_features_per_symbol = 12  # price, returns, sma, ema, rsi, macd, etc.
        n_portfolio_features = len(symbols) + 3  # positions + cash + portfolio metrics
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(symbols) * n_features_per_symbol + n_portfolio_features,),
            dtype=np.float32
        )
        
        # Performance tracking
        self.episode_returns = []
        self.max_drawdown = 0
        self.peak_portfolio_value = initial_balance
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load historical data from QuestDB"""
        try:
            logger.info(f"Loading data for symbols: {self.symbols}")
            logger.info(f"Date range: {self.start_date} to {self.end_date}")
            
            # Connect to QuestDB via PostgreSQL protocol
            conn = psycopg2.connect(
                host=self.questdb_host,
                port=self.questdb_port,
                database="qdb",
                user="admin",
                password="quest"
            )
            
            # Build query for multiple symbols
            symbols_str = "', '".join(self.symbols)
            query = f"""
            SELECT 
                timestamp,
                symbol,
                price,
                size
            FROM equity_ticks 
            WHERE symbol IN ('{symbols_str}')
                AND timestamp BETWEEN '{self.start_date}' AND '{self.end_date}'
            ORDER BY timestamp, symbol
            """
            
            self.data = pd.read_sql(query, conn)
            conn.close()
            
            if len(self.data) > 0:
                logger.info(f"Loaded {len(self.data)} data points from QuestDB")
                self._process_data()
            else:
                logger.warning("No data found in QuestDB, generating synthetic data")
                self._generate_synthetic_data()
            
        except Exception as e:
            logger.error(f"Failed to load data from QuestDB: {e}")
            logger.info("Generating synthetic data for testing")
            self._generate_synthetic_data()
    
    def _process_data(self):
        """Process raw data into features for RL"""
        logger.info("Processing market data...")
        
        # Resample to 1-minute bars for consistency
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data.set_index('timestamp', inplace=True)
        
        # Create OHLCV data
        ohlcv_data = self.data.groupby(['symbol']).resample('1min').agg({
            'price': ['first', 'max', 'min', 'last'],
            'size': 'sum'
        }).fillna(method='ffill')
        
        # Flatten column names
        ohlcv_data.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlcv_data = ohlcv_data.reset_index()
        
        # Pivot to have symbols as columns
        price_data = {}
        
        for symbol in self.symbols:
            symbol_data = ohlcv_data[ohlcv_data['symbol'] == symbol].set_index('timestamp')
            
            if len(symbol_data) > 0:
                # Basic price data
                price_data[f'{symbol}_open'] = symbol_data['open']
                price_data[f'{symbol}_high'] = symbol_data['high']
                price_data[f'{symbol}_low'] = symbol_data['low']
                price_data[f'{symbol}_close'] = symbol_data['close']
                price_data[f'{symbol}_volume'] = symbol_data['volume']
                
                # Technical indicators
                close_prices = symbol_data['close']
                
                # Returns
                price_data[f'{symbol}_returns'] = close_prices.pct_change()
                price_data[f'{symbol}_log_returns'] = np.log(close_prices / close_prices.shift(1))
                
                # Moving averages
                price_data[f'{symbol}_sma_10'] = close_prices.rolling(10).mean()
                price_data[f'{symbol}_sma_30'] = close_prices.rolling(30).mean()
                price_data[f'{symbol}_ema_12'] = close_prices.ewm(span=12).mean()
                price_data[f'{symbol}_ema_26'] = close_prices.ewm(span=26).mean()
                
                # Volatility
                price_data[f'{symbol}_volatility'] = close_prices.rolling(20).std()
                
                # RSI
                price_data[f'{symbol}_rsi'] = self._calculate_rsi(close_prices)
                
                # MACD
                ema_12 = close_prices.ewm(span=12).mean()
                ema_26 = close_prices.ewm(span=26).mean()
                price_data[f'{symbol}_macd'] = ema_12 - ema_26
                price_data[f'{symbol}_macd_signal'] = (ema_12 - ema_26).ewm(span=9).mean()
        
        self.price_data = pd.DataFrame(price_data).fillna(method='ffill').fillna(0)
        logger.info(f"Processed data shape: {self.price_data.shape}")
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing when real data unavailable"""
        logger.info("Generating synthetic market data...")
        
        # Create date range
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        dates = pd.date_range(start=start, end=end, freq='1min')
        
        # Generate correlated random walks for multiple symbols
        np.random.seed(42)
        n_steps = len(dates)
        n_symbols = len(self.symbols)
        
        # Create correlation matrix
        correlation = np.random.uniform(0.3, 0.7, (n_symbols, n_symbols))
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 1.0)
        
        # Generate correlated returns
        returns = np.random.multivariate_normal(
            mean=[0.0001] * n_symbols,
            cov=correlation * 0.0002,  # Lower volatility for more realistic data
            size=n_steps
        )
        
        # Convert to prices
        initial_prices = np.random.uniform(50, 200, n_symbols)
        prices = initial_prices * np.exp(np.cumsum(returns, axis=0))
        
        # Generate volume data
        volumes = np.random.lognormal(mean=10, sigma=1, size=(n_steps, n_symbols))
        
        # Create DataFrame with OHLCV data
        price_data = {}
        
        for i, symbol in enumerate(self.symbols):
            symbol_prices = pd.Series(prices[:, i], index=dates)
            symbol_volumes = pd.Series(volumes[:, i], index=dates)
            
            # Create OHLC from close prices (simplified)
            price_data[f'{symbol}_close'] = symbol_prices
            price_data[f'{symbol}_open'] = symbol_prices.shift(1).fillna(symbol_prices.iloc[0])
            price_data[f'{symbol}_high'] = symbol_prices * (1 + np.random.uniform(0, 0.02, len(symbol_prices)))
            price_data[f'{symbol}_low'] = symbol_prices * (1 - np.random.uniform(0, 0.02, len(symbol_prices)))
            price_data[f'{symbol}_volume'] = symbol_volumes
            
            # Technical indicators
            close_prices = symbol_prices
            
            # Returns
            price_data[f'{symbol}_returns'] = close_prices.pct_change()
            price_data[f'{symbol}_log_returns'] = np.log(close_prices / close_prices.shift(1))
            
            # Moving averages
            price_data[f'{symbol}_sma_10'] = close_prices.rolling(10).mean()
            price_data[f'{symbol}_sma_30'] = close_prices.rolling(30).mean()
            price_data[f'{symbol}_ema_12'] = close_prices.ewm(span=12).mean()
            price_data[f'{symbol}_ema_26'] = close_prices.ewm(span=26).mean()
            
            # Volatility
            price_data[f'{symbol}_volatility'] = close_prices.rolling(20).std()
            
            # RSI
            price_data[f'{symbol}_rsi'] = self._calculate_rsi(close_prices)
            
            # MACD
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            price_data[f'{symbol}_macd'] = ema_12 - ema_26
            price_data[f'{symbol}_macd_signal'] = (ema_12 - ema_26).ewm(span=9).mean()
        
        self.price_data = pd.DataFrame(price_data).fillna(method='ffill').fillna(0)
        logger.info(f"Generated synthetic data shape: {self.price_data.shape}")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window  # Start after lookback period
        self.balance = self.initial_balance
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance
        self.trades = []
        self.portfolio_history = [self.initial_balance]
        self.peak_portfolio_value = self.initial_balance
        self.max_drawdown = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.price_data) - 1:
            return self._get_observation(), 0, True, True, {}
        
        # Execute trades based on action
        reward = self._execute_trades(action)
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Track portfolio history
        self.portfolio_history.append(self.portfolio_value)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= len(self.price_data) - 1) or (self.portfolio_value <= self.initial_balance * 0.1)
        
        return self._get_observation(), reward, done, False, self._get_info()
    
    def _execute_trades(self, action: np.ndarray) -> float:
        """Execute trades based on action and calculate reward"""
        current_prices = self._get_current_prices()
        total_cost = 0
        previous_portfolio_value = self.portfolio_value
        
        for i, symbol in enumerate(self.symbols):
            if symbol not in current_prices or current_prices[symbol] <= 0:
                continue
                
            current_price = current_prices[symbol]
            action_value = np.clip(action[i], -1, 1)  # Ensure action is in valid range
            
            # Skip small actions
            if abs(action_value) < 0.05:
                continue
            
            # Calculate target position value
            target_position_value = action_value * self.portfolio_value * self.max_position_size
            target_shares = target_position_value / current_price
            
            # Calculate trade size
            current_shares = self.positions[symbol]
            trade_shares = target_shares - current_shares
            
            if abs(trade_shares) < 0.001:  # Minimum trade size
                continue
            
            # Execute trade
            trade_value = abs(trade_shares * current_price)
            transaction_cost = trade_value * self.transaction_cost
            
            if trade_shares > 0:  # Buy
                required_cash = trade_value + transaction_cost
                if self.balance >= required_cash:
                    self.balance -= required_cash
                    self.positions[symbol] += trade_shares
                    total_cost += transaction_cost
                    
                    self.trades.append({
                        'step': self.current_step,
                        'symbol': symbol,
                        'side': 'buy',
                        'shares': trade_shares,
                        'price': current_price,
                        'value': trade_value,
                        'cost': transaction_cost
                    })
            
            else:  # Sell
                shares_to_sell = abs(trade_shares)
                if self.positions[symbol] >= shares_to_sell:
                    self.balance += (trade_value - transaction_cost)
                    self.positions[symbol] -= shares_to_sell
                    total_cost += transaction_cost
                    
                    self.trades.append({
                        'step': self.current_step,
                        'symbol': symbol,
                        'side': 'sell',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'value': trade_value,
                        'cost': transaction_cost
                    })
        
        # Calculate reward
        self._update_portfolio_value()
        
        # Portfolio return
        if previous_portfolio_value > 0:
            portfolio_return = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        else:
            portfolio_return = 0
        
        # Risk-adjusted reward
        transaction_cost_penalty = total_cost / previous_portfolio_value if previous_portfolio_value > 0 else 0
        
        # Add Sharpe ratio component if we have enough history
        if len(self.portfolio_history) > 30:
            returns = np.diff(self.portfolio_history[-30:]) / np.array(self.portfolio_history[-31:-1])
            returns = returns[~np.isnan(returns)]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_bonus = np.mean(returns) / np.std(returns) * 0.01
            else:
                sharpe_bonus = 0
        else:
            sharpe_bonus = 0
        
        reward = portfolio_return - transaction_cost_penalty + sharpe_bonus
        
        return reward
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        current_prices = {}
        
        for symbol in self.symbols:
            close_col = f'{symbol}_close'
            if close_col in self.price_data.columns and self.current_step < len(self.price_data):
                price = self.price_data.iloc[self.current_step][close_col]
                if not np.isnan(price) and price > 0:
                    current_prices[symbol] = float(price)
        
        return current_prices
    
    def _update_portfolio_value(self):
        """Update total portfolio value"""
        current_prices = self._get_current_prices()
        
        position_value = 0
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
                position_value += shares * current_prices[symbol]
        
        self.portfolio_value = self.balance + position_value
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        # Update peak portfolio value
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        
        # Calculate current drawdown
        if self.peak_portfolio_value > 0:
            current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        if self.current_step >= len(self.price_data):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Technical indicators for each symbol
        technical_features = []
        
        for symbol in self.symbols:
            symbol_features = []
            
            # Price features
            for feature in ['close', 'returns', 'log_returns', 'sma_10', 'sma_30', 
                          'ema_12', 'ema_26', 'volatility', 'rsi', 'macd', 'macd_signal', 'volume']:
                col_name = f'{symbol}_{feature}'
                if col_name in self.price_data.columns:
                    value = self.price_data.iloc[self.current_step][col_name]
                    # Normalize features
                    if feature == 'close':
                        # Normalize price by its 30-day moving average
                        sma_col = f'{symbol}_sma_30'
                        if sma_col in self.price_data.columns:
                            sma_value = self.price_data.iloc[self.current_step][sma_col]
                            if sma_value > 0:
                                value = value / sma_value - 1
                            else:
                                value = 0
                        else:
                            value = 0
                    elif feature in ['returns', 'log_returns']:
                        value = np.clip(value, -0.1, 0.1)  # Clip extreme returns
                    elif feature == 'rsi':
                        value = (value - 50) / 50  # Normalize RSI to [-1, 1]
                    elif feature == 'volume':
                        # Normalize volume by its 30-day average
                        if self.current_step >= 30:
                            avg_volume = self.price_data.iloc[self.current_step-30:self.current_step][col_name].mean()
                            if avg_volume > 0:
                                value = value / avg_volume - 1
                            else:
                                value = 0
                        else:
                            value = 0
                    
                    symbol_features.append(float(value) if not np.isnan(value) else 0.0)
                else:
                    symbol_features.append(0.0)
            
            technical_features.extend(symbol_features)
        
        # Portfolio features
        current_prices = self._get_current_prices()
        position_features = []
        
        for symbol in self.symbols:
            if symbol in current_prices and current_prices[symbol] > 0 and self.portfolio_value > 0:
                position_value = self.positions[symbol] * current_prices[symbol]
                position_weight = position_value / self.portfolio_value
                position_features.append(position_weight)
            else:
                position_features.append(0.0)
        
        # Portfolio metrics
        portfolio_features = [
            self.balance / self.portfolio_value if self.portfolio_value > 0 else 1.0,  # Cash ratio
            (self.portfolio_value / self.initial_balance) - 1,  # Total return
            -self.max_drawdown  # Max drawdown (negative)
        ]
        
        # Combine all features
        observation = np.array(technical_features + position_features + portfolio_features, dtype=np.float32)
        
        # Ensure observation matches expected shape
        expected_size = self.observation_space.shape[0]
        if len(observation) != expected_size:
            if len(observation) < expected_size:
                observation = np.pad(observation, (0, expected_size - len(observation)))
            else:
                observation = observation[:expected_size]
        
        # Replace any NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        total_return = (self.portfolio_value / self.initial_balance) - 1
        
        return {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'total_trades': len(self.trades),
            'step': self.current_step,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of portfolio returns"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        returns = np.diff(self.portfolio_history) / np.array(self.portfolio_history[:-1])
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming 252 trading days)
        return np.sqrt(252) * np.mean(returns) / np.std(returns)
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            current_prices = self._get_current_prices()
            total_return = (self.portfolio_value / self.initial_balance) - 1
            
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{len(self.price_data)-1}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash Balance: ${self.balance:,.2f}")
            print(f"Total Return: {total_return:.2%}")
            print(f"Max Drawdown: {self.max_drawdown:.2%}")
            print(f"Sharpe Ratio: {self._calculate_sharpe_ratio():.2f}")
            print(f"Total Trades: {len(self.trades)}")
            
            print(f"\nPositions:")
            for symbol in self.symbols:
                shares = self.positions[symbol]
                if shares != 0 and symbol in current_prices:
                    value = shares * current_prices[symbol]
                    weight = value / self.portfolio_value if self.portfolio_value > 0 else 0
                    print(f"  {symbol}: {shares:.2f} shares, ${value:,.2f} ({weight:.1%})")
            
            print(f"{'='*60}")
    
    def get_portfolio_stats(self) -> Dict:
        """Get comprehensive portfolio statistics"""
        if len(self.portfolio_history) < 2:
            return {}
        
        returns = np.diff(self.portfolio_history) / np.array(self.portfolio_history[:-1])
        returns = returns[~np.isnan(returns)]
        
        total_return = (self.portfolio_value / self.initial_balance) - 1
        
        stats = {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(self.portfolio_history)) - 1,
            'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self.max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor()
        }
        
        return stats
    
    def _calculate_win_rate(self) -> float:
        """Calculate percentage of profitable trades"""
        if not self.trades:
            return 0.0
        
        # Group trades by symbol to calculate P&L
        profitable_trades = 0
        total_trades = 0
        
        for symbol in self.symbols:
            symbol_trades = [t for t in self.trades if t['symbol'] == symbol]
            if len(symbol_trades) < 2:
                continue
            
            # Simple P&L calculation (buy followed by sell)
            position = 0
            entry_price = 0
            
            for trade in symbol_trades:
                if trade['side'] == 'buy':
                    if position == 0:
                        entry_price = trade['price']
                    position += trade['shares']
                else:  # sell
                    if position > 0:
                        pnl = (trade['price'] - entry_price) * min(trade['shares'], position)
                        if pnl > 0:
                            profitable_trades += 1
                        total_trades += 1
                        position -= trade['shares']
        
        return profitable_trades / total_trades if total_trades > 0 else 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.trades:
            return 0.0
        
        gross_profit = 0
        gross_loss = 0
        
        for symbol in self.symbols:
            symbol_trades = [t for t in self.trades if t['symbol'] == symbol]
            if len(symbol_trades) < 2:
                continue
            
            position = 0
            entry_price = 0
            
            for trade in symbol_trades:
                if trade['side'] == 'buy':
                    if position == 0:
                        entry_price = trade['price']
                    position += trade['shares']
                else:  # sell
                    if position > 0:
                        pnl = (trade['price'] - entry_price) * min(trade['shares'], position)
                        if pnl > 0:
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                        position -= trade['shares']
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')


# Example usage and testing
if __name__ == "__main__":
    # Test the environment
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    env = TradingStackEnvironment(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_balance=100000
    )
    
    # Test reset and step
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step {i}: Reward={reward:.4f}, Portfolio=${info['portfolio_value']:,.2f}")
        
        if done:
            break
    
    # Print final statistics
    stats = env.get_portfolio_stats()
    print(f"\nFinal Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

