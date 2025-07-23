#!/usr/bin/env python3
"""
Momentum Trading Strategy Implementation
High-performance momentum strategy using vectorized operations and technical analysis
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import talib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    """Configuration for trading strategies"""
    name: str
    symbols: List[str]
    lookback_period: int
    rebalance_frequency: str
    risk_limit: float
    position_size: float
    transaction_cost: float
    max_positions: int = 10

class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(f"strategy.{config.name}")
        self.positions = {symbol: 0.0 for symbol in config.symbols}
        self.portfolio_value = 100000.0  # Initial portfolio value
        self.trades = []
        self.performance_metrics = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on market data"""
        pass
    
    @abstractmethod
    def calculate_position_sizes(self, signals: pd.DataFrame, 
                               current_prices: pd.Series) -> pd.Series:
        """Calculate position sizes based on signals and risk management"""
        pass
    
    def execute_trades(self, target_positions: pd.Series, 
                      current_prices: pd.Series) -> List[Dict]:
        """Execute trades to reach target positions"""
        trades = []
        
        for symbol in self.config.symbols:
            if symbol not in target_positions or symbol not in current_prices:
                continue
                
            current_position = self.positions[symbol]
            target_position = target_positions[symbol]
            trade_size = target_position - current_position
            
            if abs(trade_size) < 0.01:  # Minimum trade size
                continue
                
            price = current_prices[symbol]
            trade_value = abs(trade_size * price)
            transaction_cost = trade_value * self.config.transaction_cost
            
            trade = {
                'symbol': symbol,
                'side': 'buy' if trade_size > 0 else 'sell',
                'size': abs(trade_size),
                'price': price,
                'value': trade_value,
                'cost': transaction_cost,
                'timestamp': pd.Timestamp.now()
            }
            
            trades.append(trade)
            self.positions[symbol] = target_position
            self.trades.append(trade)
            
            self.logger.info(f"Trade executed: {trade['side']} {trade['size']:.2f} "
                           f"shares of {symbol} at ${price:.2f}")
            
        return trades
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0:
            return {}
        
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        winning_trades = (returns > 0).sum()
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(self.trades)
        }
        
        self.performance_metrics = metrics
        return metrics

class MomentumStrategy(BaseStrategy):
    """
    Advanced momentum trading strategy using multiple timeframe analysis
    """
    
    def __init__(self, config: StrategyConfig, 
                 short_window: int = 10, 
                 long_window: int = 30,
                 momentum_threshold: float = 0.02,
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9):
        super().__init__(config)
        self.short_window = short_window
        self.long_window = long_window
        self.momentum_threshold = momentum_threshold
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals using multiple technical indicators"""
        signals = pd.DataFrame(index=data.index)
        
        for symbol in self.config.symbols:
            if symbol not in data.columns:
                signals[symbol] = 0
                continue
                
            prices = data[symbol].dropna()
            if len(prices) < max(self.long_window, self.rsi_period):
                signals[symbol] = 0
                continue
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(prices)
            
            # Generate individual signals
            ma_signal = self._moving_average_signal(indicators)
            momentum_signal = self._momentum_signal(indicators)
            rsi_signal = self._rsi_signal(indicators)
            macd_signal = self._macd_signal(indicators)
            volume_signal = self._volume_signal(prices, data.get(f'{symbol}_volume', None))
            
            # Combine signals with weights
            combined_signal = (
                0.3 * ma_signal +
                0.25 * momentum_signal +
                0.2 * rsi_signal +
                0.15 * macd_signal +
                0.1 * volume_signal
            )
            
            # Apply threshold and normalize
            final_signal = np.where(combined_signal > 0.5, 1,
                                  np.where(combined_signal < -0.5, -1, 0))
            
            # Align with original data index
            signal_series = pd.Series(final_signal, index=indicators.index)
            signals[symbol] = signal_series.reindex(data.index, fill_value=0)
            
        return signals.fillna(0)
    
    def _calculate_indicators(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate all technical indicators for a symbol"""
        indicators = pd.DataFrame(index=prices.index)
        
        # Price data
        indicators['close'] = prices
        indicators['returns'] = prices.pct_change()
        
        # Moving averages
        indicators['sma_short'] = prices.rolling(window=self.short_window).mean()
        indicators['sma_long'] = prices.rolling(window=self.long_window).mean()
        indicators['ema_short'] = prices.ewm(span=self.short_window).mean()
        indicators['ema_long'] = prices.ewm(span=self.long_window).mean()
        
        # Momentum indicators
        indicators['momentum'] = prices.pct_change(self.config.lookback_period)
        indicators['roc'] = ((prices - prices.shift(self.config.lookback_period)) / 
                           prices.shift(self.config.lookback_period))
        
        # RSI
        try:
            indicators['rsi'] = talib.RSI(prices.values, timeperiod=self.rsi_period)
        except:
            # Fallback RSI calculation
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        try:
            macd, macd_signal, macd_hist = talib.MACD(prices.values, 
                                                     fastperiod=self.macd_fast,
                                                     slowperiod=self.macd_slow,
                                                     signalperiod=self.macd_signal)
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist
        except:
            # Fallback MACD calculation
            ema_fast = prices.ewm(span=self.macd_fast).mean()
            ema_slow = prices.ewm(span=self.macd_slow).mean()
            indicators['macd'] = ema_fast - ema_slow
            indicators['macd_signal'] = indicators['macd'].ewm(span=self.macd_signal).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma_bb = prices.rolling(window=bb_period).mean()
        bb_std_dev = prices.rolling(window=bb_period).std()
        indicators['bb_upper'] = sma_bb + (bb_std_dev * bb_std)
        indicators['bb_lower'] = sma_bb - (bb_std_dev * bb_std)
        indicators['bb_position'] = (prices - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Volatility
        indicators['volatility'] = prices.rolling(window=20).std()
        indicators['atr'] = self._calculate_atr(prices)
        
        return indicators.fillna(method='ffill').fillna(0)
    
    def _calculate_atr(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = prices  # Simplified - using close as high
        low = prices   # Simplified - using close as low
        close = prices
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _moving_average_signal(self, indicators: pd.DataFrame) -> pd.Series:
        """Generate moving average crossover signals"""
        # Simple MA crossover
        ma_cross = np.where(indicators['sma_short'] > indicators['sma_long'], 1, -1)
        
        # EMA crossover
        ema_cross = np.where(indicators['ema_short'] > indicators['ema_long'], 1, -1)
        
        # Price vs MA
        price_vs_ma = np.where(indicators['close'] > indicators['sma_long'], 1, -1)
        
        # Combine signals
        combined = (ma_cross + ema_cross + price_vs_ma) / 3
        
        return pd.Series(combined, index=indicators.index)
    
    def _momentum_signal(self, indicators: pd.DataFrame) -> pd.Series:
        """Generate momentum-based signals"""
        # Price momentum
        momentum = indicators['momentum']
        momentum_signal = np.where(momentum > self.momentum_threshold, 1,
                                 np.where(momentum < -self.momentum_threshold, -1, 0))
        
        # Rate of change
        roc = indicators['roc']
        roc_signal = np.where(roc > self.momentum_threshold, 1,
                            np.where(roc < -self.momentum_threshold, -1, 0))
        
        # Combine momentum signals
        combined = (momentum_signal + roc_signal) / 2
        
        return pd.Series(combined, index=indicators.index)
    
    def _rsi_signal(self, indicators: pd.DataFrame) -> pd.Series:
        """Generate RSI-based signals"""
        rsi = indicators['rsi']
        
        # Traditional RSI signals
        rsi_signal = np.where(rsi < 30, 1,      # Oversold - buy signal
                            np.where(rsi > 70, -1,  # Overbought - sell signal
                                   0))              # Neutral
        
        # RSI momentum (change in RSI)
        rsi_momentum = rsi.diff()
        rsi_mom_signal = np.where(rsi_momentum > 2, 1,
                                np.where(rsi_momentum < -2, -1, 0))
        
        # Combine RSI signals
        combined = (rsi_signal + rsi_mom_signal) / 2
        
        return pd.Series(combined, index=indicators.index)
    
    def _macd_signal(self, indicators: pd.DataFrame) -> pd.Series:
        """Generate MACD-based signals"""
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_hist = indicators['macd_histogram']
        
        # MACD line vs signal line
        macd_cross = np.where(macd > macd_signal, 1, -1)
        
        # MACD histogram momentum
        hist_momentum = macd_hist.diff()
        hist_signal = np.where(hist_momentum > 0, 1, -1)
        
        # Zero line cross
        zero_cross = np.where(macd > 0, 1, -1)
        
        # Combine MACD signals
        combined = (macd_cross + hist_signal + zero_cross) / 3
        
        return pd.Series(combined, index=indicators.index)
    
    def _volume_signal(self, prices: pd.Series, volume: pd.Series = None) -> pd.Series:
        """Generate volume-based signals"""
        if volume is None:
            # If no volume data, return neutral signal
            return pd.Series(0, index=prices.index)
        
        # Volume moving average
        volume_ma = volume.rolling(window=20).mean()
        
        # Volume ratio
        volume_ratio = volume / volume_ma
        
        # Price-volume relationship
        price_change = prices.pct_change()
        volume_signal = np.where((price_change > 0) & (volume_ratio > 1.5), 1,
                               np.where((price_change < 0) & (volume_ratio > 1.5), -1, 0))
        
        return pd.Series(volume_signal, index=prices.index)
    
    def calculate_position_sizes(self, signals: pd.DataFrame, 
                               current_prices: pd.Series) -> pd.Series:
        """Calculate position sizes with advanced risk management"""
        position_sizes = pd.Series(index=self.config.symbols, dtype=float)
        
        # Calculate portfolio-level metrics
        total_risk_budget = self.portfolio_value * self.config.risk_limit
        
        # Get active signals
        active_signals = {}
        for symbol in self.config.symbols:
            if symbol in signals.columns and symbol in current_prices:
                signal = signals[symbol].iloc[-1] if len(signals) > 0 else 0
                if abs(signal) > 0.1:  # Only consider significant signals
                    active_signals[symbol] = signal
        
        if not active_signals:
            return pd.Series(0, index=self.config.symbols)
        
        # Calculate individual position sizes
        for symbol, signal in active_signals.items():
            price = current_prices[symbol]
            
            # Calculate volatility-based position size
            returns = signals[symbol].pct_change().dropna()
            if len(returns) > 10:
                volatility = returns.std()
                # Kelly criterion approximation
                if volatility > 0:
                    expected_return = returns.mean()
                    kelly_fraction = expected_return / (volatility ** 2)
                    kelly_fraction = np.clip(kelly_fraction, -0.25, 0.25)  # Limit Kelly
                else:
                    kelly_fraction = 0
            else:
                kelly_fraction = 0.1  # Default for new strategies
            
            # Base position size
            base_size = self.config.position_size * abs(signal)
            
            # Apply Kelly adjustment
            adjusted_size = base_size * (1 + kelly_fraction)
            
            # Risk-based position sizing
            symbol_risk_budget = total_risk_budget / len(active_signals)
            max_position_value = symbol_risk_budget / (volatility if volatility > 0 else 0.02)
            
            # Final position size
            position_value = min(
                self.portfolio_value * adjusted_size,
                max_position_value
            )
            
            # Convert to shares
            shares = position_value / price if price > 0 else 0
            
            # Apply signal direction
            position_sizes[symbol] = shares * np.sign(signal)
        
        # Fill remaining symbols with zero
        for symbol in self.config.symbols:
            if symbol not in position_sizes:
                position_sizes[symbol] = 0
        
        return position_sizes
    
    def get_strategy_state(self) -> Dict:
        """Get current strategy state for monitoring"""
        return {
            'name': self.config.name,
            'positions': self.positions.copy(),
            'portfolio_value': self.portfolio_value,
            'total_trades': len(self.trades),
            'performance_metrics': self.performance_metrics.copy(),
            'parameters': {
                'short_window': self.short_window,
                'long_window': self.long_window,
                'momentum_threshold': self.momentum_threshold,
                'rsi_period': self.rsi_period
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    # Generate synthetic price data with realistic characteristics
    np.random.seed(42)
    data = {}
    
    for symbol in symbols:
        # Generate correlated random walk with trend and volatility clustering
        n_days = len(dates)
        returns = []
        volatility = 0.02
        
        for i in range(n_days):
            # Add volatility clustering
            if i > 0:
                volatility = 0.8 * volatility + 0.2 * abs(returns[-1])
                volatility = np.clip(volatility, 0.01, 0.05)
            
            # Add slight positive drift
            daily_return = np.random.normal(0.0005, volatility)
            returns.append(daily_return)
        
        # Convert to prices
        prices = 100 * np.exp(np.cumsum(returns))
        data[symbol] = prices
    
    price_data = pd.DataFrame(data, index=dates)
    
    # Create strategy configuration
    config = StrategyConfig(
        name="advanced_momentum",
        symbols=symbols,
        lookback_period=20,
        rebalance_frequency="daily",
        risk_limit=0.02,
        position_size=0.15,
        transaction_cost=0.001,
        max_positions=5
    )
    
    # Create and test strategy
    strategy = MomentumStrategy(
        config=config,
        short_window=10,
        long_window=30,
        momentum_threshold=0.02
    )
    
    print(f"Testing {strategy.config.name} strategy...")
    print(f"Symbols: {strategy.config.symbols}")
    print(f"Data period: {price_data.index[0]} to {price_data.index[-1]}")
    
    # Generate signals
    signals = strategy.generate_signals(price_data)
    print(f"\nGenerated signals shape: {signals.shape}")
    print(f"Signal statistics:")
    print(signals.describe())
    
    # Test position sizing
    current_prices = price_data.iloc[-1]
    position_sizes = strategy.calculate_position_sizes(signals, current_prices)
    
    print(f"\nPosition sizes:")
    for symbol, size in position_sizes.items():
        if abs(size) > 0.01:
            print(f"{symbol}: {size:.2f} shares")
    
    # Simulate strategy performance
    portfolio_values = [strategy.portfolio_value]
    
    for i in range(50, len(price_data)):  # Start after warmup period
        current_data = price_data.iloc[:i+1]
        current_signals = strategy.generate_signals(current_data)
        current_prices = price_data.iloc[i]
        
        # Calculate position sizes
        target_positions = strategy.calculate_position_sizes(current_signals, current_prices)
        
        # Execute trades (simplified)
        trades = strategy.execute_trades(target_positions, current_prices)
        
        # Update portfolio value (simplified)
        if i > 50:  # After first trades
            price_changes = price_data.iloc[i] / price_data.iloc[i-1] - 1
            portfolio_return = sum(strategy.positions[symbol] * price_changes[symbol] * current_prices[symbol] 
                                 for symbol in symbols if symbol in strategy.positions) / strategy.portfolio_value
            strategy.portfolio_value *= (1 + portfolio_return)
        
        portfolio_values.append(strategy.portfolio_value)
    
    # Calculate performance metrics
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
    metrics = strategy.calculate_performance_metrics(portfolio_returns)
    
    print(f"\nStrategy Performance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            if 'return' in metric or 'ratio' in metric:
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print(f"\nFinal portfolio value: ${strategy.portfolio_value:,.2f}")
    print(f"Total return: {(strategy.portfolio_value / 100000 - 1):.2%}")
    
    # Show strategy state
    state = strategy.get_strategy_state()
    print(f"\nStrategy State:")
    print(f"Active positions: {sum(1 for pos in state['positions'].values() if abs(pos) > 0.01)}")
    print(f"Total trades executed: {state['total_trades']}")

