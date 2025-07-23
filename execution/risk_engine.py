#!/usr/bin/env python3
"""
Real-Time Risk Management Engine
Provides comprehensive risk monitoring and control for trading operations
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskStatus(Enum):
    """Risk status enumeration"""
    OK = "OK"
    WARNING = "WARNING"
    BREACH = "BREACH"
    CRITICAL = "CRITICAL"

class LimitType(Enum):
    """Risk limit type enumeration"""
    NOTIONAL = "notional"
    VAR = "var"
    DRAWDOWN = "drawdown"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    CORRELATION = "correlation"

@dataclass
class RiskLimit:
    """Risk limit definition"""
    name: str
    limit_type: LimitType
    value: float
    currency: str = 'USD'
    scope: str = 'portfolio'  # 'portfolio', 'strategy', 'symbol'
    time_horizon: str = 'daily'  # 'intraday', 'daily', 'weekly'
    enabled: bool = True

@dataclass
class RiskMetric:
    """Risk metric calculation result"""
    name: str
    value: float
    limit: float
    utilization: float
    status: RiskStatus
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None

@dataclass
class Position:
    """Position representation for risk calculations"""
    symbol: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    cost_basis: float
    side: str  # 'LONG', 'SHORT'
    strategy_id: Optional[str] = None
    sector: Optional[str] = None
    asset_class: Optional[str] = None

@dataclass
class RiskAlert:
    """Risk alert structure"""
    alert_id: str
    timestamp: datetime
    severity: RiskStatus
    metric_name: str
    current_value: float
    limit_value: float
    message: str
    positions_affected: List[str]
    recommended_action: str

class RiskModel(ABC):
    """Abstract base class for risk models"""
    
    @abstractmethod
    def calculate_var(self, positions: List[Position], 
                     confidence_level: float = 0.05,
                     time_horizon: int = 1) -> float:
        """Calculate Value at Risk"""
        pass
    
    @abstractmethod
    def calculate_expected_shortfall(self, positions: List[Position],
                                   confidence_level: float = 0.05,
                                   time_horizon: int = 1) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        pass
    
    @abstractmethod
    def calculate_component_var(self, positions: List[Position],
                              confidence_level: float = 0.05) -> Dict[str, float]:
        """Calculate component VaR for each position"""
        pass

class HistoricalSimulationModel(RiskModel):
    """Historical simulation risk model with advanced features"""
    
    def __init__(self, lookback_days: int = 252, decay_factor: float = 0.94):
        self.lookback_days = lookback_days
        self.decay_factor = decay_factor
        self.returns_cache = {}
        self.correlation_matrix = None
        self.volatility_cache = {}
        
    def calculate_var(self, positions: List[Position], 
                     confidence_level: float = 0.05,
                     time_horizon: int = 1) -> float:
        """Calculate VaR using historical simulation with exponential weighting"""
        if not positions:
            return 0.0
        
        # Get portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(positions)
        
        if len(portfolio_returns) == 0:
            return 0.0
        
        # Apply exponential weighting
        weights = self._calculate_exponential_weights(len(portfolio_returns))
        
        # Calculate weighted percentile
        var_percentile = self._weighted_percentile(portfolio_returns, weights, confidence_level * 100)
        
        # Scale for time horizon
        time_scaling = np.sqrt(time_horizon)
        
        return abs(var_percentile) * time_scaling
    
    def calculate_expected_shortfall(self, positions: List[Position],
                                   confidence_level: float = 0.05,
                                   time_horizon: int = 1) -> float:
        """Calculate Expected Shortfall with exponential weighting"""
        if not positions:
            return 0.0
        
        portfolio_returns = self._calculate_portfolio_returns(positions)
        
        if len(portfolio_returns) == 0:
            return 0.0
        
        # Calculate VaR threshold
        weights = self._calculate_exponential_weights(len(portfolio_returns))
        var_threshold = self._weighted_percentile(portfolio_returns, weights, confidence_level * 100)
        
        # Calculate weighted expected shortfall
        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
        tail_weights = weights[portfolio_returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return abs(var_threshold)
        
        # Weighted average of tail returns
        weighted_es = np.average(tail_returns, weights=tail_weights)
        
        # Scale for time horizon
        time_scaling = np.sqrt(time_horizon)
        
        return abs(weighted_es) * time_scaling
    
    def calculate_component_var(self, positions: List[Position],
                              confidence_level: float = 0.05) -> Dict[str, float]:
        """Calculate component VaR for each position"""
        if not positions:
            return {}
        
        portfolio_var = self.calculate_var(positions, confidence_level)
        component_vars = {}
        
        # Calculate marginal VaR for each position
        for i, position in enumerate(positions):
            # Create portfolio without this position
            other_positions = positions[:i] + positions[i+1:]
            
            if other_positions:
                portfolio_var_without = self.calculate_var(other_positions, confidence_level)
                marginal_var = portfolio_var - portfolio_var_without
            else:
                marginal_var = portfolio_var
            
            # Component VaR = Marginal VaR * Position Weight
            total_portfolio_value = sum(abs(pos.market_value) for pos in positions)
            position_weight = abs(position.market_value) / total_portfolio_value if total_portfolio_value > 0 else 0
            
            component_vars[position.symbol] = marginal_var * position_weight
        
        return component_vars
    
    def _calculate_portfolio_returns(self, positions: List[Position]) -> np.ndarray:
        """Calculate historical portfolio returns"""
        portfolio_value = sum(abs(pos.market_value) for pos in positions)
        
        if portfolio_value == 0:
            return np.array([])
        
        # Generate synthetic correlated returns for demonstration
        # In production, this would fetch actual historical data
        np.random.seed(42)
        
        n_days = min(self.lookback_days, 252)
        n_assets = len(positions)
        
        # Create correlation matrix
        correlation = self._generate_correlation_matrix(positions)
        
        # Generate correlated returns
        mean_returns = np.array([0.0005] * n_assets)  # Slight positive drift
        volatilities = np.array([self._estimate_volatility(pos) for pos in positions])
        
        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation
        
        # Generate multivariate normal returns
        asset_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
        
        # Calculate portfolio returns
        weights = np.array([abs(pos.market_value) / portfolio_value for pos in positions])
        portfolio_returns = np.dot(asset_returns, weights) * portfolio_value
        
        return portfolio_returns
    
    def _generate_correlation_matrix(self, positions: List[Position]) -> np.ndarray:
        """Generate correlation matrix for positions"""
        n_assets = len(positions)
        
        # Create realistic correlation structure
        correlation = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Higher correlation for same sector/asset class
                if (positions[i].sector == positions[j].sector and 
                    positions[i].sector is not None):
                    corr = np.random.uniform(0.6, 0.8)
                elif (positions[i].asset_class == positions[j].asset_class and
                      positions[i].asset_class is not None):
                    corr = np.random.uniform(0.3, 0.6)
                else:
                    corr = np.random.uniform(0.1, 0.4)
                
                correlation[i, j] = corr
                correlation[j, i] = corr
        
        return correlation
    
    def _estimate_volatility(self, position: Position) -> float:
        """Estimate volatility for a position"""
        # Simplified volatility estimation
        # In practice, use historical price data
        base_vol = 0.02  # 2% daily volatility
        
        # Adjust based on asset class
        if position.asset_class == 'crypto':
            return base_vol * 3
        elif position.asset_class == 'equity':
            return base_vol
        elif position.asset_class == 'bond':
            return base_vol * 0.3
        else:
            return base_vol
    
    def _calculate_exponential_weights(self, n_observations: int) -> np.ndarray:
        """Calculate exponential decay weights"""
        weights = np.array([self.decay_factor ** i for i in range(n_observations)])
        weights = weights[::-1]  # Most recent observations get highest weight
        return weights / weights.sum()
    
    def _weighted_percentile(self, data: np.ndarray, weights: np.ndarray, percentile: float) -> float:
        """Calculate weighted percentile"""
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        cumulative_weights = np.cumsum(sorted_weights)
        cumulative_weights /= cumulative_weights[-1]  # Normalize to [0, 1]
        
        # Find percentile
        percentile_normalized = percentile / 100.0
        idx = np.searchsorted(cumulative_weights, percentile_normalized)
        
        if idx == 0:
            return sorted_data[0]
        elif idx >= len(sorted_data):
            return sorted_data[-1]
        else:
            # Linear interpolation
            weight_diff = cumulative_weights[idx] - cumulative_weights[idx-1]
            if weight_diff > 0:
                alpha = (percentile_normalized - cumulative_weights[idx-1]) / weight_diff
                return sorted_data[idx-1] + alpha * (sorted_data[idx] - sorted_data[idx-1])
            else:
                return sorted_data[idx]

class MonteCarloModel(RiskModel):
    """Monte Carlo simulation risk model with advanced features"""
    
    def __init__(self, num_simulations: int = 10000, random_seed: int = 42):
        self.num_simulations = num_simulations
        self.random_seed = random_seed
        
    def calculate_var(self, positions: List[Position], 
                     confidence_level: float = 0.05,
                     time_horizon: int = 1) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        if not positions:
            return 0.0
        
        simulated_returns = self._run_monte_carlo(positions, time_horizon)
        var_percentile = np.percentile(simulated_returns, confidence_level * 100)
        
        return abs(var_percentile)
    
    def calculate_expected_shortfall(self, positions: List[Position],
                                   confidence_level: float = 0.05,
                                   time_horizon: int = 1) -> float:
        """Calculate Expected Shortfall using Monte Carlo"""
        if not positions:
            return 0.0
        
        simulated_returns = self._run_monte_carlo(positions, time_horizon)
        var_threshold = np.percentile(simulated_returns, confidence_level * 100)
        
        tail_returns = simulated_returns[simulated_returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return abs(var_threshold)
        
        return abs(np.mean(tail_returns))
    
    def calculate_component_var(self, positions: List[Position],
                              confidence_level: float = 0.05) -> Dict[str, float]:
        """Calculate component VaR using Monte Carlo"""
        # Similar to historical simulation but using Monte Carlo
        # Implementation would be similar to HistoricalSimulationModel
        return {}
    
    def _run_monte_carlo(self, positions: List[Position], time_horizon: int) -> np.ndarray:
        """Run Monte Carlo simulations"""
        np.random.seed(self.random_seed)
        
        portfolio_value = sum(abs(pos.market_value) for pos in positions)
        if portfolio_value == 0:
            return np.array([])
        
        # Simplified Monte Carlo for demonstration
        portfolio_volatility = 0.15  # 15% annual volatility
        daily_volatility = portfolio_volatility / np.sqrt(252)
        
        # Scale for time horizon
        scaled_volatility = daily_volatility * np.sqrt(time_horizon)
        
        # Generate random returns
        random_returns = np.random.normal(0, scaled_volatility, self.num_simulations)
        
        # Convert to dollar returns
        dollar_returns = random_returns * portfolio_value
        
        return dollar_returns

class RealTimeRiskEngine:
    """Comprehensive real-time risk monitoring and control engine"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.risk_limits = {}
        self.risk_models = {
            'historical': HistoricalSimulationModel(),
            'monte_carlo': MonteCarloModel()
        }
        self.positions = {}
        self.risk_metrics = {}
        self.alerts = []
        self.circuit_breakers = {}
        self.performance_history = []
        
        # Setup default configuration
        self._setup_default_limits()
        self._setup_circuit_breakers()
        
    def _setup_default_limits(self):
        """Setup comprehensive default risk limits"""
        self.risk_limits = {
            'portfolio_var_daily': RiskLimit(
                name='Portfolio VaR Daily',
                limit_type=LimitType.VAR,
                value=50000,  # $50k daily VaR limit
                scope='portfolio',
                time_horizon='daily'
            ),
            'portfolio_var_intraday': RiskLimit(
                name='Portfolio VaR Intraday',
                limit_type=LimitType.VAR,
                value=25000,  # $25k intraday VaR limit
                scope='portfolio',
                time_horizon='intraday'
            ),
            'portfolio_notional': RiskLimit(
                name='Portfolio Gross Notional',
                limit_type=LimitType.NOTIONAL,
                value=1000000,  # $1M gross notional limit
                scope='portfolio',
                time_horizon='intraday'
            ),
            'portfolio_net_notional': RiskLimit(
                name='Portfolio Net Notional',
                limit_type=LimitType.NOTIONAL,
                value=500000,  # $500k net notional limit
                scope='portfolio',
                time_horizon='intraday'
            ),
            'max_position_size': RiskLimit(
                name='Maximum Position Size',
                limit_type=LimitType.NOTIONAL,
                value=100000,  # $100k per position
                scope='symbol',
                time_horizon='intraday'
            ),
            'max_drawdown': RiskLimit(
                name='Maximum Drawdown',
                limit_type=LimitType.DRAWDOWN,
                value=0.1,  # 10% maximum drawdown
                scope='portfolio',
                time_horizon='daily'
            ),
            'max_leverage': RiskLimit(
                name='Maximum Leverage',
                limit_type=LimitType.LEVERAGE,
                value=3.0,  # 3:1 maximum leverage
                scope='portfolio',
                time_horizon='intraday'
            ),
            'concentration_limit': RiskLimit(
                name='Position Concentration',
                limit_type=LimitType.CONCENTRATION,
                value=0.25,  # 25% max in single position
                scope='portfolio',
                time_horizon='intraday'
            )
        }
    
    def _setup_circuit_breakers(self):
        """Setup automated circuit breakers"""
        self.circuit_breakers = {
            'daily_loss_limit': {
                'threshold': -25000,  # $25k daily loss limit
                'action': 'close_all_positions',
                'enabled': True
            },
            'var_breach': {
                'threshold': 1.5,  # 150% of VaR limit
                'action': 'halt_new_orders',
                'enabled': True
            },
            'concentration_breach': {
                'threshold': 0.3,  # 30% concentration
                'action': 'reduce_positions',
                'enabled': True
            }
        }
    
    def add_risk_limit(self, limit: RiskLimit):
        """Add or update a risk limit"""
        self.risk_limits[limit.name] = limit
        logger.info(f"Added risk limit: {limit.name} = {limit.value}")
        
        # Store in Redis
        self._store_risk_limits()
    
    def update_positions(self, positions: List[Position]):
        """Update current positions for risk monitoring"""
        self.positions = {pos.symbol: pos for pos in positions}
        
        # Store in Redis for persistence and cross-process access
        positions_data = {
            pos.symbol: asdict(pos) for pos in positions
        }
        
        self.redis_client.setex(
            'risk_engine:positions',
            timedelta(hours=1).total_seconds(),
            json.dumps(positions_data, default=str)
        )
        
        logger.debug(f"Updated {len(positions)} positions")
    
    async def check_pre_trade_risk(self, symbol: str, quantity: float, 
                                 price: float, strategy_id: str = None) -> Tuple[bool, List[str]]:
        """Comprehensive pre-trade risk check"""
        violations = []
        
        # Calculate proposed trade impact
        proposed_value = abs(quantity * price)
        
        # Check individual position limits
        current_position = self.positions.get(symbol)
        if current_position:
            new_position_value = abs(current_position.market_value + (quantity * price))
        else:
            new_position_value = proposed_value
        
        max_position_limit = self.risk_limits.get('max_position_size')
        if max_position_limit and max_position_limit.enabled:
            if new_position_value > max_position_limit.value:
                violations.append(f"Position size limit exceeded for {symbol}: "
                                f"${new_position_value:,.0f} > ${max_position_limit.value:,.0f}")
        
        # Check portfolio notional limits
        current_gross_notional = sum(abs(pos.market_value) for pos in self.positions.values())
        new_gross_notional = current_gross_notional + proposed_value
        
        gross_notional_limit = self.risk_limits.get('portfolio_notional')
        if gross_notional_limit and gross_notional_limit.enabled:
            if new_gross_notional > gross_notional_limit.value:
                violations.append(f"Portfolio gross notional limit exceeded: "
                                f"${new_gross_notional:,.0f} > ${gross_notional_limit.value:,.0f}")
        
        # Check concentration limits
        if new_gross_notional > 0:
            new_concentration = new_position_value / new_gross_notional
            concentration_limit = self.risk_limits.get('concentration_limit')
            if concentration_limit and concentration_limit.enabled:
                if new_concentration > concentration_limit.value:
                    violations.append(f"Position concentration limit exceeded for {symbol}: "
                                    f"{new_concentration:.1%} > {concentration_limit.value:.1%}")
        
        # Check VaR impact
        if len(self.positions) > 0:
            current_var = await self._calculate_portfolio_var()
            
            # Estimate VaR impact (simplified)
            position_volatility = 0.02  # Assume 2% daily volatility
            var_impact = proposed_value * position_volatility * 1.645  # 95% confidence
            estimated_new_var = current_var + var_impact
            
            var_limit = self.risk_limits.get('portfolio_var_daily')
            if var_limit and var_limit.enabled:
                if estimated_new_var > var_limit.value:
                    violations.append(f"Portfolio VaR limit would be exceeded: "
                                    f"${estimated_new_var:,.0f} > ${var_limit.value:,.0f}")
        
        # Check circuit breakers
        for breaker_name, breaker_config in self.circuit_breakers.items():
            if not breaker_config['enabled']:
                continue
                
            if breaker_name == 'var_breach' and len(self.positions) > 0:
                current_var = await self._calculate_portfolio_var()
                var_limit = self.risk_limits.get('portfolio_var_daily')
                if var_limit and current_var > var_limit.value * breaker_config['threshold']:
                    violations.append(f"Circuit breaker triggered: {breaker_name}")
        
        # Return approval status and violations
        approved = len(violations) == 0
        
        # Log pre-trade check
        logger.info(f"Pre-trade check for {symbol}: {quantity} @ ${price:.2f} - "
                   f"Approved: {approved}, Violations: {len(violations)}")
        
        return approved, violations
    
    async def calculate_real_time_risk(self) -> Dict[str, RiskMetric]:
        """Calculate comprehensive real-time risk metrics"""
        if not self.positions:
            return {}
        
        positions_list = list(self.positions.values())
        metrics = {}
        
        try:
            # Portfolio VaR calculations
            var_daily = await self._calculate_portfolio_var(time_horizon=1)
            var_intraday = await self._calculate_portfolio_var(time_horizon=0.25)  # 6 hours
            
            # VaR metrics
            for var_type, var_value, limit_key in [
                ('daily', var_daily, 'portfolio_var_daily'),
                ('intraday', var_intraday, 'portfolio_var_intraday')
            ]:
                var_limit = self.risk_limits.get(limit_key)
                if var_limit and var_limit.enabled:
                    metrics[f'portfolio_var_{var_type}'] = RiskMetric(
                        name=f'Portfolio VaR {var_type.title()}',
                        value=var_value,
                        limit=var_limit.value,
                        utilization=var_value / var_limit.value,
                        status=self._determine_risk_status(var_value, var_limit.value),
                        timestamp=datetime.now(),
                        details={'time_horizon': var_type}
                    )
            
            # Notional exposure metrics
            gross_notional = sum(abs(pos.market_value) for pos in positions_list)
            net_notional = sum(pos.market_value for pos in positions_list)
            
            for notional_type, notional_value, limit_key in [
                ('gross', gross_notional, 'portfolio_notional'),
                ('net', abs(net_notional), 'portfolio_net_notional')
            ]:
                notional_limit = self.risk_limits.get(limit_key)
                if notional_limit and notional_limit.enabled:
                    metrics[f'portfolio_{notional_type}_notional'] = RiskMetric(
                        name=f'Portfolio {notional_type.title()} Notional',
                        value=notional_value,
                        limit=notional_limit.value,
                        utilization=notional_value / notional_limit.value,
                        status=self._determine_risk_status(notional_value, notional_limit.value),
                        timestamp=datetime.now(),
                        details={'exposure_type': notional_type}
                    )
            
            # Drawdown calculation
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions_list)
            total_cost_basis = sum(abs(pos.cost_basis) for pos in positions_list if pos.cost_basis != 0)
            
            if total_cost_basis > 0:
                current_drawdown = abs(min(0, total_unrealized_pnl / total_cost_basis))
                drawdown_limit = self.risk_limits.get('max_drawdown')
                if drawdown_limit and drawdown_limit.enabled:
                    metrics['max_drawdown'] = RiskMetric(
                        name='Maximum Drawdown',
                        value=current_drawdown,
                        limit=drawdown_limit.value,
                        utilization=current_drawdown / drawdown_limit.value,
                        status=self._determine_risk_status(current_drawdown, drawdown_limit.value),
                        timestamp=datetime.now(),
                        details={'unrealized_pnl': total_unrealized_pnl, 'cost_basis': total_cost_basis}
                    )
            
            # Concentration risk
            concentration_risk = self._calculate_concentration_risk(positions_list)
            concentration_limit = self.risk_limits.get('concentration_limit')
            if concentration_limit and concentration_limit.enabled:
                metrics['concentration_risk'] = RiskMetric(
                    name='Position Concentration',
                    value=concentration_risk,
                    limit=concentration_limit.value,
                    utilization=concentration_risk / concentration_limit.value,
                    status=self._determine_risk_status(concentration_risk, concentration_limit.value),
                    timestamp=datetime.now(),
                    details={'herfindahl_index': concentration_risk}
                )
            
            # Leverage calculation
            if total_cost_basis > 0:
                leverage = gross_notional / total_cost_basis
                leverage_limit = self.risk_limits.get('max_leverage')
                if leverage_limit and leverage_limit.enabled:
                    metrics['leverage'] = RiskMetric(
                        name='Portfolio Leverage',
                        value=leverage,
                        limit=leverage_limit.value,
                        utilization=leverage / leverage_limit.value,
                        status=self._determine_risk_status(leverage, leverage_limit.value),
                        timestamp=datetime.now(),
                        details={'gross_notional': gross_notional, 'cost_basis': total_cost_basis}
                    )
            
            # Expected Shortfall
            expected_shortfall = await self._calculate_expected_shortfall()
            if expected_shortfall > 0:
                metrics['expected_shortfall'] = RiskMetric(
                    name='Expected Shortfall',
                    value=expected_shortfall,
                    limit=var_daily * 1.3,  # ES typically 30% higher than VaR
                    utilization=expected_shortfall / (var_daily * 1.3) if var_daily > 0 else 0,
                    status=RiskStatus.OK if expected_shortfall <= var_daily * 1.3 else RiskStatus.WARNING,
                    timestamp=datetime.now(),
                    details={'confidence_level': 0.05}
                )
            
            self.risk_metrics = metrics
            
            # Store metrics in Redis
            await self._store_risk_metrics(metrics)
            
            # Check for alerts
            await self._check_risk_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
        
        return metrics
    
    def _determine_risk_status(self, current_value: float, limit_value: float) -> RiskStatus:
        """Determine risk status based on utilization"""
        if limit_value <= 0:
            return RiskStatus.OK
        
        utilization = current_value / limit_value
        
        if utilization >= 1.0:
            return RiskStatus.BREACH
        elif utilization >= 0.9:
            return RiskStatus.CRITICAL
        elif utilization >= 0.8:
            return RiskStatus.WARNING
        else:
            return RiskStatus.OK
    
    async def _calculate_portfolio_var(self, confidence_level: float = 0.05, 
                                     time_horizon: float = 1) -> float:
        """Calculate portfolio VaR using configured risk model"""
        positions_list = list(self.positions.values())
        
        # Use historical simulation model by default
        risk_model = self.risk_models['historical']
        var = risk_model.calculate_var(positions_list, confidence_level, int(time_horizon))
        
        return var
    
    async def _calculate_expected_shortfall(self, confidence_level: float = 0.05) -> float:
        """Calculate portfolio Expected Shortfall"""
        positions_list = list(self.positions.values())
        
        risk_model = self.risk_models['historical']
        es = risk_model.calculate_expected_shortfall(positions_list, confidence_level)
        
        return es
    
    def _calculate_concentration_risk(self, positions: List[Position]) -> float:
        """Calculate portfolio concentration risk using Herfindahl index"""
        if not positions:
            return 0.0
        
        total_value = sum(abs(pos.market_value) for pos in positions)
        if total_value == 0:
            return 0.0
        
        # Calculate Herfindahl index
        weights = [abs(pos.market_value) / total_value for pos in positions]
        herfindahl = sum(w**2 for w in weights)
        
        return herfindahl
    
    async def _check_risk_alerts(self, metrics: Dict[str, RiskMetric]):
        """Check for risk alerts and generate notifications"""
        for metric_name, metric in metrics.items():
            if metric.status in [RiskStatus.WARNING, RiskStatus.BREACH, RiskStatus.CRITICAL]:
                alert = RiskAlert(
                    alert_id=f"{metric_name}_{int(datetime.now().timestamp())}",
                    timestamp=datetime.now(),
                    severity=metric.status,
                    metric_name=metric_name,
                    current_value=metric.value,
                    limit_value=metric.limit,
                    message=f"{metric.name} {metric.status.value}: {metric.value:.2f} "
                           f"({metric.utilization:.1%} of limit {metric.limit:.2f})",
                    positions_affected=[pos.symbol for pos in self.positions.values()],
                    recommended_action=self._get_recommended_action(metric)
                )
                
                self.alerts.append(alert)
                
                # Keep only recent alerts
                if len(self.alerts) > 1000:
                    self.alerts = self.alerts[-1000:]
                
                logger.warning(f"Risk alert: {alert.message}")
                
                # Store alert in Redis
                await self._store_alert(alert)
    
    def _get_recommended_action(self, metric: RiskMetric) -> str:
        """Get recommended action for risk metric"""
        if metric.status == RiskStatus.BREACH:
            if 'var' in metric.name.lower():
                return "Reduce position sizes or hedge exposure"
            elif 'notional' in metric.name.lower():
                return "Close positions to reduce gross exposure"
            elif 'concentration' in metric.name.lower():
                return "Diversify holdings across more positions"
            elif 'drawdown' in metric.name.lower():
                return "Consider defensive positioning or stop losses"
            elif 'leverage' in metric.name.lower():
                return "Reduce leverage by closing leveraged positions"
        elif metric.status == RiskStatus.CRITICAL:
            return "Monitor closely and prepare to reduce risk"
        elif metric.status == RiskStatus.WARNING:
            return "Review position sizing and risk allocation"
        
        return "Continue monitoring"
    
    async def _store_risk_metrics(self, metrics: Dict[str, RiskMetric]):
        """Store risk metrics in Redis"""
        try:
            metrics_data = {
                name: {
                    'value': metric.value,
                    'limit': metric.limit,
                    'utilization': metric.utilization,
                    'status': metric.status.value,
                    'timestamp': metric.timestamp.isoformat(),
                    'details': metric.details
                }
                for name, metric in metrics.items()
            }
            
            self.redis_client.setex(
                'risk_engine:metrics',
                timedelta(minutes=10).total_seconds(),
                json.dumps(metrics_data, default=str)
            )
        except Exception as e:
            logger.error(f"Error storing risk metrics: {e}")
    
    async def _store_alert(self, alert: RiskAlert):
        """Store risk alert in Redis"""
        try:
            alert_data = asdict(alert)
            alert_data['timestamp'] = alert.timestamp.isoformat()
            
            self.redis_client.lpush('risk_engine:alerts', json.dumps(alert_data, default=str))
            self.redis_client.ltrim('risk_engine:alerts', 0, 999)  # Keep last 1000 alerts
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    def _store_risk_limits(self):
        """Store risk limits in Redis"""
        try:
            limits_data = {
                name: asdict(limit) for name, limit in self.risk_limits.items()
            }
            
            self.redis_client.set(
                'risk_engine:limits',
                json.dumps(limits_data, default=str)
            )
        except Exception as e:
            logger.error(f"Error storing risk limits: {e}")
    
    async def monitor_risk_continuously(self, interval_seconds: int = 30):
        """Continuously monitor risk metrics"""
        logger.info(f"Starting continuous risk monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                # Calculate current risk metrics
                metrics = await self.calculate_real_time_risk()
                
                # Update performance history
                if metrics:
                    portfolio_value = sum(abs(pos.market_value) for pos in self.positions.values())
                    self.performance_history.append({
                        'timestamp': datetime.now(),
                        'portfolio_value': portfolio_value,
                        'var_daily': metrics.get('portfolio_var_daily', {}).get('value', 0),
                        'gross_notional': metrics.get('portfolio_gross_notional', {}).get('value', 0)
                    })
                    
                    # Keep only recent history
                    if len(self.performance_history) > 1440:  # 24 hours at 1-minute intervals
                        self.performance_history = self.performance_history[-1440:]
                
                # Sleep until next check
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(interval_seconds)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'positions_count': len(self.positions),
            'risk_metrics': {
                name: {
                    'value': metric.value,
                    'limit': metric.limit,
                    'utilization': metric.utilization,
                    'status': metric.status.value
                }
                for name, metric in self.risk_metrics.items()
            },
            'recent_alerts': [asdict(alert) for alert in self.alerts[-10:]],
            'risk_limits': {
                name: asdict(limit) for name, limit in self.risk_limits.items()
            },
            'circuit_breakers': self.circuit_breakers,
            'performance_summary': {
                'total_positions': len(self.positions),
                'gross_exposure': sum(abs(pos.market_value) for pos in self.positions.values()),
                'net_exposure': sum(pos.market_value for pos in self.positions.values()),
                'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
            }
        }

# Example usage and testing
async def main():
    """Test the comprehensive risk engine"""
    
    # Create risk engine
    risk_engine = RealTimeRiskEngine()
    
    # Create sample positions with enhanced data
    positions = [
        Position(
            symbol='AAPL',
            quantity=1000,
            market_value=150000,
            unrealized_pnl=5000,
            cost_basis=145000,
            side='LONG',
            strategy_id='momentum_001',
            sector='Technology',
            asset_class='equity'
        ),
        Position(
            symbol='GOOGL',
            quantity=100,
            market_value=250000,
            unrealized_pnl=-10000,
            cost_basis=260000,
            side='LONG',
            strategy_id='momentum_001',
            sector='Technology',
            asset_class='equity'
        ),
        Position(
            symbol='TSLA',
            quantity=-500,
            market_value=-100000,
            unrealized_pnl=8000,
            cost_basis=-108000,
            side='SHORT',
            strategy_id='pairs_001',
            sector='Automotive',
            asset_class='equity'
        ),
        Position(
            symbol='BTC-USD',
            quantity=2,
            market_value=100000,
            unrealized_pnl=15000,
            cost_basis=85000,
            side='LONG',
            strategy_id='crypto_001',
            sector='Cryptocurrency',
            asset_class='crypto'
        )
    ]
    
    # Update positions
    risk_engine.update_positions(positions)
    
    print("Comprehensive Risk Engine Test")
    print("=" * 50)
    
    # Test pre-trade risk check
    approved, violations = await risk_engine.check_pre_trade_risk('NVDA', 1000, 500, 'momentum_002')
    print(f"\nPre-trade check for NVDA (1000 @ $500):")
    print(f"Approved: {approved}")
    if violations:
        print("Violations:")
        for violation in violations:
            print(f"  - {violation}")
    
    # Calculate comprehensive risk metrics
    print("\nCalculating comprehensive risk metrics...")
    metrics = await risk_engine.calculate_real_time_risk()
    
    print(f"\nRisk Metrics ({len(metrics)} total):")
    for name, metric in metrics.items():
        status_color = {
            RiskStatus.OK: "✅",
            RiskStatus.WARNING: "⚠️",
            RiskStatus.CRITICAL: "🔶",
            RiskStatus.BREACH: "🚨"
        }.get(metric.status, "❓")
        
        print(f"{status_color} {metric.name}: {metric.value:,.2f} / {metric.limit:,.2f} "
              f"({metric.utilization:.1%}) - {metric.status.value}")
    
    # Get comprehensive risk summary
    summary = risk_engine.get_risk_summary()
    print(f"\nRisk Summary:")
    print(f"Positions: {summary['positions_count']}")
    print(f"Gross Exposure: ${summary['performance_summary']['gross_exposure']:,.2f}")
    print(f"Net Exposure: ${summary['performance_summary']['net_exposure']:,.2f}")
    print(f"Unrealized P&L: ${summary['performance_summary']['unrealized_pnl']:,.2f}")
    print(f"Active Metrics: {len(summary['risk_metrics'])}")
    print(f"Recent Alerts: {len(summary['recent_alerts'])}")
    
    # Test circuit breakers
    print(f"\nCircuit Breakers:")
    for name, config in summary['circuit_breakers'].items():
        status = "🟢 ENABLED" if config['enabled'] else "🔴 DISABLED"
        print(f"{status} {name}: {config['action']} at {config['threshold']}")

if __name__ == "__main__":
    asyncio.run(main())

