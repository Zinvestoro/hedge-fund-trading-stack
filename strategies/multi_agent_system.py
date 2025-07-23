#!/usr/bin/env python3
"""
Multi-Agent Trading System Implementation
Coordinates multiple AI agents for sophisticated trading decisions
"""

import asyncio
import json
import logging
import redis
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure for agent analysis"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    volatility: float
    trend_strength: float
    rsi: float
    macd: float

@dataclass
class TradingSignal:
    """Trading signal structure"""
    agent_id: str
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    quantity: int
    price_target: float
    stop_loss: float
    reasoning: str
    timestamp: datetime

@dataclass
class RiskAssessment:
    """Risk assessment structure"""
    agent_id: str
    portfolio_var: float
    max_drawdown: float
    position_concentration: float
    correlation_risk: float
    liquidity_risk: float
    recommendation: str
    timestamp: datetime

class BaseAgent(ABC):
    """Abstract base class for trading agents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.performance_history = []
        
    @abstractmethod
    async def analyze(self, market_data: List[MarketData]) -> Any:
        """Analyze market data and generate output"""
        pass
    
    def update_performance(self, performance_metric: float):
        """Update agent performance history"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'performance': performance_metric
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

class MarketAnalystAgent(BaseAgent):
    """Agent specialized in market analysis and signal generation"""
    
    def __init__(self, agent_id: str = "market_analyst", config: Dict[str, Any] = None):
        super().__init__(agent_id, config or {})
        self.technical_indicators = ['rsi', 'macd', 'bollinger_bands', 'moving_averages']
        self.sentiment_sources = ['news', 'social_media', 'options_flow']
        
    async def analyze(self, market_data: List[MarketData]) -> List[TradingSignal]:
        """Analyze market data and generate trading signals"""
        signals = []
        
        for data in market_data:
            signal = await self._generate_signal(data)
            if signal:
                signals.append(signal)
        
        self.logger.info(f"Generated {len(signals)} trading signals")
        return signals
    
    async def _generate_signal(self, data: MarketData) -> Optional[TradingSignal]:
        """Generate trading signal for a single symbol"""
        try:
            # Technical analysis
            technical_score = self._calculate_technical_score(data)
            
            # Momentum analysis
            momentum_score = self._calculate_momentum_score(data)
            
            # Volatility analysis
            volatility_score = self._calculate_volatility_score(data)
            
            # Combine scores
            combined_score = (
                0.4 * technical_score +
                0.4 * momentum_score +
                0.2 * volatility_score
            )
            
            # Generate signal based on combined score
            if combined_score > 0.6:
                action = 'buy'
                confidence = min(combined_score, 0.95)
            elif combined_score < -0.6:
                action = 'sell'
                confidence = min(abs(combined_score), 0.95)
            else:
                action = 'hold'
                confidence = 0.5
            
            # Calculate position sizing
            quantity = self._calculate_quantity(data, confidence)
            
            # Set price targets
            price_target = self._calculate_price_target(data, action)
            stop_loss = self._calculate_stop_loss(data, action)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(data, technical_score, momentum_score, volatility_score)
            
            return TradingSignal(
                agent_id=self.agent_id,
                symbol=data.symbol,
                action=action,
                confidence=confidence,
                quantity=quantity,
                price_target=price_target,
                stop_loss=stop_loss,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {data.symbol}: {e}")
            return None
    
    def _calculate_technical_score(self, data: MarketData) -> float:
        """Calculate technical analysis score"""
        score = 0.0
        
        # RSI analysis
        if data.rsi < 30:
            score += 0.3  # Oversold - bullish
        elif data.rsi > 70:
            score -= 0.3  # Overbought - bearish
        
        # MACD analysis
        if data.macd > 0:
            score += 0.2  # Bullish momentum
        else:
            score -= 0.2  # Bearish momentum
        
        # Trend strength
        score += data.trend_strength * 0.5
        
        return np.clip(score, -1.0, 1.0)
    
    def _calculate_momentum_score(self, data: MarketData) -> float:
        """Calculate momentum score"""
        # Simplified momentum calculation
        # In practice, this would use historical price data
        momentum = data.trend_strength
        
        if momentum > 0.5:
            return 0.8
        elif momentum < -0.5:
            return -0.8
        else:
            return momentum
    
    def _calculate_volatility_score(self, data: MarketData) -> float:
        """Calculate volatility-adjusted score"""
        # High volatility reduces confidence
        if data.volatility > 0.3:
            return -0.2
        elif data.volatility < 0.1:
            return 0.1
        else:
            return 0.0
    
    def _calculate_quantity(self, data: MarketData, confidence: float) -> int:
        """Calculate position quantity based on confidence"""
        base_quantity = 100
        return int(base_quantity * confidence)
    
    def _calculate_price_target(self, data: MarketData, action: str) -> float:
        """Calculate price target"""
        if action == 'buy':
            return data.price * 1.05  # 5% upside target
        elif action == 'sell':
            return data.price * 0.95  # 5% downside target
        else:
            return data.price
    
    def _calculate_stop_loss(self, data: MarketData, action: str) -> float:
        """Calculate stop loss level"""
        volatility_multiplier = max(2.0, data.volatility * 10)
        
        if action == 'buy':
            return data.price * (1 - 0.02 * volatility_multiplier)
        elif action == 'sell':
            return data.price * (1 + 0.02 * volatility_multiplier)
        else:
            return data.price
    
    def _generate_reasoning(self, data: MarketData, tech_score: float, 
                          momentum_score: float, vol_score: float) -> str:
        """Generate human-readable reasoning for the signal"""
        reasoning_parts = []
        
        if tech_score > 0.3:
            reasoning_parts.append("Strong technical indicators")
        elif tech_score < -0.3:
            reasoning_parts.append("Weak technical indicators")
        
        if momentum_score > 0.3:
            reasoning_parts.append("Positive momentum")
        elif momentum_score < -0.3:
            reasoning_parts.append("Negative momentum")
        
        if data.volatility > 0.3:
            reasoning_parts.append("High volatility environment")
        
        if data.rsi < 30:
            reasoning_parts.append("Oversold conditions")
        elif data.rsi > 70:
            reasoning_parts.append("Overbought conditions")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "Neutral market conditions"

class RiskManagerAgent(BaseAgent):
    """Agent specialized in risk management and portfolio protection"""
    
    def __init__(self, agent_id: str = "risk_manager", config: Dict[str, Any] = None):
        super().__init__(agent_id, config or {})
        self.max_portfolio_var = config.get('max_portfolio_var', 0.02)
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_correlation = config.get('max_correlation', 0.7)
        
    async def analyze(self, portfolio_data: Dict[str, Any]) -> RiskAssessment:
        """Analyze portfolio risk and generate assessment"""
        try:
            # Calculate portfolio VaR
            portfolio_var = self._calculate_portfolio_var(portfolio_data)
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio_data)
            
            # Check position concentration
            position_concentration = self._calculate_position_concentration(portfolio_data)
            
            # Assess correlation risk
            correlation_risk = self._calculate_correlation_risk(portfolio_data)
            
            # Assess liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(portfolio_data)
            
            # Generate recommendation
            recommendation = self._generate_risk_recommendation(
                portfolio_var, max_drawdown, position_concentration, 
                correlation_risk, liquidity_risk
            )
            
            return RiskAssessment(
                agent_id=self.agent_id,
                portfolio_var=portfolio_var,
                max_drawdown=max_drawdown,
                position_concentration=position_concentration,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                recommendation=recommendation,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in risk analysis: {e}")
            return RiskAssessment(
                agent_id=self.agent_id,
                portfolio_var=0.0,
                max_drawdown=0.0,
                position_concentration=0.0,
                correlation_risk=0.0,
                liquidity_risk=0.0,
                recommendation="Unable to assess risk",
                timestamp=datetime.now()
            )
    
    def _calculate_portfolio_var(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate portfolio Value at Risk"""
        # Simplified VaR calculation
        positions = portfolio_data.get('positions', {})
        if not positions:
            return 0.0
        
        # Mock calculation - in practice, use historical returns and correlations
        total_value = sum(pos.get('value', 0) for pos in positions.values())
        portfolio_volatility = 0.15  # Assume 15% annual volatility
        confidence_level = 0.05  # 95% confidence
        
        # Daily VaR
        daily_var = total_value * portfolio_volatility / np.sqrt(252) * 1.645
        return daily_var / total_value if total_value > 0 else 0.0
    
    def _calculate_max_drawdown(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate maximum drawdown"""
        portfolio_history = portfolio_data.get('value_history', [])
        if len(portfolio_history) < 2:
            return 0.0
        
        values = [entry['value'] for entry in portfolio_history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_position_concentration(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate position concentration risk"""
        positions = portfolio_data.get('positions', {})
        if not positions:
            return 0.0
        
        total_value = sum(pos.get('value', 0) for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        # Calculate Herfindahl index
        weights = [pos.get('value', 0) / total_value for pos in positions.values()]
        herfindahl = sum(w**2 for w in weights)
        
        return herfindahl
    
    def _calculate_correlation_risk(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate correlation risk between positions"""
        # Simplified correlation risk assessment
        positions = portfolio_data.get('positions', {})
        if len(positions) < 2:
            return 0.0
        
        # Mock correlation calculation
        # In practice, calculate actual correlations between assets
        return 0.6  # Assume moderate correlation
    
    def _calculate_liquidity_risk(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate liquidity risk"""
        positions = portfolio_data.get('positions', {})
        if not positions:
            return 0.0
        
        # Simplified liquidity assessment based on position sizes
        total_value = sum(pos.get('value', 0) for pos in positions.values())
        large_positions = sum(pos.get('value', 0) for pos in positions.values() 
                            if pos.get('value', 0) / total_value > 0.1)
        
        return large_positions / total_value if total_value > 0 else 0.0
    
    def _generate_risk_recommendation(self, portfolio_var: float, max_drawdown: float,
                                    position_concentration: float, correlation_risk: float,
                                    liquidity_risk: float) -> str:
        """Generate risk management recommendation"""
        recommendations = []
        
        if portfolio_var > self.max_portfolio_var:
            recommendations.append("Reduce portfolio risk - VaR exceeds limit")
        
        if max_drawdown > 0.1:
            recommendations.append("High drawdown detected - consider defensive positioning")
        
        if position_concentration > 0.3:
            recommendations.append("High position concentration - diversify holdings")
        
        if correlation_risk > self.max_correlation:
            recommendations.append("High correlation risk - reduce correlated positions")
        
        if liquidity_risk > 0.2:
            recommendations.append("Liquidity risk elevated - reduce large positions")
        
        if not recommendations:
            return "Risk levels acceptable - maintain current strategy"
        
        return "; ".join(recommendations)

class PortfolioManagerAgent(BaseAgent):
    """Agent responsible for portfolio optimization and allocation decisions"""
    
    def __init__(self, agent_id: str = "portfolio_manager", config: Dict[str, Any] = None):
        super().__init__(agent_id, config or {})
        self.target_volatility = config.get('target_volatility', 0.15)
        self.max_positions = config.get('max_positions', 20)
        
    async def analyze(self, signals: List[TradingSignal], 
                     risk_assessment: RiskAssessment) -> Dict[str, Any]:
        """Optimize portfolio based on signals and risk assessment"""
        try:
            # Filter and rank signals
            filtered_signals = self._filter_signals(signals, risk_assessment)
            
            # Optimize portfolio allocation
            allocation = self._optimize_allocation(filtered_signals, risk_assessment)
            
            # Generate portfolio recommendations
            recommendations = self._generate_recommendations(allocation, risk_assessment)
            
            return {
                'agent_id': self.agent_id,
                'allocation': allocation,
                'recommendations': recommendations,
                'filtered_signals': len(filtered_signals),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return {
                'agent_id': self.agent_id,
                'allocation': {},
                'recommendations': ["Error in portfolio optimization"],
                'filtered_signals': 0,
                'timestamp': datetime.now()
            }
    
    def _filter_signals(self, signals: List[TradingSignal], 
                       risk_assessment: RiskAssessment) -> List[TradingSignal]:
        """Filter signals based on risk constraints"""
        filtered = []
        
        for signal in signals:
            # Filter by confidence threshold
            if signal.confidence < 0.6:
                continue
            
            # Filter by risk assessment
            if risk_assessment.portfolio_var > 0.02 and signal.action == 'buy':
                continue  # Don't add risk in high-risk environment
            
            filtered.append(signal)
        
        # Sort by confidence and limit number
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        return filtered[:self.max_positions]
    
    def _optimize_allocation(self, signals: List[TradingSignal], 
                           risk_assessment: RiskAssessment) -> Dict[str, float]:
        """Optimize portfolio allocation weights"""
        if not signals:
            return {}
        
        allocation = {}
        total_confidence = sum(signal.confidence for signal in signals)
        
        for signal in signals:
            # Base weight from confidence
            base_weight = signal.confidence / total_confidence
            
            # Adjust for risk
            risk_adjustment = 1.0
            if risk_assessment.portfolio_var > 0.015:
                risk_adjustment = 0.5  # Reduce allocation in high-risk environment
            
            # Adjust for action type
            if signal.action == 'sell':
                base_weight *= -1  # Negative weight for short positions
            
            allocation[signal.symbol] = base_weight * risk_adjustment
        
        # Normalize weights
        total_weight = sum(abs(w) for w in allocation.values())
        if total_weight > 0:
            allocation = {k: v / total_weight for k, v in allocation.items()}
        
        return allocation
    
    def _generate_recommendations(self, allocation: Dict[str, float], 
                                risk_assessment: RiskAssessment) -> List[str]:
        """Generate portfolio management recommendations"""
        recommendations = []
        
        if not allocation:
            recommendations.append("No suitable trading opportunities identified")
            return recommendations
        
        # Allocation recommendations
        long_positions = {k: v for k, v in allocation.items() if v > 0}
        short_positions = {k: v for k, v in allocation.items() if v < 0}
        
        if long_positions:
            top_long = max(long_positions.items(), key=lambda x: x[1])
            recommendations.append(f"Largest long allocation: {top_long[0]} ({top_long[1]:.1%})")
        
        if short_positions:
            top_short = min(short_positions.items(), key=lambda x: x[1])
            recommendations.append(f"Largest short allocation: {top_short[0]} ({abs(top_short[1]):.1%})")
        
        # Risk-based recommendations
        if risk_assessment.portfolio_var > 0.02:
            recommendations.append("Consider reducing position sizes due to elevated risk")
        
        if risk_assessment.position_concentration > 0.3:
            recommendations.append("Increase diversification to reduce concentration risk")
        
        return recommendations

class MultiAgentOrchestrator:
    """Orchestrates multiple agents for collaborative trading decisions"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.agents = {}
        self.decision_history = []
        self._setup_agents()
        
    def _setup_agents(self):
        """Initialize all trading agents"""
        self.agents['analyst'] = MarketAnalystAgent()
        self.agents['risk_manager'] = RiskManagerAgent()
        self.agents['portfolio_manager'] = PortfolioManagerAgent()
        
        logger.info(f"Initialized {len(self.agents)} trading agents")
    
    async def execute_trading_cycle(self, market_data: List[MarketData], 
                                  portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete trading decision cycle"""
        cycle_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Step 1: Market Analysis
            logger.info("Starting market analysis...")
            signals = await self.agents['analyst'].analyze(market_data)
            
            # Step 2: Risk Assessment
            logger.info("Conducting risk assessment...")
            risk_assessment = await self.agents['risk_manager'].analyze(portfolio_data)
            
            # Step 3: Portfolio Optimization
            logger.info("Optimizing portfolio allocation...")
            portfolio_decision = await self.agents['portfolio_manager'].analyze(
                signals, risk_assessment
            )
            
            # Compile results
            decision = {
                'cycle_id': cycle_id,
                'timestamp': start_time,
                'market_data_points': len(market_data),
                'signals_generated': len(signals),
                'risk_assessment': asdict(risk_assessment),
                'portfolio_decision': portfolio_decision,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            
            # Store decision in Redis
            await self._store_decision(decision)
            
            # Update decision history
            self.decision_history.append(decision)
            if len(self.decision_history) > 100:
                self.decision_history = self.decision_history[-100:]
            
            logger.info(f"Trading cycle completed in {decision['execution_time']:.2f} seconds")
            return decision
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return {
                'cycle_id': cycle_id,
                'timestamp': start_time,
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def _store_decision(self, decision: Dict[str, Any]):
        """Store trading decision in Redis"""
        try:
            key = f"trading_decision:{decision['cycle_id']}"
            self.redis_client.setex(
                key, 
                timedelta(days=7).total_seconds(),  # 7-day expiration
                json.dumps(decision, default=str)
            )
        except Exception as e:
            logger.error(f"Error storing decision in Redis: {e}")
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all agents"""
        performance = {}
        
        for agent_id, agent in self.agents.items():
            if agent.performance_history:
                recent_performance = agent.performance_history[-10:]
                avg_performance = np.mean([p['performance'] for p in recent_performance])
                performance[agent_id] = {
                    'average_performance': avg_performance,
                    'total_decisions': len(agent.performance_history),
                    'last_updated': recent_performance[-1]['timestamp'] if recent_performance else None
                }
            else:
                performance[agent_id] = {
                    'average_performance': 0.0,
                    'total_decisions': 0,
                    'last_updated': None
                }
        
        return performance
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trading decisions"""
        return self.decision_history[-limit:] if self.decision_history else []

# Example usage and testing
async def main():
    """Test the multi-agent trading system"""
    
    # Create sample market data
    market_data = [
        MarketData(
            timestamp=datetime.now(),
            symbol='AAPL',
            price=150.0,
            volume=1000000,
            bid=149.95,
            ask=150.05,
            volatility=0.2,
            trend_strength=0.6,
            rsi=65.0,
            macd=0.5
        ),
        MarketData(
            timestamp=datetime.now(),
            symbol='GOOGL',
            price=2500.0,
            volume=500000,
            bid=2499.0,
            ask=2501.0,
            volatility=0.25,
            trend_strength=-0.3,
            rsi=45.0,
            macd=-0.2
        )
    ]
    
    # Create sample portfolio data
    portfolio_data = {
        'positions': {
            'AAPL': {'value': 15000, 'shares': 100},
            'GOOGL': {'value': 25000, 'shares': 10}
        },
        'value_history': [
            {'timestamp': datetime.now() - timedelta(days=1), 'value': 39000},
            {'timestamp': datetime.now(), 'value': 40000}
        ]
    }
    
    # Initialize orchestrator
    try:
        orchestrator = MultiAgentOrchestrator()
        
        # Execute trading cycle
        decision = await orchestrator.execute_trading_cycle(market_data, portfolio_data)
        
        print("Multi-Agent Trading Decision:")
        print(f"Cycle ID: {decision['cycle_id']}")
        print(f"Execution Time: {decision['execution_time']:.2f} seconds")
        print(f"Signals Generated: {decision['signals_generated']}")
        
        if 'risk_assessment' in decision:
            risk = decision['risk_assessment']
            print(f"Portfolio VaR: {risk['portfolio_var']:.2%}")
            print(f"Risk Recommendation: {risk['recommendation']}")
        
        if 'portfolio_decision' in decision:
            portfolio = decision['portfolio_decision']
            print(f"Portfolio Recommendations: {portfolio['recommendations']}")
        
        # Get agent performance
        performance = orchestrator.get_agent_performance()
        print("\nAgent Performance:")
        for agent_id, perf in performance.items():
            print(f"{agent_id}: {perf['total_decisions']} decisions")
        
    except Exception as e:
        print(f"Error running multi-agent system: {e}")
        print("Note: This example requires Redis to be running")

if __name__ == "__main__":
    asyncio.run(main())

