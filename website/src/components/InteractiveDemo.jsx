import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar
} from 'recharts'
import { 
  Play, 
  Pause, 
  RotateCcw, 
  TrendingUp, 
  TrendingDown,
  Activity,
  DollarSign,
  Zap,
  Shield
} from 'lucide-react'

const InteractiveDemo = () => {
  const [isRunning, setIsRunning] = useState(false)
  const [currentPrice, setCurrentPrice] = useState(150.25)
  const [portfolioValue, setPortfolioValue] = useState(1000000)
  const [pnl, setPnl] = useState(25000)
  const [varValue, setVarValue] = useState(35000)

  // Sample data for charts
  const [priceData, setPriceData] = useState([
    { time: '09:30', price: 148.50, volume: 1200 },
    { time: '09:45', price: 149.20, volume: 1800 },
    { time: '10:00', price: 150.25, volume: 2100 },
    { time: '10:15', price: 149.80, volume: 1600 },
    { time: '10:30', price: 151.10, volume: 2400 },
    { time: '10:45', price: 150.75, volume: 1900 }
  ])

  const performanceData = [
    { date: 'Jan', returns: 2.5, benchmark: 1.8 },
    { date: 'Feb', returns: 3.2, benchmark: 2.1 },
    { date: 'Mar', returns: -1.1, benchmark: -0.8 },
    { date: 'Apr', returns: 4.8, benchmark: 3.2 },
    { date: 'May', returns: 2.9, benchmark: 2.4 },
    { date: 'Jun', returns: 5.1, benchmark: 3.8 }
  ]

  const riskMetrics = [
    { metric: 'Daily VaR', value: 35000, limit: 50000 },
    { metric: 'Position Limit', value: 75000, limit: 100000 },
    { metric: 'Leverage', value: 2.1, limit: 3.0 },
    { metric: 'Concentration', value: 18, limit: 25 }
  ]

  // Simulate real-time updates
  useEffect(() => {
    let interval
    if (isRunning) {
      interval = setInterval(() => {
        const change = (Math.random() - 0.5) * 2
        setCurrentPrice(prev => Math.max(140, Math.min(160, prev + change)))
        setPnl(prev => prev + (Math.random() - 0.5) * 1000)
        setPortfolioValue(prev => prev + (Math.random() - 0.5) * 5000)
        setVarValue(prev => Math.max(20000, Math.min(45000, prev + (Math.random() - 0.5) * 2000)))
        
        // Update price data
        setPriceData(prev => {
          const newData = [...prev]
          const lastTime = new Date()
          const timeStr = lastTime.toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit' 
          })
          
          if (newData.length >= 10) {
            newData.shift()
          }
          
          newData.push({
            time: timeStr,
            price: currentPrice,
            volume: Math.floor(Math.random() * 3000) + 1000
          })
          
          return newData
        })
      }, 2000)
    }
    
    return () => clearInterval(interval)
  }, [isRunning, currentPrice])

  const codeExamples = {
    strategy: `# Momentum Strategy Implementation
from strategies.base_strategy import BaseStrategy
import numpy as np
import pandas as pd

class MomentumStrategy(BaseStrategy):
    def __init__(self, fast_period=12, slow_period=26):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def on_bar(self, bar):
        # Calculate EMAs
        fast_ema = self.calculate_ema(bar.close, self.fast_period)
        slow_ema = self.calculate_ema(bar.close, self.slow_period)
        
        # Generate signals
        if fast_ema > slow_ema and not self.position:
            self.buy(bar.symbol, quantity=100)
        elif fast_ema < slow_ema and self.position:
            self.sell(bar.symbol, quantity=self.position.quantity)`,
    
    risk: `# Risk Engine Configuration
class RiskEngine:
    def __init__(self):
        self.max_var = 50000  # Daily VaR limit
        self.max_position = 100000  # Per position limit
        self.max_leverage = 3.0  # Maximum leverage
        
    def check_risk(self, order):
        # Pre-trade risk checks
        if self.calculate_var() > self.max_var:
            return False, "VaR limit exceeded"
            
        if order.quantity * order.price > self.max_position:
            return False, "Position limit exceeded"
            
        return True, "Risk check passed"`,
    
    monitoring: `# Custom Metrics Export
from prometheus_client import Gauge, Counter, Histogram

# Define custom metrics
portfolio_value = Gauge('trading_portfolio_value_usd', 
                       'Portfolio value in USD')
order_latency = Histogram('trading_order_latency_seconds',
                         'Order execution latency')
trades_total = Counter('trading_trades_total',
                      'Total number of trades')

# Update metrics
portfolio_value.set(1000000)
order_latency.observe(0.0012)  # 1.2ms
trades_total.inc()`
  }

  return (
    <section className="py-16 bg-accent/5">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <Badge variant="secondary" className="mb-4">
            Interactive Demo
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Live Trading
            </span>{' '}
            Dashboard
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Experience the trading stack in action with real-time market simulation, 
            performance monitoring, and risk management controls.
          </p>
        </div>

        {/* Demo Controls */}
        <div className="flex justify-center mb-8">
          <div className="flex items-center space-x-4">
            <Button
              onClick={() => setIsRunning(!isRunning)}
              className={`${isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'} text-white`}
            >
              {isRunning ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
              {isRunning ? 'Pause Simulation' : 'Start Simulation'}
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                setCurrentPrice(150.25)
                setPnl(25000)
                setPortfolioValue(1000000)
                setVarValue(35000)
              }}
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset
            </Button>
          </div>
        </div>

        {/* Real-time Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <Card className="border-2 border-blue-200 dark:border-blue-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center space-x-2">
                <DollarSign className="w-4 h-4 text-blue-600" />
                <span>Portfolio Value</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">
                ${portfolioValue.toLocaleString()}
              </div>
              <div className="text-xs text-muted-foreground">
                {isRunning ? 'Live' : 'Simulated'}
              </div>
            </CardContent>
          </Card>

          <Card className="border-2 border-green-200 dark:border-green-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center space-x-2">
                <TrendingUp className="w-4 h-4 text-green-600" />
                <span>P&L</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {pnl >= 0 ? '+' : ''}${pnl.toLocaleString()}
              </div>
              <div className="text-xs text-muted-foreground">
                Unrealized
              </div>
            </CardContent>
          </Card>

          <Card className="border-2 border-purple-200 dark:border-purple-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center space-x-2">
                <Activity className="w-4 h-4 text-purple-600" />
                <span>Current Price</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-600">
                ${currentPrice.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground">
                AAPL
              </div>
            </CardContent>
          </Card>

          <Card className="border-2 border-orange-200 dark:border-orange-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center space-x-2">
                <Shield className="w-4 h-4 text-orange-600" />
                <span>Daily VaR</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-orange-600">
                ${varValue.toLocaleString()}
              </div>
              <div className="text-xs text-muted-foreground">
                {((varValue / 50000) * 100).toFixed(1)}% of limit
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Charts and Code Examples */}
        <Tabs defaultValue="charts" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="charts">Live Charts</TabsTrigger>
            <TabsTrigger value="code">Code Examples</TabsTrigger>
            <TabsTrigger value="risk">Risk Metrics</TabsTrigger>
          </TabsList>

          <TabsContent value="charts" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Price Chart */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Real-time Price Action</CardTitle>
                  <CardDescription>Live market data simulation</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={priceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis domain={['dataMin - 1', 'dataMax + 1']} />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="price" 
                        stroke="#3b82f6" 
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Performance Chart */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Strategy Performance</CardTitle>
                  <CardDescription>Monthly returns vs benchmark</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="returns" fill="#3b82f6" name="Strategy" />
                      <Bar dataKey="benchmark" fill="#94a3b8" name="Benchmark" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="code" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-1 gap-6">
              <Tabs defaultValue="strategy" className="w-full">
                <TabsList>
                  <TabsTrigger value="strategy">Strategy</TabsTrigger>
                  <TabsTrigger value="risk">Risk Engine</TabsTrigger>
                  <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
                </TabsList>
                
                {Object.entries(codeExamples).map(([key, code]) => (
                  <TabsContent key={key} value={key}>
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg capitalize">{key} Implementation</CardTitle>
                        <CardDescription>Production-ready code example</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                          <code>{code}</code>
                        </pre>
                      </CardContent>
                    </Card>
                  </TabsContent>
                ))}
              </Tabs>
            </div>
          </TabsContent>

          <TabsContent value="risk" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {riskMetrics.map((metric, index) => (
                <Card key={index}>
                  <CardHeader>
                    <CardTitle className="text-lg">{metric.metric}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-2xl font-bold">
                          {typeof metric.value === 'number' && metric.value > 1000 
                            ? `$${metric.value.toLocaleString()}` 
                            : metric.value}
                        </span>
                        <Badge variant={metric.value / metric.limit > 0.8 ? "destructive" : "secondary"}>
                          {((metric.value / metric.limit) * 100).toFixed(1)}%
                        </Badge>
                      </div>
                      <div className="w-full bg-muted rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full transition-all duration-300 ${
                            metric.value / metric.limit > 0.8 
                              ? 'bg-red-500' 
                              : metric.value / metric.limit > 0.6 
                                ? 'bg-yellow-500' 
                                : 'bg-green-500'
                          }`}
                          style={{ width: `${Math.min((metric.value / metric.limit) * 100, 100)}%` }}
                        ></div>
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Limit: {typeof metric.limit === 'number' && metric.limit > 1000 
                          ? `$${metric.limit.toLocaleString()}` 
                          : metric.limit}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </section>
  )
}

export default InteractiveDemo

