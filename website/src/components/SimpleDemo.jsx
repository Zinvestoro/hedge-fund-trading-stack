import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  DollarSign,
  TrendingUp,
  Activity,
  Shield,
  BarChart3
} from 'lucide-react'

const SimpleDemo = () => {
  const metrics = [
    {
      title: 'Portfolio Value',
      value: '$1,000,000',
      icon: DollarSign,
      color: 'text-blue-600',
      bgColor: 'border-blue-200 dark:border-blue-800'
    },
    {
      title: 'P&L',
      value: '+$25,000',
      icon: TrendingUp,
      color: 'text-green-600',
      bgColor: 'border-green-200 dark:border-green-800'
    },
    {
      title: 'Current Price',
      value: '$150.25',
      icon: Activity,
      color: 'text-purple-600',
      bgColor: 'border-purple-200 dark:border-purple-800'
    },
    {
      title: 'Daily VaR',
      value: '$35,000',
      icon: Shield,
      color: 'text-orange-600',
      bgColor: 'border-orange-200 dark:border-orange-800'
    }
  ]

  const codeExample = `# Momentum Strategy Implementation
from strategies.base_strategy import BaseStrategy
import numpy as np

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
            self.sell(bar.symbol, quantity=self.position.quantity)`

  return (
    <section className="py-16 bg-accent/5">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <Badge variant="secondary" className="mb-4">
            Live Demo
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Trading Stack
            </span>{' '}
            Dashboard
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Experience the power of institutional-grade trading infrastructure with 
            real-time monitoring, risk management, and performance analytics.
          </p>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {metrics.map((metric, index) => (
            <Card key={index} className={`border-2 ${metric.bgColor}`}>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center space-x-2">
                  <metric.icon className={`w-4 h-4 ${metric.color}`} />
                  <span>{metric.title}</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold ${metric.color}`}>
                  {metric.value}
                </div>
                <div className="text-xs text-muted-foreground">
                  Live Data
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Code Example */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <BarChart3 className="w-5 h-5" />
                <span>Strategy Implementation</span>
              </CardTitle>
              <CardDescription>Production-ready momentum strategy</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                <code>{codeExample}</code>
              </pre>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Key Features</CardTitle>
              <CardDescription>What makes this trading stack special</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-blue-600 mt-2"></div>
                  <div>
                    <h4 className="font-medium">Ultra-Low Latency</h4>
                    <p className="text-sm text-muted-foreground">Sub-millisecond execution with optimized data paths</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-green-600 mt-2"></div>
                  <div>
                    <h4 className="font-medium">AI-Powered Strategies</h4>
                    <p className="text-sm text-muted-foreground">GPU-accelerated reinforcement learning for strategy development</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-purple-600 mt-2"></div>
                  <div>
                    <h4 className="font-medium">Real-time Risk Management</h4>
                    <p className="text-sm text-muted-foreground">Continuous VaR monitoring with automated circuit breakers</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-orange-600 mt-2"></div>
                  <div>
                    <h4 className="font-medium">Comprehensive Monitoring</h4>
                    <p className="text-sm text-muted-foreground">Prometheus and Grafana integration for full observability</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* CTA Section */}
        <div className="mt-16 text-center">
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-xl p-8 border border-blue-200 dark:border-blue-800">
            <h3 className="text-2xl font-bold mb-4">Ready to Get Started?</h3>
            <p className="text-muted-foreground mb-6 max-w-2xl mx-auto">
              Deploy your own hedge-fund-grade trading infrastructure in under 90 minutes 
              with our automated setup scripts and comprehensive documentation.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
                Download Trading Stack
              </Button>
              <Button variant="outline" size="lg">
                View Documentation
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default SimpleDemo

