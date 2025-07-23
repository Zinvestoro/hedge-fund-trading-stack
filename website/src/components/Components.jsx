import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  Database, 
  Zap, 
  Shield, 
  BarChart3, 
  Cpu, 
  Network,
  GitBranch,
  Code,
  ExternalLink,
  CheckCircle
} from 'lucide-react'

const Components = () => {
  const components = [
    {
      category: 'Data Infrastructure',
      icon: Database,
      color: 'from-blue-500 to-cyan-500',
      items: [
        {
          name: 'Polygon.io Client',
          description: 'Professional market data integration with real-time streaming capabilities',
          features: ['Real-time quotes', 'Historical data', 'Multiple asset classes', 'WebSocket streaming'],
          status: 'Production Ready',
          file: 'polygon_client.py'
        },
        {
          name: 'Crypto Aggregator',
          description: 'Multi-exchange cryptocurrency data aggregation and normalization',
          features: ['Multi-exchange support', 'Data normalization', 'Orderbook streaming', 'Trade aggregation'],
          status: 'Production Ready',
          file: 'crypto_aggregator.py'
        },
        {
          name: 'QuestDB Integration',
          description: 'High-performance time-series database for market data storage',
          features: ['Columnar storage', 'SQL interface', 'High ingestion rate', 'Real-time queries'],
          status: 'Production Ready',
          file: 'docker-compose.yml'
        }
      ]
    },
    {
      category: 'Execution & Risk',
      icon: Zap,
      color: 'from-green-500 to-emerald-500',
      items: [
        {
          name: 'NautilusTrader Engine',
          description: 'High-performance execution engine with institutional-grade features',
          features: ['Sub-millisecond latency', 'FIX protocol support', 'Smart order routing', 'Risk controls'],
          status: 'Production Ready',
          file: 'nautilus_config.py'
        },
        {
          name: 'Risk Management Engine',
          description: 'Real-time risk monitoring with VaR calculations and circuit breakers',
          features: ['Real-time VaR', 'Position limits', 'Circuit breakers', 'Compliance tracking'],
          status: 'Production Ready',
          file: 'risk_engine.py'
        }
      ]
    },
    {
      category: 'Strategy Development',
      icon: Cpu,
      color: 'from-purple-500 to-pink-500',
      items: [
        {
          name: 'FinRL Environment',
          description: 'Custom reinforcement learning environment for strategy development',
          features: ['GPU acceleration', 'Custom rewards', 'Multi-asset support', 'Backtesting integration'],
          status: 'Production Ready',
          file: 'trading_env.py'
        },
        {
          name: 'Momentum Strategy',
          description: 'Production-ready momentum trading strategy with risk controls',
          features: ['EMA crossover signals', 'Position sizing', 'Stop losses', 'Performance tracking'],
          status: 'Production Ready',
          file: 'momentum_strategy.py'
        },
        {
          name: 'Multi-Agent System',
          description: 'Coordinated multi-agent trading system with LLM-based routing',
          features: ['Agent coordination', 'Strategy selection', 'Performance optimization', 'Risk allocation'],
          status: 'Production Ready',
          file: 'multi_agent_system.py'
        }
      ]
    },
    {
      category: 'Monitoring & Analytics',
      icon: BarChart3,
      color: 'from-indigo-500 to-blue-500',
      items: [
        {
          name: 'Prometheus Exporter',
          description: 'Custom metrics exporter for trading-specific KPIs and system health',
          features: ['Custom metrics', 'Real-time monitoring', 'Alert integration', 'Performance tracking'],
          status: 'Production Ready',
          file: 'prometheus_exporter.py'
        },
        {
          name: 'Grafana Dashboards',
          description: 'Professional dashboards for portfolio monitoring and system observability',
          features: ['Real-time charts', 'Custom panels', 'Alert visualization', 'Mobile responsive'],
          status: 'Production Ready',
          file: 'setup_monitoring.sh'
        }
      ]
    }
  ]

  return (
    <section id="components" className="py-16">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <Badge variant="secondary" className="mb-4">
            Core Components
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Production-Ready
            </span>{' '}
            Components
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            15+ carefully crafted components providing end-to-end trading infrastructure 
            with institutional-grade reliability and performance.
          </p>
        </div>

        {/* Components Grid */}
        <div className="space-y-12">
          {components.map((category, categoryIndex) => (
            <div key={categoryIndex}>
              {/* Category Header */}
              <div className="flex items-center space-x-3 mb-6">
                <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${category.color} flex items-center justify-center`}>
                  <category.icon className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-2xl font-bold">{category.category}</h3>
              </div>

              {/* Category Components */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {category.items.map((component, componentIndex) => (
                  <Card key={componentIndex} className="group hover:shadow-xl transition-all duration-300 border-2 hover:border-primary/50">
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div>
                          <CardTitle className="text-lg group-hover:text-primary transition-colors mb-2">
                            {component.name}
                          </CardTitle>
                          <CardDescription className="text-sm mb-3">
                            {component.description}
                          </CardDescription>
                        </div>
                        <Badge variant="secondary" className="text-xs">
                          <CheckCircle className="w-3 h-3 mr-1" />
                          {component.status}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      {/* Features */}
                      <div className="mb-4">
                        <h4 className="font-medium text-sm mb-2">Key Features:</h4>
                        <div className="grid grid-cols-2 gap-2">
                          {component.features.map((feature, featureIndex) => (
                            <div key={featureIndex} className="flex items-center space-x-2 text-xs">
                              <div className="w-1.5 h-1.5 rounded-full bg-primary"></div>
                              <span className="text-muted-foreground">{feature}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center justify-between pt-4 border-t border-border">
                        <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                          <Code className="w-3 h-3" />
                          <span>{component.file}</span>
                        </div>
                        <Button variant="outline" size="sm" className="text-xs">
                          View Code
                          <ExternalLink className="w-3 h-3 ml-1" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Integration Note */}
        <div className="mt-16 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-xl p-8 border border-blue-200 dark:border-blue-800">
          <div className="text-center">
            <GitBranch className="w-12 h-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-xl font-bold mb-2">Seamless Integration</h3>
            <p className="text-muted-foreground mb-4">
              All components are designed to work together seamlessly with automated setup scripts 
              and comprehensive integration testing.
            </p>
            <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
              View Integration Guide
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Components

