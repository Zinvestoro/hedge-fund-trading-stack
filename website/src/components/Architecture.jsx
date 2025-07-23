import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  Database, 
  Zap, 
  Shield, 
  BarChart3, 
  Cpu, 
  Network, 
  Monitor,
  GitBranch,
  ArrowRight
} from 'lucide-react'

const Architecture = () => {
  const layers = [
    {
      title: 'Data Ingestion Layer',
      description: 'Real-time market data streaming and processing',
      icon: Database,
      color: 'from-blue-500 to-cyan-500',
      components: [
        { name: 'Apache Kafka', desc: 'Message streaming platform' },
        { name: 'QuestDB', desc: 'Time-series database' },
        { name: 'Polygon.io API', desc: 'Market data provider' },
        { name: 'Crypto Aggregator', desc: 'Multi-exchange data' }
      ]
    },
    {
      title: 'Strategy Development',
      description: 'AI-powered trading strategy creation and backtesting',
      icon: Cpu,
      color: 'from-purple-500 to-pink-500',
      components: [
        { name: 'FinRL Framework', desc: 'Reinforcement learning' },
        { name: 'JupyterLab', desc: 'Research environment' },
        { name: 'GPU Acceleration', desc: 'CUDA-optimized training' },
        { name: 'Multi-Agent System', desc: 'Strategy coordination' }
      ]
    },
    {
      title: 'Risk Management',
      description: 'Real-time risk monitoring and automated controls',
      icon: Shield,
      color: 'from-red-500 to-orange-500',
      components: [
        { name: 'Real-time VaR', desc: 'Value at Risk calculation' },
        { name: 'Circuit Breakers', desc: 'Automated risk controls' },
        { name: 'Position Limits', desc: 'Exposure management' },
        { name: 'Compliance Engine', desc: 'Regulatory compliance' }
      ]
    },
    {
      title: 'Execution Engine',
      description: 'High-performance order execution and management',
      icon: Zap,
      color: 'from-green-500 to-emerald-500',
      components: [
        { name: 'NautilusTrader', desc: 'Execution framework' },
        { name: 'FIX Protocol', desc: 'Broker connectivity' },
        { name: 'Order Management', desc: 'Smart order routing' },
        { name: 'Latency Optimization', desc: 'Microsecond execution' }
      ]
    },
    {
      title: 'Monitoring & Analytics',
      description: 'Comprehensive system observability and performance tracking',
      icon: Monitor,
      color: 'from-indigo-500 to-blue-500',
      components: [
        { name: 'Prometheus', desc: 'Metrics collection' },
        { name: 'Grafana', desc: 'Visualization dashboards' },
        { name: 'Custom Metrics', desc: 'Trading-specific KPIs' },
        { name: 'Alert System', desc: 'Real-time notifications' }
      ]
    }
  ]

  const dataFlow = [
    'Market Data Ingestion',
    'Signal Generation',
    'Risk Assessment',
    'Order Execution',
    'Performance Monitoring'
  ]

  return (
    <section id="architecture" className="py-16 bg-accent/5">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <Badge variant="secondary" className="mb-4">
            System Architecture
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Institutional-Grade
            </span>{' '}
            Architecture
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Modular, scalable architecture designed for high-frequency trading operations 
            with enterprise-level reliability and performance.
          </p>
        </div>

        {/* Data Flow Visualization */}
        <div className="mb-16">
          <h3 className="text-2xl font-bold text-center mb-8">Data Flow Pipeline</h3>
          <div className="flex flex-wrap justify-center items-center gap-4 mb-8">
            {dataFlow.map((step, index) => (
              <div key={index} className="flex items-center">
                <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-2 rounded-lg font-medium text-sm whitespace-nowrap">
                  {step}
                </div>
                {index < dataFlow.length - 1 && (
                  <ArrowRight className="w-5 h-5 text-muted-foreground mx-2" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Architecture Layers */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
          {layers.map((layer, index) => (
            <Card key={index} className="group hover:shadow-xl transition-all duration-300 border-2 hover:border-primary/50">
              <CardHeader>
                <div className="flex items-center space-x-3 mb-3">
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${layer.color} flex items-center justify-center`}>
                    <layer.icon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-lg group-hover:text-primary transition-colors">
                      {layer.title}
                    </CardTitle>
                  </div>
                </div>
                <CardDescription className="text-sm">
                  {layer.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {layer.components.map((component, compIndex) => (
                    <div key={compIndex} className="flex items-start space-x-3 p-3 rounded-lg bg-accent/30 hover:bg-accent/50 transition-colors">
                      <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0"></div>
                      <div>
                        <div className="font-medium text-sm">{component.name}</div>
                        <div className="text-xs text-muted-foreground">{component.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Technical Specifications */}
        <div className="mt-16 bg-card border border-border rounded-xl p-8">
          <h3 className="text-2xl font-bold mb-6 text-center">Technical Specifications</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-3">
                <Zap className="w-8 h-8 text-white" />
              </div>
              <h4 className="font-semibold mb-2">Ultra-Low Latency</h4>
              <p className="text-sm text-muted-foreground">Sub-millisecond execution with optimized data paths</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-green-600 to-emerald-600 rounded-full flex items-center justify-center mx-auto mb-3">
                <Database className="w-8 h-8 text-white" />
              </div>
              <h4 className="font-semibold mb-2">High Throughput</h4>
              <p className="text-sm text-muted-foreground">Process millions of market data points per second</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full flex items-center justify-center mx-auto mb-3">
                <Shield className="w-8 h-8 text-white" />
              </div>
              <h4 className="font-semibold mb-2">Enterprise Security</h4>
              <p className="text-sm text-muted-foreground">Bank-grade security with comprehensive audit trails</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-orange-600 to-red-600 rounded-full flex items-center justify-center mx-auto mb-3">
                <BarChart3 className="w-8 h-8 text-white" />
              </div>
              <h4 className="font-semibold mb-2">Real-time Analytics</h4>
              <p className="text-sm text-muted-foreground">Live performance monitoring and risk assessment</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Architecture

