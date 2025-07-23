import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  Download,
  Settings,
  Play,
  CheckCircle,
  Terminal,
  Database,
  Monitor,
  TestTube,
  ArrowRight,
  Clock,
  AlertCircle
} from 'lucide-react'

const SetupGuide = () => {
  const setupSteps = [
    {
      step: 1,
      title: 'Environment Setup',
      description: 'Prepare your RTX 4070 workstation with required dependencies',
      icon: Settings,
      duration: '10 minutes',
      difficulty: 'Easy',
      commands: [
        'git clone <repository-url>',
        'cd trading-stack',
        'chmod +x scripts/*.sh'
      ],
      requirements: [
        'Ubuntu 22.04 LTS (WSL2 supported)',
        'RTX 4070 GPU with CUDA drivers',
        '32GB RAM minimum',
        '1TB NVMe SSD storage'
      ]
    },
    {
      step: 2,
      title: 'Data Infrastructure',
      description: 'Deploy Kafka, QuestDB, and market data ingestion pipeline',
      icon: Database,
      duration: '15 minutes',
      difficulty: 'Medium',
      commands: [
        './scripts/setup_data_ingestion.sh',
        'docker-compose up -d',
        'python data_ingestion/polygon_client.py'
      ],
      requirements: [
        'Docker and Docker Compose',
        'Polygon.io API key',
        'Network connectivity',
        'Port 9092 (Kafka) available'
      ]
    },
    {
      step: 3,
      title: 'Research Environment',
      description: 'Configure JupyterLab and FinRL for strategy development',
      icon: Terminal,
      duration: '20 minutes',
      difficulty: 'Medium',
      commands: [
        './scripts/setup_research_environment.sh',
        'jupyter lab --ip=0.0.0.0 --port=8888',
        'python research/environments/trading_env.py'
      ],
      requirements: [
        'CUDA toolkit installed',
        'Python 3.11+',
        'GPU memory available',
        'Port 8888 available'
      ]
    },
    {
      step: 4,
      title: 'Execution Engine',
      description: 'Deploy NautilusTrader and configure broker connections',
      icon: Play,
      duration: '25 minutes',
      difficulty: 'Advanced',
      commands: [
        './scripts/setup_execution_environment.sh',
        './execution/start_execution.sh',
        'python execution/nautilus_config.py'
      ],
      requirements: [
        'Interactive Brokers account',
        'TWS or IB Gateway installed',
        'API permissions enabled',
        'Risk limits configured'
      ]
    },
    {
      step: 5,
      title: 'Monitoring Stack',
      description: 'Set up Prometheus, Grafana, and custom metrics',
      icon: Monitor,
      duration: '15 minutes',
      difficulty: 'Easy',
      commands: [
        './scripts/setup_monitoring.sh',
        './monitoring/start_monitoring.sh',
        './monitoring/start_metrics_exporter.sh'
      ],
      requirements: [
        'Ports 3000, 9090, 9091 available',
        'Docker containers running',
        'Sufficient disk space',
        'Network access for alerts'
      ]
    },
    {
      step: 6,
      title: 'Integration Testing',
      description: 'Validate all components with comprehensive test suite',
      icon: TestTube,
      duration: '10 minutes',
      difficulty: 'Easy',
      commands: [
        'python scripts/integration_test.py',
        'python scripts/performance_benchmark.py',
        'curl http://localhost:9090/health'
      ],
      requirements: [
        'All services running',
        'Test data available',
        'Network connectivity',
        'Sufficient resources'
      ]
    }
  ]

  const quickStart = [
    'Clone repository and set permissions',
    'Run data ingestion setup script',
    'Configure API keys and credentials',
    'Start monitoring infrastructure',
    'Execute integration tests',
    'Access Grafana dashboards'
  ]

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'Easy': return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
      case 'Medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
      case 'Advanced': return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
    }
  }

  return (
    <section id="setup" className="py-16 bg-accent/5">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <Badge variant="secondary" className="mb-4">
            Setup Guide
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Quick Setup
            </span>{' '}
            Guide
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Get your hedge-fund-grade trading stack up and running in under 90 minutes 
            with our automated setup scripts and comprehensive documentation.
          </p>
        </div>

        {/* Quick Start Overview */}
        <div className="mb-16 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-xl p-8 border border-blue-200 dark:border-blue-800">
          <h3 className="text-xl font-bold mb-4 text-center">Quick Start Checklist</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {quickStart.map((item, index) => (
              <div key={index} className="flex items-center space-x-3 p-3 bg-white/50 dark:bg-black/20 rounded-lg">
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                <span className="text-sm font-medium">{item}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Setup Steps */}
        <div className="space-y-8">
          {setupSteps.map((step, index) => (
            <Card key={index} className="group hover:shadow-xl transition-all duration-300 border-2 hover:border-primary/50">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      {step.step}
                    </div>
                    <div>
                      <CardTitle className="text-xl group-hover:text-primary transition-colors mb-2">
                        {step.title}
                      </CardTitle>
                      <CardDescription className="text-sm">
                        {step.description}
                      </CardDescription>
                    </div>
                  </div>
                  <div className="flex flex-col items-end space-y-2">
                    <Badge className={getDifficultyColor(step.difficulty)}>
                      {step.difficulty}
                    </Badge>
                    <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                      <Clock className="w-3 h-3" />
                      <span>{step.duration}</span>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Commands */}
                  <div>
                    <h4 className="font-medium text-sm mb-3 flex items-center space-x-2">
                      <Terminal className="w-4 h-4" />
                      <span>Commands</span>
                    </h4>
                    <div className="bg-muted rounded-lg p-4 font-mono text-sm space-y-2">
                      {step.commands.map((command, cmdIndex) => (
                        <div key={cmdIndex} className="flex items-center space-x-2">
                          <span className="text-muted-foreground">$</span>
                          <span>{command}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Requirements */}
                  <div>
                    <h4 className="font-medium text-sm mb-3 flex items-center space-x-2">
                      <AlertCircle className="w-4 h-4" />
                      <span>Requirements</span>
                    </h4>
                    <div className="space-y-2">
                      {step.requirements.map((req, reqIndex) => (
                        <div key={reqIndex} className="flex items-center space-x-2 text-sm">
                          <div className="w-1.5 h-1.5 rounded-full bg-primary"></div>
                          <span className="text-muted-foreground">{req}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Step Navigation */}
                <div className="flex items-center justify-between pt-6 border-t border-border mt-6">
                  <div className="flex items-center space-x-2">
                    <step.icon className="w-5 h-5 text-primary" />
                    <span className="text-sm font-medium">Step {step.step} of {setupSteps.length}</span>
                  </div>
                  {index < setupSteps.length - 1 && (
                    <Button variant="outline" size="sm">
                      Next Step
                      <ArrowRight className="w-4 h-4 ml-1" />
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Download Section */}
        <div className="mt-16 text-center">
          <div className="bg-card border border-border rounded-xl p-8">
            <Download className="w-16 h-16 text-primary mx-auto mb-4" />
            <h3 className="text-2xl font-bold mb-4">Ready to Get Started?</h3>
            <p className="text-muted-foreground mb-6 max-w-2xl mx-auto">
              Download the complete trading stack with all components, documentation, 
              and automated setup scripts. Everything you need to deploy institutional-grade 
              trading infrastructure.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
                <Download className="w-5 h-5 mr-2" />
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

export default SetupGuide

