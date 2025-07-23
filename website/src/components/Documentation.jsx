import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  BookOpen,
  FileText,
  Code,
  Shield,
  BarChart3,
  Settings,
  ExternalLink,
  Download,
  Search,
  Star,
  Users,
  MessageCircle
} from 'lucide-react'

const Documentation = () => {
  const docSections = [
    {
      category: 'Getting Started',
      icon: BookOpen,
      color: 'from-green-500 to-emerald-500',
      docs: [
        {
          title: 'Quick Start Guide',
          description: 'Get up and running in under 90 minutes with automated setup scripts',
          pages: '12 pages',
          level: 'Beginner',
          topics: ['Installation', 'Configuration', 'First Trade', 'Monitoring']
        },
        {
          title: 'System Requirements',
          description: 'Hardware and software requirements for optimal performance',
          pages: '8 pages',
          level: 'Beginner',
          topics: ['Hardware Specs', 'OS Configuration', 'Dependencies', 'Network Setup']
        },
        {
          title: 'Architecture Overview',
          description: 'Comprehensive system architecture and component relationships',
          pages: '15 pages',
          level: 'Intermediate',
          topics: ['Data Flow', 'Component Design', 'Scalability', 'Performance']
        }
      ]
    },
    {
      category: 'Development',
      icon: Code,
      color: 'from-blue-500 to-cyan-500',
      docs: [
        {
          title: 'Strategy Development',
          description: 'Create and deploy custom trading strategies with FinRL and GPU acceleration',
          pages: '25 pages',
          level: 'Advanced',
          topics: ['RL Environments', 'Custom Strategies', 'Backtesting', 'Deployment']
        },
        {
          title: 'API Reference',
          description: 'Complete API documentation for all trading stack components',
          pages: '40 pages',
          level: 'Intermediate',
          topics: ['REST APIs', 'WebSocket Streams', 'Authentication', 'Rate Limits']
        },
        {
          title: 'Integration Guide',
          description: 'Integrate with external brokers, data providers, and third-party systems',
          pages: '18 pages',
          level: 'Advanced',
          topics: ['Broker APIs', 'Data Sources', 'Custom Connectors', 'Testing']
        }
      ]
    },
    {
      category: 'Operations',
      icon: Settings,
      color: 'from-purple-500 to-pink-500',
      docs: [
        {
          title: 'Deployment Guide',
          description: 'Production deployment strategies and best practices',
          pages: '22 pages',
          level: 'Advanced',
          topics: ['Docker Deployment', 'Cloud Setup', 'Security', 'Scaling']
        },
        {
          title: 'Monitoring & Alerting',
          description: 'Set up comprehensive monitoring with Prometheus and Grafana',
          pages: '16 pages',
          level: 'Intermediate',
          topics: ['Metrics Collection', 'Dashboards', 'Alerts', 'Performance Tuning']
        },
        {
          title: 'Troubleshooting',
          description: 'Common issues, debugging techniques, and performance optimization',
          pages: '20 pages',
          level: 'Intermediate',
          topics: ['Common Issues', 'Log Analysis', 'Performance', 'Recovery']
        }
      ]
    },
    {
      category: 'Risk & Compliance',
      icon: Shield,
      color: 'from-red-500 to-orange-500',
      docs: [
        {
          title: 'Risk Management',
          description: 'Configure and customize risk controls, VaR calculations, and circuit breakers',
          pages: '28 pages',
          level: 'Advanced',
          topics: ['VaR Models', 'Position Limits', 'Circuit Breakers', 'Stress Testing']
        },
        {
          title: 'Compliance Framework',
          description: 'Regulatory compliance, audit trails, and reporting requirements',
          pages: '24 pages',
          level: 'Advanced',
          topics: ['Audit Trails', 'Regulatory Reporting', 'Data Retention', 'Privacy']
        }
      ]
    }
  ]

  const resources = [
    {
      title: 'Complete Setup Guide',
      description: '50+ page comprehensive guide covering every aspect of the trading stack',
      icon: FileText,
      size: '2.1 MB',
      format: 'PDF'
    },
    {
      title: 'Source Code',
      description: 'Complete source code with 15+ production-ready components',
      icon: Code,
      size: '45 MB',
      format: 'ZIP'
    },
    {
      title: 'Sample Strategies',
      description: 'Example trading strategies and backtesting notebooks',
      icon: BarChart3,
      size: '12 MB',
      format: 'ZIP'
    }
  ]

  const getLevelColor = (level) => {
    switch (level) {
      case 'Beginner': return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
      case 'Intermediate': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
      case 'Advanced': return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
    }
  }

  return (
    <section id="docs" className="py-16">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <Badge variant="secondary" className="mb-4">
            Documentation
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Comprehensive
            </span>{' '}
            Documentation
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Detailed guides, API references, and best practices to help you master 
            every aspect of the trading stack infrastructure.
          </p>
        </div>

        {/* Search Bar */}
        <div className="max-w-2xl mx-auto mb-12">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-5 h-5" />
            <input
              type="text"
              placeholder="Search documentation..."
              className="w-full pl-10 pr-4 py-3 border border-border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
            />
          </div>
        </div>

        {/* Documentation Sections */}
        <div className="space-y-12">
          {docSections.map((section, sectionIndex) => (
            <div key={sectionIndex}>
              {/* Section Header */}
              <div className="flex items-center space-x-3 mb-6">
                <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${section.color} flex items-center justify-center`}>
                  <section.icon className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-2xl font-bold">{section.category}</h3>
              </div>

              {/* Section Documents */}
              <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                {section.docs.map((doc, docIndex) => (
                  <Card key={docIndex} className="group hover:shadow-xl transition-all duration-300 border-2 hover:border-primary/50">
                    <CardHeader>
                      <div className="flex items-start justify-between mb-2">
                        <CardTitle className="text-lg group-hover:text-primary transition-colors">
                          {doc.title}
                        </CardTitle>
                        <Badge className={getLevelColor(doc.level)}>
                          {doc.level}
                        </Badge>
                      </div>
                      <CardDescription className="text-sm">
                        {doc.description}
                      </CardDescription>
                      <div className="text-xs text-muted-foreground">
                        {doc.pages}
                      </div>
                    </CardHeader>
                    <CardContent>
                      {/* Topics */}
                      <div className="mb-4">
                        <h4 className="font-medium text-sm mb-2">Topics Covered:</h4>
                        <div className="flex flex-wrap gap-1">
                          {doc.topics.map((topic, topicIndex) => (
                            <Badge key={topicIndex} variant="outline" className="text-xs">
                              {topic}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center space-x-2">
                        <Button variant="outline" size="sm" className="flex-1">
                          Read Online
                          <ExternalLink className="w-3 h-3 ml-1" />
                        </Button>
                        <Button variant="outline" size="sm">
                          <Download className="w-3 h-3" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Resources Section */}
        <div className="mt-16">
          <h3 className="text-2xl font-bold text-center mb-8">Additional Resources</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {resources.map((resource, index) => (
              <Card key={index} className="group hover:shadow-xl transition-all duration-300 text-center">
                <CardHeader>
                  <div className="w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <resource.icon className="w-8 h-8 text-white" />
                  </div>
                  <CardTitle className="text-lg group-hover:text-primary transition-colors">
                    {resource.title}
                  </CardTitle>
                  <CardDescription className="text-sm">
                    {resource.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center space-x-4 text-xs text-muted-foreground mb-4">
                    <span>{resource.size}</span>
                    <span>•</span>
                    <span>{resource.format}</span>
                  </div>
                  <Button className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Community Section */}
        <div className="mt-16 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-xl p-8 border border-blue-200 dark:border-blue-800">
          <div className="text-center">
            <Users className="w-12 h-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-xl font-bold mb-2">Join the Community</h3>
            <p className="text-muted-foreground mb-6">
              Connect with other traders, developers, and quants using the trading stack. 
              Share strategies, get help, and contribute to the project.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button variant="outline" className="flex items-center space-x-2">
                <MessageCircle className="w-4 h-4" />
                <span>Discord Community</span>
              </Button>
              <Button variant="outline" className="flex items-center space-x-2">
                <Star className="w-4 h-4" />
                <span>GitHub Discussions</span>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Documentation

