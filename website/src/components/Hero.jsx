import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ArrowRight, Zap, Shield, TrendingUp, Cpu, Database, BarChart3 } from 'lucide-react'

const Hero = () => {
  const features = [
    { icon: Zap, text: 'Real-time Execution' },
    { icon: Shield, text: 'Risk Management' },
    { icon: TrendingUp, text: 'AI Strategies' },
    { icon: Cpu, text: 'GPU Accelerated' },
    { icon: Database, text: 'High-frequency Data' },
    { icon: BarChart3, text: 'Advanced Analytics' }
  ]

  return (
    <section id="overview" className="pt-24 pb-16 bg-gradient-to-br from-background via-background to-accent/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-4xl mx-auto">
          {/* Badge */}
          <Badge variant="secondary" className="mb-6 px-4 py-2 text-sm font-medium">
            🚀 Production-Ready Trading Infrastructure
          </Badge>

          {/* Main Heading */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-6 leading-tight">
            <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent">
              Hedge-Fund-Grade
            </span>
            <br />
            <span className="text-foreground">Trading Stack</span>
          </h1>

          {/* Subtitle */}
          <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
            Complete institutional-quality algorithmic trading infrastructure designed for RTX 4070 workstations. 
            Features real-time data ingestion, AI-powered strategies, comprehensive risk management, and high-performance execution.
          </p>

          {/* Feature Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-10">
            {features.map((feature, index) => (
              <div
                key={index}
                className="flex flex-col items-center p-4 rounded-lg bg-card border border-border hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:scale-105"
              >
                <feature.icon className="w-6 h-6 text-primary mb-2" />
                <span className="text-sm font-medium text-center">{feature.text}</span>
              </div>
            ))}
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
            <Button 
              size="lg" 
              className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-3 text-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
            >
              Get Started
              <ArrowRight className="ml-2 w-5 h-5" />
            </Button>
            <Button 
              variant="outline" 
              size="lg" 
              className="px-8 py-3 text-lg font-semibold border-2 hover:bg-accent/50 transition-all duration-300"
            >
              View Documentation
            </Button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-2xl mx-auto">
            <div className="text-center">
              <div className="text-3xl font-bold text-primary mb-2">15+</div>
              <div className="text-muted-foreground">Core Components</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-primary mb-2">&lt;1ms</div>
              <div className="text-muted-foreground">Execution Latency</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-primary mb-2">24/7</div>
              <div className="text-muted-foreground">Monitoring</div>
            </div>
          </div>
        </div>
      </div>

      {/* Background Elements */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 right-1/4 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl"></div>
      </div>
    </section>
  )
}

export default Hero

