import { Button } from '@/components/ui/button'
import { 
  Github, 
  Twitter, 
  Linkedin, 
  Mail, 
  Zap,
  ExternalLink,
  Heart
} from 'lucide-react'

const Footer = () => {
  const footerSections = [
    {
      title: 'Product',
      links: [
        { name: 'Overview', href: '#overview' },
        { name: 'Architecture', href: '#architecture' },
        { name: 'Components', href: '#components' },
        { name: 'Pricing', href: '#pricing' }
      ]
    },
    {
      title: 'Documentation',
      links: [
        { name: 'Quick Start', href: '#setup' },
        { name: 'API Reference', href: '#docs' },
        { name: 'Tutorials', href: '#docs' },
        { name: 'Examples', href: '#docs' }
      ]
    },
    {
      title: 'Community',
      links: [
        { name: 'GitHub', href: 'https://github.com', external: true },
        { name: 'Discord', href: 'https://discord.com', external: true },
        { name: 'Discussions', href: 'https://github.com', external: true },
        { name: 'Blog', href: '#blog', external: true }
      ]
    },
    {
      title: 'Support',
      links: [
        { name: 'Help Center', href: '#help' },
        { name: 'Contact Us', href: '#contact' },
        { name: 'Status Page', href: '#status' },
        { name: 'Bug Reports', href: '#bugs' }
      ]
    }
  ]

  const socialLinks = [
    { icon: Github, href: 'https://github.com', label: 'GitHub' },
    { icon: Twitter, href: 'https://twitter.com', label: 'Twitter' },
    { icon: Linkedin, href: 'https://linkedin.com', label: 'LinkedIn' },
    { icon: Mail, href: 'mailto:support@trading-stack.dev', label: 'Email' }
  ]

  return (
    <footer className="bg-card border-t border-border">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Footer Content */}
        <div className="py-16">
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
            {/* Brand Section */}
            <div className="lg:col-span-2">
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                  <Zap className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Trading Stack
                </span>
              </div>
              <p className="text-muted-foreground mb-6 max-w-md">
                Institutional-grade algorithmic trading infrastructure designed for RTX 4070 workstations. 
                Build, deploy, and scale your trading operations with professional-grade tools.
              </p>
              
              {/* Social Links */}
              <div className="flex items-center space-x-4">
                {socialLinks.map((social, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    className="w-10 h-10 p-0"
                    asChild
                  >
                    <a
                      href={social.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label={social.label}
                    >
                      <social.icon className="w-4 h-4" />
                    </a>
                  </Button>
                ))}
              </div>
            </div>

            {/* Footer Links */}
            {footerSections.map((section, index) => (
              <div key={index}>
                <h3 className="font-semibold mb-4">{section.title}</h3>
                <ul className="space-y-3">
                  {section.links.map((link, linkIndex) => (
                    <li key={linkIndex}>
                      <a
                        href={link.href}
                        className="text-muted-foreground hover:text-foreground transition-colors duration-200 text-sm flex items-center space-x-1"
                        target={link.external ? '_blank' : undefined}
                        rel={link.external ? 'noopener noreferrer' : undefined}
                      >
                        <span>{link.name}</span>
                        {link.external && <ExternalLink className="w-3 h-3" />}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* Newsletter Section */}
        <div className="py-8 border-t border-border">
          <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
            <div>
              <h3 className="font-semibold mb-2">Stay Updated</h3>
              <p className="text-muted-foreground text-sm">
                Get the latest updates on new features, releases, and trading insights.
              </p>
            </div>
            <div className="flex items-center space-x-2 w-full md:w-auto">
              <input
                type="email"
                placeholder="Enter your email"
                className="flex-1 md:w-64 px-3 py-2 border border-border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary text-sm"
              />
              <Button size="sm" className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
                Subscribe
              </Button>
            </div>
          </div>
        </div>

        {/* Bottom Footer */}
        <div className="py-6 border-t border-border">
          <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
            <div className="flex items-center space-x-4 text-sm text-muted-foreground">
              <span>© 2024 Trading Stack. All rights reserved.</span>
              <span>•</span>
              <a href="#privacy" className="hover:text-foreground transition-colors">
                Privacy Policy
              </a>
              <span>•</span>
              <a href="#terms" className="hover:text-foreground transition-colors">
                Terms of Service
              </a>
            </div>
            
            <div className="flex items-center space-x-1 text-sm text-muted-foreground">
              <span>Built with</span>
              <Heart className="w-4 h-4 text-red-500 fill-current" />
              <span>by the Trading Stack Community</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer

