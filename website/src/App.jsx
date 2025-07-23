import './App.css'
import Header from './components/Header'
import Hero from './components/Hero'
import Architecture from './components/Architecture'
import Components from './components/Components'
import SimpleDemo from './components/SimpleDemo'
import SetupGuide from './components/SetupGuide'
import Documentation from './components/Documentation'
import Footer from './components/Footer'

function App() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <Hero />
        <Architecture />
        <Components />
        <SimpleDemo />
        <SetupGuide />
        <Documentation />
      </main>
      <Footer />
    </div>
  )
}

export default App
