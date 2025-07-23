#!/bin/bash

# Trading Stack Research Environment Setup Script
# Sets up JupyterLab, FinRL, and other research tools

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRADING_STACK_DIR="$(dirname "$SCRIPT_DIR")"
RESEARCH_DIR="$TRADING_STACK_DIR/research"

log "Setting up trading stack research environment..."
log "Research directory: $RESEARCH_DIR"

# Create research directory structure
log "Creating research directory structure..."
mkdir -p "$RESEARCH_DIR"/{notebooks,environments,models,data,experiments,reports}

# Activate virtual environment
cd "$TRADING_STACK_DIR"
if [[ ! -d "venv" ]]; then
    error "Virtual environment not found. Please run setup_data_ingestion.sh first."
    exit 1
fi

source venv/bin/activate

# Install JupyterLab and extensions
log "Installing JupyterLab and extensions..."
pip install --upgrade pip

# Core JupyterLab
pip install jupyterlab==4.0.5
pip install jupyter-ai==2.1.0
pip install jupyterlab-git==0.42.0
pip install jupyterlab-lsp==5.0.0

# Visualization extensions
pip install jupyterlab-plotly==5.15.0
pip install jupyterlab-widgets==3.0.8
pip install ipywidgets==8.1.0

# System monitoring
pip install jupyterlab-system-monitor==0.8.0

# Install machine learning frameworks
log "Installing machine learning frameworks..."

# PyTorch (CPU version for sandbox, GPU version for actual deployment)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==0.15.2

# TensorFlow
pip install tensorflow==2.13.0

# Scikit-learn and related
pip install scikit-learn==1.3.0
pip install xgboost==1.7.6
pip install lightgbm==4.0.0

# Install FinRL and reinforcement learning libraries
log "Installing FinRL and RL libraries..."
pip install finrl==3.0.0
pip install stable-baselines3[extra]==2.0.0
pip install sb3-contrib==2.0.0
pip install optuna==3.3.0
pip install ray[rllib]==2.6.3
pip install gymnasium==0.29.1

# Install financial data libraries
log "Installing financial data libraries..."
pip install yfinance==0.2.18
pip install pandas-datareader==0.10.0
pip install fredapi==0.5.0
pip install alpha-vantage==2.3.1
pip install polygon-api-client==1.12.0

# Install backtesting frameworks
log "Installing backtesting frameworks..."
pip install vectorbt==0.25.2
pip install zipline-reloaded==2.2.0
pip install backtrader==1.9.78.123

# Install additional analysis libraries
log "Installing analysis libraries..."
pip install ta==0.10.2  # Technical analysis
pip install pyfolio==0.9.2  # Portfolio analysis
pip install empyrical==0.5.5  # Performance metrics
pip install quantstats==0.0.62  # Quantitative statistics

# Install optimization libraries
log "Installing optimization libraries..."
pip install cvxpy==1.3.2
pip install pyportfolioopt==1.5.5
pip install scipy==1.11.1

# Configure JupyterLab
log "Configuring JupyterLab..."
jupyter lab --generate-config

# Create JupyterLab configuration
cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
# JupyterLab Configuration for Trading Stack

# Server configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_remote_access = True

# Security configuration (development settings)
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = True

# Performance optimization
c.ServerApp.max_buffer_size = 268435456  # 256MB
c.ServerApp.iopub_data_rate_limit = 10000000  # 10MB/s
c.ServerApp.rate_limit_window = 3.0

# Resource limits for large datasets
c.NotebookApp.max_buffer_size = 268435456
c.NotebookApp.iopub_data_rate_limit = 10000000

# Kernel management
c.MappingKernelManager.cull_idle_timeout = 3600  # 1 hour
c.MappingKernelManager.cull_interval = 300  # 5 minutes
c.MappingKernelManager.cull_connected = False

# Working directory
c.ServerApp.root_dir = '/home/ubuntu/trading-stack/research'
EOF

# Create custom kernel for trading stack
log "Creating custom kernel..."
python -m ipykernel install --user --name trading-stack --display-name "Trading Stack"

# Create kernel configuration
mkdir -p ~/.local/share/jupyter/kernels/trading-stack
cat > ~/.local/share/jupyter/kernels/trading-stack/kernel.json << EOF
{
    "argv": [
        "$TRADING_STACK_DIR/venv/bin/python",
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}"
    ],
    "display_name": "Trading Stack",
    "language": "python",
    "env": {
        "PYTHONPATH": "$TRADING_STACK_DIR:$PYTHONPATH"
    }
}
EOF

# Clone FinRL-DeepSeek for LLM-enhanced RL
log "Setting up FinRL-DeepSeek..."
cd "$RESEARCH_DIR"
if [[ ! -d "FinRL-DeepSeek" ]]; then
    git clone https://github.com/AI4Finance-Foundation/FinRL-DeepSeek.git
    cd FinRL-DeepSeek
    pip install -e .
    cd ..
fi

# Install ABIDES for market simulation
log "Setting up ABIDES market simulation..."
if [[ ! -d "abides" ]]; then
    git clone https://github.com/abides-sim/abides.git
    cd abides
    pip install -e .
    cd ..
fi

# Install NautilusTrader for high-performance backtesting
log "Installing NautilusTrader..."
pip install nautilus_trader==1.190.0
pip install uvloop==0.17.0

# Create example notebooks
log "Creating example notebooks..."

# Basic environment test notebook
cat > "$RESEARCH_DIR/notebooks/01_environment_test.ipynb" << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Trading Stack Environment Test\n",
    "\n",
    "This notebook tests the custom FinRL environment integration with the trading stack."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('/home/ubuntu/trading-stack')\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from research.environments.trading_env import TradingStackEnvironment\n",
    "\n",
    "print(\"Trading Stack Environment Test\")\n",
    "print(\"=============================\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create environment\n",
    "symbols = ['AAPL', 'GOOGL', 'MSFT']\n",
    "start_date = '2023-01-01'\n",
    "end_date = '2023-12-31'\n",
    "\n",
    "env = TradingStackEnvironment(\n",
    "    symbols=symbols,\n",
    "    start_date=start_date,\n",
    "    end_date=end_date,\n",
    "    initial_balance=100000\n",
    ")\n",
    "\n",
    "print(f\"Environment created successfully!\")\n",
    "print(f\"Action space: {env.action_space}\")\n",
    "print(f\"Observation space: {env.observation_space}\")\n",
    "print(f\"Data shape: {env.price_data.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test environment\n",
    "obs, info = env.reset()\n",
    "print(f\"Initial observation shape: {obs.shape}\")\n",
    "print(f\"Initial portfolio value: ${info['portfolio_value']:,.2f}\")\n",
    "\n",
    "# Run random actions for testing\n",
    "portfolio_values = [info['portfolio_value']]\n",
    "\n",
    "for i in range(100):\n",
    "    action = env.action_space.sample()\n",
    "    obs, reward, done, truncated, info = env.step(action)\n",
    "    \n",
    "    portfolio_values.append(info['portfolio_value'])\n",
    "    \n",
    "    if i % 20 == 0:\n",
    "        print(f\"Step {i}: Portfolio=${info['portfolio_value']:,.2f}, Reward={reward:.4f}\")\n",
    "    \n",
    "    if done:\n",
    "        break\n",
    "\n",
    "print(f\"\\nFinal portfolio value: ${info['portfolio_value']:,.2f}\")\n",
    "print(f\"Total return: {(info['portfolio_value']/100000 - 1)*100:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot portfolio performance\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.plot(portfolio_values)\n",
    "plt.title('Portfolio Value Over Time (Random Actions)')\n",
    "plt.xlabel('Steps')\n",
    "plt.ylabel('Portfolio Value ($)')\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n",
    "# Get final statistics\n",
    "stats = env.get_portfolio_stats()\n",
    "print(\"\\nPortfolio Statistics:\")\n",
    "for key, value in stats.items():\n",
    "    if isinstance(value, float):\n",
    "        print(f\"{key}: {value:.4f}\")\n",
    "    else:\n",
    "        print(f\"{key}: {value}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Trading Stack",
   "language": "python",
   "name": "trading-stack"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# FinRL training notebook
cat > "$RESEARCH_DIR/notebooks/02_finrl_training.ipynb" << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# FinRL Agent Training\n",
    "\n",
    "This notebook demonstrates training a reinforcement learning agent using FinRL and the custom trading environment."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('/home/ubuntu/trading-stack')\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from stable_baselines3 import PPO, A2C, SAC\n",
    "from stable_baselines3.common.vec_env import DummyVecEnv\n",
    "from stable_baselines3.common.callbacks import EvalCallback\n",
    "from research.environments.trading_env import TradingStackEnvironment\n",
    "\n",
    "print(\"FinRL Agent Training\")\n",
    "print(\"===================\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create training and evaluation environments\n",
    "symbols = ['AAPL', 'GOOGL', 'MSFT']\n",
    "train_start = '2023-01-01'\n",
    "train_end = '2023-09-30'\n",
    "test_start = '2023-10-01'\n",
    "test_end = '2023-12-31'\n",
    "\n",
    "# Training environment\n",
    "train_env = TradingStackEnvironment(\n",
    "    symbols=symbols,\n",
    "    start_date=train_start,\n",
    "    end_date=train_end,\n",
    "    initial_balance=100000\n",
    ")\n",
    "\n",
    "# Evaluation environment\n",
    "eval_env = TradingStackEnvironment(\n",
    "    symbols=symbols,\n",
    "    start_date=test_start,\n",
    "    end_date=test_end,\n",
    "    initial_balance=100000\n",
    ")\n",
    "\n",
    "# Wrap environments\n",
    "train_env = DummyVecEnv([lambda: train_env])\n",
    "eval_env = DummyVecEnv([lambda: eval_env])\n",
    "\n",
    "print(\"Environments created successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create PPO agent\n",
    "model = PPO(\n",
    "    'MlpPolicy',\n",
    "    train_env,\n",
    "    verbose=1,\n",
    "    learning_rate=3e-4,\n",
    "    n_steps=2048,\n",
    "    batch_size=64,\n",
    "    n_epochs=10,\n",
    "    gamma=0.99,\n",
    "    gae_lambda=0.95,\n",
    "    clip_range=0.2,\n",
    "    ent_coef=0.01,\n",
    "    tensorboard_log=\"./tensorboard_logs/\"\n",
    ")\n",
    "\n",
    "print(\"PPO agent created!\")\n",
    "print(f\"Policy: {model.policy}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set up evaluation callback\n",
    "eval_callback = EvalCallback(\n",
    "    eval_env,\n",
    "    best_model_save_path='./models/',\n",
    "    log_path='./logs/',\n",
    "    eval_freq=10000,\n",
    "    deterministic=True,\n",
    "    render=False\n",
    ")\n",
    "\n",
    "# Train the agent\n",
    "print(\"Starting training...\")\n",
    "model.learn(\n",
    "    total_timesteps=50000,\n",
    "    callback=eval_callback,\n",
    "    progress_bar=True\n",
    ")\n",
    "\n",
    "print(\"Training completed!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test the trained agent\n",
    "test_env = TradingStackEnvironment(\n",
    "    symbols=symbols,\n",
    "    start_date=test_start,\n",
    "    end_date=test_end,\n",
    "    initial_balance=100000\n",
    ")\n",
    "\n",
    "obs, info = test_env.reset()\n",
    "portfolio_values = [info['portfolio_value']]\n",
    "actions_taken = []\n",
    "\n",
    "for i in range(1000):\n",
    "    action, _states = model.predict(obs, deterministic=True)\n",
    "    obs, reward, done, truncated, info = test_env.step(action)\n",
    "    \n",
    "    portfolio_values.append(info['portfolio_value'])\n",
    "    actions_taken.append(action)\n",
    "    \n",
    "    if i % 100 == 0:\n",
    "        print(f\"Step {i}: Portfolio=${info['portfolio_value']:,.2f}\")\n",
    "    \n",
    "    if done:\n",
    "        break\n",
    "\n",
    "print(f\"\\nFinal Results:\")\n",
    "print(f\"Portfolio value: ${info['portfolio_value']:,.2f}\")\n",
    "print(f\"Total return: {(info['portfolio_value']/100000 - 1)*100:.2f}%\")\n",
    "\n",
    "stats = test_env.get_portfolio_stats()\n",
    "for key, value in stats.items():\n",
    "    if isinstance(value, float):\n",
    "        print(f\"{key}: {value:.4f}\")\n",
    "    else:\n",
    "        print(f\"{key}: {value}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot results\n",
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))\n",
    "\n",
    "# Portfolio performance\n",
    "ax1.plot(portfolio_values)\n",
    "ax1.set_title('Portfolio Performance (Trained Agent)')\n",
    "ax1.set_xlabel('Steps')\n",
    "ax1.set_ylabel('Portfolio Value ($)')\n",
    "ax1.grid(True)\n",
    "\n",
    "# Actions over time\n",
    "actions_array = np.array(actions_taken)\n",
    "for i, symbol in enumerate(symbols):\n",
    "    ax2.plot(actions_array[:, i], label=symbol, alpha=0.7)\n",
    "\n",
    "ax2.set_title('Actions Over Time')\n",
    "ax2.set_xlabel('Steps')\n",
    "ax2.set_ylabel('Action Value')\n",
    "ax2.legend()\n",
    "ax2.grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Trading Stack",
   "language": "python",
   "name": "trading-stack"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create startup script for JupyterLab
log "Creating JupyterLab startup script..."
cat > "$TRADING_STACK_DIR/scripts/start_jupyter.sh" << 'EOF'
#!/bin/bash

# Start JupyterLab for Trading Stack Research

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRADING_STACK_DIR="$(dirname "$SCRIPT_DIR")"

cd "$TRADING_STACK_DIR"

# Activate virtual environment
source venv/bin/activate

# Start JupyterLab
echo "Starting JupyterLab..."
echo "Access at: http://localhost:8888"
echo "Working directory: $TRADING_STACK_DIR/research"
echo ""
echo "Press Ctrl+C to stop"

cd research
jupyter lab --no-browser --allow-root
EOF

chmod +x "$TRADING_STACK_DIR/scripts/start_jupyter.sh"

# Create requirements file for research environment
log "Creating research requirements file..."
cat > "$RESEARCH_DIR/requirements.txt" << 'EOF'
# Trading Stack Research Environment Requirements

# Core scientific computing
numpy==1.24.3
pandas==2.1.0
scipy==1.11.1
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# Machine learning frameworks
torch==2.0.1
torchvision==0.15.2
torchaudio==0.15.2
tensorflow==2.13.0
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0

# Reinforcement learning
finrl==3.0.0
stable-baselines3[extra]==2.0.0
sb3-contrib==2.0.0
optuna==3.3.0
ray[rllib]==2.6.3
gymnasium==0.29.1

# Financial data and analysis
yfinance==0.2.18
pandas-datareader==0.10.0
fredapi==0.5.0
alpha-vantage==2.3.1
polygon-api-client==1.12.0
ta==0.10.2
pyfolio==0.9.2
empyrical==0.5.5
quantstats==0.0.62

# Backtesting frameworks
vectorbt==0.25.2
zipline-reloaded==2.2.0
backtrader==1.9.78.123
nautilus_trader==1.190.0

# Portfolio optimization
cvxpy==1.3.2
pyportfolioopt==1.5.5

# JupyterLab and extensions
jupyterlab==4.0.5
jupyter-ai==2.1.0
jupyterlab-git==0.42.0
jupyterlab-lsp==5.0.0
jupyterlab-plotly==5.15.0
jupyterlab-widgets==3.0.8
ipywidgets==8.1.0
jupyterlab-system-monitor==0.8.0

# Database connectivity
psycopg2-binary==2.9.7
questdb==1.1.0

# Utilities
tqdm==4.66.1
joblib==1.3.2
dask==2023.8.1
EOF

# Create README for research environment
log "Creating research environment README..."
cat > "$RESEARCH_DIR/README.md" << 'EOF'
# Trading Stack Research Environment

This directory contains the research and development environment for the trading stack, including Jupyter notebooks, custom environments, and machine learning models.

## Directory Structure

```
research/
├── notebooks/          # Jupyter notebooks for research and analysis
├── environments/       # Custom RL environments
├── models/            # Trained models and checkpoints
├── data/              # Research datasets
├── experiments/       # Experiment configurations and results
└── reports/           # Research reports and documentation
```

## Getting Started

1. **Start JupyterLab:**
   ```bash
   cd /home/ubuntu/trading-stack
   ./scripts/start_jupyter.sh
   ```

2. **Access JupyterLab:**
   Open your browser and navigate to `http://localhost:8888`

3. **Run Example Notebooks:**
   - `01_environment_test.ipynb` - Test the custom trading environment
   - `02_finrl_training.ipynb` - Train a reinforcement learning agent

## Custom Environment

The `TradingStackEnvironment` class provides integration between FinRL and the trading stack infrastructure:

- **Data Source:** QuestDB for historical market data
- **Action Space:** Continuous actions for each symbol [-1, 1]
- **Observation Space:** Technical indicators + portfolio state
- **Reward Function:** Risk-adjusted portfolio returns

## Available Frameworks

### Reinforcement Learning
- **FinRL:** Financial reinforcement learning framework
- **Stable Baselines3:** State-of-the-art RL algorithms
- **Ray RLlib:** Distributed RL training

### Backtesting
- **VectorBT:** High-performance vectorized backtesting
- **NautilusTrader:** Event-driven backtesting with microsecond precision
- **Backtrader:** Flexible backtesting framework

### Portfolio Optimization
- **PyPortfolioOpt:** Modern portfolio theory implementation
- **CVXPY:** Convex optimization for portfolio construction

## Model Training

Models are automatically saved to the `models/` directory with timestamps and performance metrics. Use the evaluation callbacks to monitor training progress and prevent overfitting.

## Data Access

The environment automatically connects to:
- **QuestDB:** Real-time and historical market data
- **PostgreSQL:** Order and portfolio data
- **Redis:** Caching and session storage

## Performance Monitoring

Use the built-in performance metrics to evaluate strategy performance:
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor

## Best Practices

1. **Data Validation:** Always validate data quality before training
2. **Cross-Validation:** Use time-series cross-validation for financial data
3. **Risk Management:** Implement position sizing and stop-loss mechanisms
4. **Backtesting:** Test strategies on out-of-sample data
5. **Documentation:** Document all experiments and results
EOF

success "Research environment setup completed!"
echo ""
echo "Next steps:"
echo "1. Start JupyterLab: ./scripts/start_jupyter.sh"
echo "2. Open browser to: http://localhost:8888"
echo "3. Run example notebooks in research/notebooks/"
echo ""
echo "Available notebooks:"
echo "- 01_environment_test.ipynb - Test custom trading environment"
echo "- 02_finrl_training.ipynb - Train RL agents"
echo ""
echo "Research directory: $RESEARCH_DIR"
EOF

chmod +x /home/ubuntu/trading-stack/scripts/setup_research_environment.sh

