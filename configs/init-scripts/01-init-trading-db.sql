-- Trading Stack Database Initialization Script
-- Creates tables for order management, portfolio tracking, and audit logs

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS risk;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set default search path
ALTER DATABASE trading SET search_path TO trading, risk, audit, public;

-- ============================================================================
-- TRADING SCHEMA - Core trading operations
-- ============================================================================

-- Instruments table
CREATE TABLE trading.instruments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(50) NOT NULL UNIQUE,
    exchange VARCHAR(50) NOT NULL,
    instrument_type VARCHAR(20) NOT NULL CHECK (instrument_type IN ('equity', 'crypto', 'option', 'future', 'forex')),
    base_currency VARCHAR(10),
    quote_currency VARCHAR(10),
    tick_size DECIMAL(20, 10),
    lot_size DECIMAL(20, 10),
    min_quantity DECIMAL(20, 10),
    max_quantity DECIMAL(20, 10),
    is_active BOOLEAN DEFAULT true,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for instruments
CREATE INDEX idx_instruments_symbol ON trading.instruments(symbol);
CREATE INDEX idx_instruments_exchange ON trading.instruments(exchange);
CREATE INDEX idx_instruments_type ON trading.instruments(instrument_type);
CREATE INDEX idx_instruments_active ON trading.instruments(is_active);

-- Strategies table
CREATE TABLE trading.strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    strategy_type VARCHAR(50) NOT NULL,
    description TEXT,
    parameters JSONB,
    risk_limits JSONB,
    is_active BOOLEAN DEFAULT true,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Orders table
CREATE TABLE trading.orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID REFERENCES trading.strategies(id),
    instrument_id UUID REFERENCES trading.instruments(id),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(20, 10) NOT NULL,
    price DECIMAL(20, 10),
    stop_price DECIMAL(20, 10),
    time_in_force VARCHAR(10) DEFAULT 'GTC' CHECK (time_in_force IN ('GTC', 'IOC', 'FOK', 'DAY')),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'submitted', 'partial', 'filled', 'cancelled', 'rejected')),
    filled_quantity DECIMAL(20, 10) DEFAULT 0,
    avg_fill_price DECIMAL(20, 10),
    commission DECIMAL(20, 10) DEFAULT 0,
    external_order_id VARCHAR(100),
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    submitted_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for orders
CREATE INDEX idx_orders_strategy ON trading.orders(strategy_id);
CREATE INDEX idx_orders_instrument ON trading.orders(instrument_id);
CREATE INDEX idx_orders_status ON trading.orders(status);
CREATE INDEX idx_orders_created_at ON trading.orders(created_at);
CREATE INDEX idx_orders_external_id ON trading.orders(external_order_id);

-- Executions table (fills)
CREATE TABLE trading.executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID REFERENCES trading.orders(id),
    execution_id VARCHAR(100) NOT NULL,
    quantity DECIMAL(20, 10) NOT NULL,
    price DECIMAL(20, 10) NOT NULL,
    commission DECIMAL(20, 10) DEFAULT 0,
    liquidity_flag VARCHAR(10) CHECK (liquidity_flag IN ('maker', 'taker')),
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for executions
CREATE INDEX idx_executions_order ON trading.executions(order_id);
CREATE INDEX idx_executions_executed_at ON trading.executions(executed_at);
CREATE UNIQUE INDEX idx_executions_unique ON trading.executions(execution_id, order_id);

-- Positions table
CREATE TABLE trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID REFERENCES trading.strategies(id),
    instrument_id UUID REFERENCES trading.instruments(id),
    quantity DECIMAL(20, 10) NOT NULL DEFAULT 0,
    avg_price DECIMAL(20, 10),
    unrealized_pnl DECIMAL(20, 10) DEFAULT 0,
    realized_pnl DECIMAL(20, 10) DEFAULT 0,
    last_price DECIMAL(20, 10),
    market_value DECIMAL(20, 10),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(strategy_id, instrument_id)
);

-- Create indexes for positions
CREATE INDEX idx_positions_strategy ON trading.positions(strategy_id);
CREATE INDEX idx_positions_instrument ON trading.positions(instrument_id);
CREATE INDEX idx_positions_updated_at ON trading.positions(updated_at);

-- ============================================================================
-- RISK SCHEMA - Risk management and monitoring
-- ============================================================================

-- Risk limits table
CREATE TABLE risk.limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID REFERENCES trading.strategies(id),
    limit_type VARCHAR(50) NOT NULL,
    limit_value DECIMAL(20, 10) NOT NULL,
    current_value DECIMAL(20, 10) DEFAULT 0,
    breach_action VARCHAR(20) DEFAULT 'alert' CHECK (breach_action IN ('alert', 'block', 'liquidate')),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk events table
CREATE TABLE risk.events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID REFERENCES trading.strategies(id),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    message TEXT NOT NULL,
    data JSONB,
    acknowledged BOOLEAN DEFAULT false,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for risk events
CREATE INDEX idx_risk_events_strategy ON risk.events(strategy_id);
CREATE INDEX idx_risk_events_type ON risk.events(event_type);
CREATE INDEX idx_risk_events_severity ON risk.events(severity);
CREATE INDEX idx_risk_events_created_at ON risk.events(created_at);

-- ============================================================================
-- AUDIT SCHEMA - Audit trails and compliance
-- ============================================================================

-- Audit log table
CREATE TABLE audit.log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(100) NOT NULL,
    operation VARCHAR(10) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    record_id UUID NOT NULL,
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for audit log
CREATE INDEX idx_audit_log_table ON audit.log(table_name);
CREATE INDEX idx_audit_log_operation ON audit.log(operation);
CREATE INDEX idx_audit_log_record_id ON audit.log(record_id);
CREATE INDEX idx_audit_log_changed_at ON audit.log(changed_at);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers to relevant tables
CREATE TRIGGER update_instruments_updated_at BEFORE UPDATE ON trading.instruments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON trading.strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON trading.orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON trading.positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_limits_updated_at BEFORE UPDATE ON risk.limits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function for audit logging
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit.log (table_name, operation, record_id, old_values)
        VALUES (TG_TABLE_NAME, TG_OP, OLD.id, row_to_json(OLD));
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit.log (table_name, operation, record_id, old_values, new_values)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(OLD), row_to_json(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit.log (table_name, operation, record_id, new_values)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(NEW));
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers to critical tables
CREATE TRIGGER audit_orders AFTER INSERT OR UPDATE OR DELETE ON trading.orders
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_executions AFTER INSERT OR UPDATE OR DELETE ON trading.executions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_positions AFTER INSERT OR UPDATE OR DELETE ON trading.positions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert common instruments
INSERT INTO trading.instruments (symbol, exchange, instrument_type, base_currency, quote_currency, tick_size, lot_size, min_quantity) VALUES
('AAPL', 'NASDAQ', 'equity', 'USD', 'USD', 0.01, 1, 1),
('GOOGL', 'NASDAQ', 'equity', 'USD', 'USD', 0.01, 1, 1),
('MSFT', 'NASDAQ', 'equity', 'USD', 'USD', 0.01, 1, 1),
('TSLA', 'NASDAQ', 'equity', 'USD', 'USD', 0.01, 1, 1),
('NVDA', 'NASDAQ', 'equity', 'USD', 'USD', 0.01, 1, 1),
('BTC-USD', 'COINBASE', 'crypto', 'BTC', 'USD', 0.01, 0.00000001, 0.001),
('ETH-USD', 'COINBASE', 'crypto', 'ETH', 'USD', 0.01, 0.000001, 0.001),
('BTCUSDT', 'BINANCE', 'crypto', 'BTC', 'USDT', 0.01, 0.00000001, 0.001),
('ETHUSDT', 'BINANCE', 'crypto', 'ETH', 'USDT', 0.01, 0.000001, 0.001);

-- Create default strategy
INSERT INTO trading.strategies (name, strategy_type, description, parameters) VALUES
('default', 'manual', 'Default manual trading strategy', '{"max_position_size": 10000, "max_daily_loss": 1000}');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA risk TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA risk TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO trading_user;

