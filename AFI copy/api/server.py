from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, WebSocket, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field
import json
import logging
import asyncio
import uvicorn
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import yaml
from enum import Enum
import sys
import os
from contextlib import asynccontextmanager

# Add the parent directory to Python path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

# Get secret from environment variable or use default
BOTPRESS_SECRET = os.environ.get("BOTPRESS_SECRET", "ianssantos.2005@mvndoco")

# Pydantic models
class SystemStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

class NotificationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BotpressConfig(BaseModel):
    bot_id: str
    api_token: str
    webhook_url: HttpUrl

class NotificationMessage(BaseModel):
    type: str
    title: str
    content: str
    priority: NotificationPriority
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BotpressMessage(BaseModel):
    text: Optional[str] = None
    conversation_id: Optional[str] = None
    user_id: Optional[str] = Field(default=None, alias="userId")
    type: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"  # Allow extra fields that aren't in the model

class MockOrderExecutor:
    def __init__(self):
        self.positions = []
        self.risk_manager = self.MockRiskManager()
    
    def get_performance_report(self):
        return {
            "success_rate": 0.85,
            "total_trades": 0,
            "profitable_trades": 0,
            "average_profit": 0,
            "max_drawdown": 0.12
        }
    
    class MockRiskManager:
        def get_risk_metrics(self):
            return {
                "current_exposure": 0.35,
                "max_drawdown": 0.12,
                "sharpe_ratio": 1.8
            }

class MockMarketAnalyzer:
    def get_regime_groups(self):
        return ["bullish", "neutral", "bearish"], {"regime": "bullish"}

class MockOrchestrator:
    def __init__(self):
        self.active_strategy = type('MockStrategy', (), {'__class__': type('MockClass', (), {'__name__': 'MomentumStrategy'})})()

class TradingSystemManager:
    def __init__(self):
        self.system = None
        self.state = SystemStatus.STARTING
        self.running_task: Optional[asyncio.Task] = None
        self.notification_queue = asyncio.Queue()
        self.websocket_clients: Set[WebSocket] = set()

    async def _broadcast_notification(self, notification: NotificationMessage):
        """Broadcast notification to all WebSocket clients"""
        dead_clients = set()
        for client in self.websocket_clients:
            try:
                await client.send_json(notification.dict())
            except Exception as e:
                logging.error(f"Failed to send notification: {e}")
                dead_clients.add(client)
        
        # Remove dead clients
        self.websocket_clients -= dead_clients

    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self.system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        try:
            # Get component statuses
            market_analyzer = self.system.components['market_analyzer']
            order_executor = self.system.components['order_executor']
            orchestrator = self.system.components['orchestrator']

            # Get current market analysis
            regime_groups, analysis = market_analyzer.get_regime_groups()
            
            return {
                "status": self.state,
                "market_regime": analysis.get('regime', 'Unknown'),
                "active_strategy": orchestrator.active_strategy.__class__.__name__ if orchestrator.active_strategy else None,
                "positions": len(order_executor.positions),
                "performance": order_executor.get_performance_report(),
                "risk_metrics": order_executor.risk_manager.get_risk_metrics(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logging.error(f"Error getting system status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def initialize_mock_system(self):
        """Initialize a mock trading system for testing"""
        class MockSystem:
            def __init__(self):
                self.components = {
                    'market_analyzer': MockMarketAnalyzer(),
                    'order_executor': MockOrderExecutor(),
                    'orchestrator': MockOrchestrator()
                }
        
        self.system = MockSystem()
        self.state = SystemStatus.RUNNING
        return {"message": "Mock trading system initialized"}

# Initialize system manager
system_manager = TradingSystemManager()

# Create FastAPI app
app = FastAPI(
    title="Trading Bot API",
    description="API for managing automated crypto trading system with Botpress integration",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    system_manager.initialize_mock_system()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Trading Bot API"}

@app.get("/system/status")
async def get_system_status():
    """Get current system status"""
    return await system_manager.get_system_status()

@app.post("/system/init")
async def initialize_system():
    """Manually initialize the mock trading system"""
    return system_manager.initialize_mock_system()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    system_manager.websocket_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        system_manager.websocket_clients.remove(websocket)

async def verify_botpress_token(authorization: Optional[str] = Header(None)):
    """Verify the Botpress authorization token"""
    if not authorization or authorization != BOTPRESS_SECRET:
        logging.warning("Unauthorized Botpress request received")
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

@app.post("/webhook/botpress")
async def botpress_webhook(request: Request, authorized: bool = Depends(verify_botpress_token)):
    """Handle incoming Botpress messages"""
    try:
        # Get the raw request body, handling empty bodies
        try:
            body_text = await request.body()
            logging.info(f"Raw request body: {body_text}")
            
            if not body_text:
                logging.warning("Empty request body received")
                return {"response": "Empty request received. Please send a valid JSON payload."}
                
            body = json.loads(body_text)
            logging.info(f"Received Botpress webhook: {body}")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in request body: {e}")
            return {"response": "Invalid JSON payload received."}
        
        # Try to parse as BotpressMessage, but don't fail if it doesn't match
        try:
            message = BotpressMessage(**body)
        except Exception as e:
            logging.warning(f"Could not parse Botpress message: {e}")
            message = None
        
        if not system_manager.system:
            return {"response": "Trading system is initializing. Please try again later. Use /system/init to initialize a mock system."}
        
        # Process commands if text is present
        if message and message.text:
            text = message.text.strip().lower()
            
            if text.startswith('/status'):
                # Current status response
                status = await system_manager.get_system_status()
                response = (
                    f"Trading System Status:\n"
                    f"State: {status['status']}\n"
                    f"Market Regime: {status['market_regime']}\n"
                    f"Active Positions: {status['positions']}\n"
                    f"Success Rate: {status['performance']['success_rate']:.1%}"
                )
                return {"response": response}
            
            elif text.startswith('/performance'):
                # Detailed performance metrics
                status = await system_manager.get_system_status()
                perf = status['performance']
                detailed_response = (
                    f"Performance Metrics:\n"
                    f"Success Rate: {perf['success_rate']:.1%}\n"
                    f"Total Trades: {perf['total_trades']}\n"
                    f"Profitable Trades: {perf['profitable_trades']}\n"
                    f"Average Profit: {perf['average_profit']:.2%}\n"
                    f"Max Drawdown: {perf['max_drawdown']:.2%}"
                )
                return {"response": detailed_response}
            
            elif text.startswith('/risk'):
                # Risk metrics
                status = await system_manager.get_system_status()
                risk = status['risk_metrics']
                risk_response = (
                    f"Risk Metrics:\n"
                    f"Current Exposure: {risk['current_exposure']:.2%}\n"
                    f"Max Drawdown: {risk['max_drawdown']:.2%}\n"
                    f"Sharpe Ratio: {risk['sharpe_ratio']:.2f}"
                )
                return {"response": risk_response}
            
            elif text.startswith('/help'):
                # Help message
                help_text = (
                    "Available Commands:\n"
                    "/status - Get current system status\n"
                    "/performance - Get detailed performance metrics\n"
                    "/risk - Get risk management metrics\n"
                    "/help - Show this help message"
                )
                return {"response": help_text}
        
        # Default response if no command matched
        status = await system_manager.get_system_status()
        response = (
            f"Trading System Status:\n"
            f"State: {status['status']}\n"
            f"Market Regime: {status['market_regime']}\n"
            f"Active Strategy: {status['active_strategy']}\n"
            f"Active Positions: {status['positions']}\n"
            f"Success Rate: {status['performance']['success_rate']:.1%}\n\n"
            f"Type /help for available commands."
        )
        
        return {"response": response}
    except Exception as e:
        logging.error(f"Webhook error: {e}", exc_info=True)
        return {"response": f"Error processing request: {str(e)}"}

@app.post("/")
async def root_post(request: Request):
    """Handle POST requests to the root path (for debugging)"""
    try:
        body = await request.json()
        logging.info(f"Received POST to root: {body}")
        return {"message": "This is the root endpoint. Did you mean to use /webhook/botpress?"}
    except Exception as e:
        logging.error(f"Root POST error: {e}")
        return {"error": str(e)}

@app.get("/webhook/botpress/test")
async def test_botpress_webhook():
    """Simple test endpoint for Botpress webhook"""
    return {
        "status": "ok",
        "message": "Botpress webhook test endpoint is working",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,  # Changed port to 8001 to avoid conflicts
        reload=True,
        log_level="info"
    )
