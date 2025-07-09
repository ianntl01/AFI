import asyncio
import websockets
import httpx
import json
import logging

# Configuration
BOTPRESS_API = "https://api.botpress.cloud/v1/bots/3b2d2c97-42d9-43fc-bf8e-88e7a931e81c/converse/user_01JZMTE5Z560YGQ2H9DP4K8N9T"
BOTPRESS_TOKEN = "bp_pat_iq4GSkkCdx5MH6yhRjhkWBbJYKPDzyqSp37J"
WEBSOCKET_URL = "wss://9dcbbcc8d244.ngrok-free.app/ws"

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def forward_to_botpress(message: str):
    """Forward WebSocket message to Botpress API"""
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "type": "text",
                "text": message
            }
            headers = {
                "Authorization": f"Bearer {BOTPRESS_TOKEN}",
                "Content-Type": "application/json"
            }
            response = await client.post(BOTPRESS_API, json=payload, headers=headers)
            response.raise_for_status()
            logging.info("Message forwarded successfully to Botpress")
    except Exception as e:
        logging.error(f"Failed to forward message: {e}")

async def websocket_listener():
    """Connect to WebSocket and listen for messages"""
    while True:
        try:
            async with websockets.connect(WEBSOCKET_URL) as websocket:
                logging.info("Connected to WebSocket server")
                while True:
                    message = await websocket.recv()
                    logging.info(f"Received message: {message}")
                    await forward_to_botpress(message)
        except Exception as e:
            logging.error(f"WebSocket error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(websocket_listener())
    except KeyboardInterrupt:
        logging.info("Proxy service stopped")
