# rp_handler.py
import runpod
import sys
import os
from pathlib import Path
import logging
from loguru import logger

# Configure detailed logging
logger.add("handler.log", rotation="500 MB")

# Ensure we're in the correct directory and add to path
WORKSPACE_DIR = Path(__file__).parent.absolute()
UI_DIR = WORKSPACE_DIR / "ui"
sys.path.append(str(UI_DIR))

logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")
logger.info(f"UI directory contents: {os.listdir('ui')}")

# Import the Flask app with error handling
try:
    logger.info("Attempting to import Flask app")
    from app import app
    logger.success("Flask app imported successfully")
except Exception as e:
    logger.error(f"Failed to import Flask app: {str(e)}")
    raise

def handler(event):
    """
    Handler for RunPod serverless requests
    Processes both GET and POST requests to Flask endpoints
    """
    try:
        input_data = event.get('input', {})
        endpoint = input_data.get('endpoint', '')
        method = input_data.get('method', 'GET')
        data = input_data.get('data', {})
        
        logger.info(f"Processing request: endpoint={endpoint}, method={method}, data={data}")
        
        with app.test_client() as client:
            if method == 'GET':
                response = client.get(f'/{endpoint}')
            elif method == 'POST':
                response = client.post(f'/{endpoint}', json=data)
            else:
                error_msg = f"Unsupported HTTP method: {method}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            try:
                response_data = response.get_json()
            except:
                response_data = response.data.decode('utf-8')
            
            logger.success(f"Request processed successfully: {response_data}")
            return {
                "status": response.status_code,
                "data": response_data
            }
            
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

if __name__ == "__main__":
    logger.info("Starting RunPod handler")
    try:
        # Ensure CUDA is available
        import torch
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Start the serverless handler
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        logger.error(f"Failed to start handler: {str(e)}", exc_info=True)
        raise