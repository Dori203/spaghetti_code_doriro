# rp_handler.py
import runpod
import sys
import os
import logging
import base64

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def print_directory_contents():
    """Print current directory structure for debugging"""
    logger.debug(f"Current working directory: {os.getcwd()}")
    logger.debug(f"Directory contents: {os.listdir('.')}")
    if os.path.exists('ui'):
        logger.debug(f"UI directory contents: {os.listdir('ui')}")


try:
    print_directory_contents()
    logger.debug("Setting up Python path")
    sys.path.append('ui')

    logger.debug("Importing Flask app")
    from ui.app import app

    logger.debug("Flask app imported successfully")

except Exception as e:
    logger.error(f"Failed to start up: {str(e)}", exc_info=True)
    raise


def handler(event):
    """Handler for RunPod serverless requests"""
    try:
        input_data = event.get('input', {})
        endpoint = input_data.get('endpoint', '')
        method = input_data.get('method', 'GET')
        data = input_data.get('data', {})

        logger.debug(f"Processing request: endpoint={endpoint}, method={method}, data={data}")

        with app.test_client() as client:
            if method == 'GET':
                response = client.get(f'/{endpoint}')
            elif method == 'POST':
                response = client.post(f'/{endpoint}', json=data)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}

            print(f"Response type: {type(response)}")
            print(f"Response mimetype: {response.mimetype}")
            print(f"Response data size: {len(response.data) if response.data else 0}")

            # Handle potential binary response
            print("before mimetype")
            if response.mimetype in ['model/gltf-binary', 'application/octet-stream']:
                print("Received binary data")
                logger.debug("Received binary data")
                binary_data = response.data
                print("After binary_data")
                base64_data = base64.b64encode(binary_data).decode('utf-8')
                print(f"Encoded data size: {len(base64_data)}")
                encoded_response = {
                    "status": response.status_code,
                    "data": base64_data,
                    "is_binary": True,
                    "mimetype": response.mimetype
                }
                print(f"Final response size: {len(str(encoded_response))}")
                return encoded_response
                # return {
                #     "status": response.status_code,
                #     "data": base64_data,
                #     "is_binary": True,
                #     "mimetype": response.mimetype
                # }
            else:
                try:
                    response_data = response.get_json()
                except:
                    try:
                        response_data = response.data.decode('utf-8')
                    except UnicodeDecodeError:
                        # Fallback for any other binary data
                        response_data = base64.b64encode(response.data).decode('utf-8')

            return {
                "status": response.status_code,
                "data": response_data
            }

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    logger.info("Starting RunPod handler")
    try:
        # Check CUDA
        import torch

        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        runpod.serverless.start({"handler": handler})
    except Exception as e:
        logger.error(f"Failed to start handler: {str(e)}", exc_info=True)
        raise