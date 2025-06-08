# conftest.py
import pytest
import requests
import os
import time
import uvicorn
from threading import Thread

# Import the FastAPI app instance from the corrected file
from main import app, sqlite_file_name

# --- Test Configuration ---
HOST = "127.0.0.1"
PORT = 8001
BASE_URL = f"http://{HOST}:{PORT}"

def run_server():
    """Function to run the Uvicorn server in a thread."""
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")

@pytest.fixture(scope="session")
def api_client():
    """
    A session-scoped fixture to start and stop the FastAPI server.
    - 'scope="session"' means this fixture will be set up once for the entire
      test session (all test files).
    - It yields the base URL of the running server.
    """
    # Clean up database file from previous runs if it exists
    if os.path.exists(sqlite_file_name):
        os.remove(sqlite_file_name)
    
    # Start the server in a daemon thread so it shuts down with the main thread
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for the server to be ready
    # In a real-world scenario, you might use a more robust health check
    time.sleep(2)
    
    yield BASE_URL
    
    # Teardown is not strictly necessary for a daemon thread, but good practice
    # The pytest process will exit, and the daemon thread will be terminated.

@pytest.fixture(autouse=True)
def reset_db_state(api_client):
    """
    A function-scoped fixture to reset the database before each test.
    - 'autouse=True' means this fixture will be automatically used by every
      test function, so we don't have to add it as a parameter everywhere.
    - It depends on the `api_client` fixture to ensure the server is running.
    """
    response = requests.post(f"{api_client}/testing/reset-database")
    assert response.status_code == 200, "Database reset failed"
    # No need to yield anything, its purpose is just the reset action.
