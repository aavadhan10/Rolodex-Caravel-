from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

# Retrieve the Claude API key from environment variables
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

# You can also add other environment variables here if needed
# For example, if you have a base URL or any other configuration settings:
# CLAUDE_API_URL = os.getenv('CLAUDE_API_URL')

# Optional: Raise an error if the API key is not found
if not CLAUDE_API_KEY:
    raise ValueError("Claude API key not found in environment variables.")
