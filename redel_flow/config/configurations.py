import os
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY") #optional, only needed if using the 'search' tool

if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY is not set. Please configure your .env file.")

MAXIMUM_RETRY = 1
