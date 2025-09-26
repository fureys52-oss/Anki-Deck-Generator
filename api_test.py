# api_test.py

import os
import requests
from dotenv import load_dotenv

# --- Configuration ---
# This script assumes it's in the same folder as your .env file
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash-latest"
TEST_PROMPT = "In one sentence, what is a large language model?"
# ---------------------

def run_test():
    """Runs a direct, simple test of the Gemini API key and endpoint."""
    print("--- Starting Gemini API Diagnostic Test ---")

    # 1. Load the environment variables from your .env file
    env_path = '.env'
    if not os.path.exists(env_path):
        print(f"\n[FATAL ERROR] Could not find the .env file in the current directory.")
        print("SOLUTION: Please ensure this script is in the same folder as your .env file.")
        return

    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("GEMINI_API_KEY")

    # 2. Validate the API Key
    if not api_key:
        print("\n[FATAL ERROR] GEMINI_API_KEY not found or is empty in your .env file.")
        print("SOLUTION: Please open your .env file and ensure the key is correctly pasted after 'GEMINI_API_KEY='")
        return

    print(f"Found API Key starting with: '{api_key[:4]}...'")

    # 3. Construct the exact URL and make the API call
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": TEST_PROMPT}]
        }]
    }

    print(f"\nAttempting to call API endpoint:\n{url.split('?')[0]}\n")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        # 4. Analyze the response
        if response.status_code == 200:
            print("[SUCCESS] The API call was successful!")
            print("\n--- TEST RESULT ---")
            print("Your API key and the endpoint are working correctly.")
            print("The problem is a subtle and hidden bug within the main application's code.")
            print(f"\nAPI Response: {response.json()['candidates'][0]['content']['parts'][0]['text'].strip()}")
        elif response.status_code == 404:
            print("[FAILURE] The API call failed with a 404 Not Found error.")
            print("\n--- TEST RESULT ---")
            print("This means the URL is incorrect OR the model name is not available to your key.")
            print("This points to a bug in the application's URL construction.")
            print(f"\nFull server response: {response.text}")
        elif response.status_code == 400:
             print("[FAILURE] The API call failed with a 400 Bad Request error.")
             print("\n--- TEST RESULT ---")
             print("This almost always means your API key is invalid or disabled.")
             print("SOLUTION: Please go to Google AI Studio, regenerate a new API key, and paste it into your .env file.")
             print(f"\nFull server response: {response.text}")
        else:
            print(f"[FAILURE] The API call failed with an unexpected status code: {response.status_code}")
            print("\n--- TEST RESULT ---")
            print("An unknown error occurred.")
            print(f"\nFull server response: {response.text}")

    except requests.exceptions.RequestException as e:
        print("[FATAL ERROR] A network error occurred.")
        print(f"Could not connect to the Google API server. Error: {e}")
        print("SOLUTION: Please check your internet connection and any firewall settings.")

if __name__ == "__main__":
    run_test()