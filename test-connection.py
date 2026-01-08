import requests
import json
from google import genai
from google.genai import types

# --- CONFIGURATION ---
GOOGLE_API_KEY = "YOUR_GOOGLE_KEY_HERE"  # Paste your key here
client = genai.Client(api_key="AIzaSyDYnaTUMpbFZUc4i_Vy5PgdULC_HQHSrwk")

def test_anki():
    """Checks if Anki is open and AnkiConnect is listening."""
    try:
        response = requests.post("http://localhost:8765", json={
            "action": "version",
            "version": 6
        })
        print(f"✅ Anki Connect is working! Version: {response.json()['result']}")
    except Exception as e:
        print("❌ Anki Connect failed. Is Anki open?")

def test_gemini():
    """Checks if we can talk to Google's Brain."""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents="Say 'Hello Nicholas' in Spanish."
        )
        print(f"✅ Gemini is working! Response: {response.text.strip()}")
    except Exception as e:
        print(f"❌ Gemini failed: {e}")

if __name__ == "__main__":
    print("--- TESTING CONNECTIONS ---")
    test_anki()
    test_gemini()