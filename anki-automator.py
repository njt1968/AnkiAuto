import json
import re
import requests
import base64
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ No API Key found! Check your .env file.")
# --- CONFIGURATION ---
GOOGLE_API_KEY = "YOUR_GOOGLE_KEY_HERE"  # <--- PASTE YOUR KEY AGAIN
DECK_NAME = "Nicholas_Immersion"         # The name of your Anki Deck
MODEL_NAME = "gemini-2.0-flash"          # Fast and free-tier friendly

# Initialize Google Client
client = genai.Client(api_key=f"{os.getenv("GOOGLE_API_KEY")}")

# --- ANKI CONNECT FUNCTIONS ---

def invoke_anki(action, params={}):
    """Helper to talk to Anki."""
    requestJson = json.dumps({"action": action, "version": 6, "params": params})
    try:
        response = requests.post("http://localhost:8765", data=requestJson).json()
        if len(response) != 2:
            raise Exception("Response has an unexpected number of fields.")
        if "error" not in response:
            raise Exception("Response is missing required error field.")
        if response["error"] is not None:
            raise Exception(response["error"])
        return response["result"]
    except Exception as e:
        print(f"❌ Anki Error ({action}): {e}")
        return None

def setup_anki():
    """Ensures the Deck and Note Type exist."""
    print("⚙️ Checking Anki setup...")
    invoke_anki("createDeck", {"deck": DECK_NAME})
    
    # Define our custom Note Type
    model_name = "AI_Immersion_Card"
    model_fields = ["TargetWord", "Definition", "Sentence", "Translation", "Scenario", "Image"]
    
    # Check if model exists, if not, create it
    existing_models = invoke_anki("modelNames")
    if model_name not in existing_models:
        print(f"🛠 Creating new Note Type: {model_name}")
        invoke_anki("createModel", {
            "modelName": model_name,
            "inOrderFields": model_fields,
            "css": ".card { font-family: arial; font-size: 20px; text-align: center; color: black; background-color: white; } img { max-width: 300px; }",
            "cardTemplates": [
                {
                    "Name": "Card 1",
                    "Front": "{{TargetWord}}",
                    "Back": "{{FrontSide}}<hr id=answer>{{Definition}}<br><br><i>{{Sentence}}</i><br><small>{{Translation}}</small><br><hr>{{Image}}<br><small>{{Scenario}}</small>"
                }
            ]
        })
    else:
        print(f"✅ Note Type '{model_name}' already exists.")

# --- AI PROCESSING ---

def process_word(raw_input):
    """Parses input and calls Gemini with a flexible 'Phrase/Sentence' mindset."""
    
    # 1. Regex Parse
    match = re.match(r"^(.*?)\s*(?:\((.*)\))?$", raw_input.strip())
    input_text = match.group(1)
    hint = match.group(2) if match.group(2) else "None"
    
    print(f"🧠 Processing: '{input_text}'...")

    # 2. The Updated Prompt (Handles Words, Phrases, and Sentences)
    prompt = f"""
    You are an expert language tutor. I am giving you a Spanish or Mandarin text snippet: "{input_text}".
    Context/Hint: "{hint}".
    
    Your task:
    1. Identify if this is a single word, a phrase, or a full sentence.
    2. If it's a WORD: Provide the definition.
    3. If it's a PHRASE/SENTENCE: Provide the meaning/intent (not a literal dictionary definition).
    
    Return a valid JSON object with these keys:
    - definition: The definition OR the meaning of the phrase.
    - sentence: If the input was just a word, create a new example sentence. If the input WAS a sentence, improve it or correct it if natural; otherwise, keep it.
    - translation: English translation of the sentence.
    - scenario: A short (2 sentence) English description of a visual scenario that encapsulates the core meaning.
    - image_prompt: A DALL-E style prompt for a minimal, vector-style cartoon of that scenario.
    """

    # 3. Call Gemini
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    return json.loads(response.text), input_text

# --- MAIN EXECUTION ---

def run_batch(word_list):
    setup_anki()
    
    for entry in word_list:
        try:
            # Step 1: Get Data from AI
            data, target_word = process_word(entry)
            
            # Step 2: Push to Anki
            note = {
                "deckName": DECK_NAME,
                "modelName": "AI_Immersion_Card",
                "fields": {
                    "TargetWord": target_word,
                    "Definition": data["definition"],
                    "Sentence": data["sentence"],
                    "Translation": data["translation"],
                    "Scenario": data["scenario"],
                    "Image": "" # Placeholder for Phase 3
                },
                "tags": ["auto-generated"]
            }
            
            result = invoke_anki("addNote", {"note": note})
            if result:
                print(f"✅ Added: {target_word}")
        except Exception as e:
            print(f"❌ Failed on '{entry}': {e}")

# --- YOUR INBOX ---
# Add your words here for testing!
my_words = [
    "Sobremesa (culture)",            # Word
    "Echar la mano (help)",           # Idiom/Phrase
    "No tengo vela en este entierro"  # Full Sentence
]

if __name__ == "__main__":
    run_batch(my_words)