import os
import re
import json
import time
from google import genai
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SPREADSHEET_NAME = "Anki Staging"
INPUT_FILE_NAME = "to_stg.txt"  

# --- SETUP GEMINI ---
google_client = genai.Client(api_key=GOOGLE_API_KEY)

# --- FUNCTIONS ---

def clean_text(file_path):
    """
    Parses the raw text file and extracts only the user's bookmarked words.
    Removes page numbers, metadata headers, source tags, and blank lines.
    """
    cleaned_items = []
    
    # Regex patterns to identify noise
    # Matches "", "Page 10 | Highlight", "kindle", or standalone numbers
    patterns_to_remove = [
        r"^\\",      # Source tags
        r"^Page\s+\d+\s*\|\s*Highlight", # Kindle header
        r"^kindle\s*$",             # Kindle footer
        r"^\d+$",                   # Standalone page numbers
        r"^--- PAGE \d+ ---"        # Page delimiters
    ]
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        
        # 1. Skip empty lines
        if not line:
            continue
            
        # 2. Check against noise patterns
        is_noise = False
        for pattern in patterns_to_remove:
            if re.search(pattern, line, re.IGNORECASE):
                is_noise = True
                break
        
        if is_noise:
            continue
            
        # 3. Clean specific artifacts from the line itself (if mixed)
        # Sometimes source tags might be on the same line as text in copy-pastes
        line = re.sub(r"\\", "", line).strip()
        
        # 4. Final filter: If line is very short or looks like just punctuation/symbols
        if len(line) < 2 and not line.isalpha():
            continue
            
        cleaned_items.append(line)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(cleaned_items))

def analyze_with_gemini(word_list):
    """
    Sends the list of words to Gemini to generate context/meanings.
    """
    print(f"Sending {len(word_list)} items to Gemini for analysis...")
    
    # We define a structured prompt to ensure easy parsing later
    prompt = f"""
    You are a helpful assistant for a Spanish language learner.
    I will provide a list of Spanish words or phrases extracted from a book.
    
    Your task:
    0. Remove all extraneous markings (Page numbers, source tags, etc.) from the input.
    1. Analyze each word/phrase.
    2. Identify if the word has multiple common meanings or requires context (e.g., 'banco' can be bench or bank).
    3. If it has multiple meanings or is ambiguous, identify the MOST LIKELY meaning based on general usage.
    4. Revise and make sure the output has at most one definition per word.
    5. Return the data in a strict JSON list format.
    
    Input List:
    {word_list}
    
    Output Format (JSON only):
    [
      {{"word": "Spanish Word (english definition)",""}}
      {{"word": "Unambiguous Word"}} 
    ]
    
    Rules:
    - If the word is straightforward (e.g. "rojo"), or if it is an idiomatic phrase which requires no disambiguation, leave "english definition" empty.
    - If the word is ambiguous (e.g. "cólera" -> anger vs cholera), put the most likely definition in parentheses (e.g. "(anger)").
    - Do not include markdown code blocks ```json ... ```, just the raw JSON string.
    """

    try:
        response = google_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )
        clean_response = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_response)
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return []

def save_to_sheets(data):
    """
    Appends the processed data to the Google Sheet.
    """
    print("Connecting to Google Sheets...")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    
    try:
        sheet = client.open(SPREADSHEET_NAME).sheet1
        
        # Prepare rows: [Word, Meaning]
        rows_to_add = []
        for item in data:
            rows_to_add.append([item.get('word'), item.get('meaning')])
            
        if rows_to_add:
            sheet.append_rows(rows_to_add)
            print(f"Successfully added {len(rows_to_add)} rows to '{SPREADSHEET_NAME}'.")
        else:
            print("No data to add.")
            
    except Exception as e:
        print(f"Error saving to Sheets: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Check if input file exists
    if not os.path.exists(INPUT_FILE_NAME):
        print(f"Error: '{INPUT_FILE_NAME}' not found. Please create this file with your raw text.")
    else:
        # 2. Clean Data
        cleaned_words = clean_text(INPUT_FILE_NAME)
        print(f"Extracted {len(cleaned_words)} unique words/phrases.")
        
        if cleaned_words:
            # 3. Analyze with Gemini
            # If list is huge, you might want to slice it (e.g. cleaned_words[:50])
            analyzed_data = analyze_with_gemini(cleaned_words)
            
            # 4. Save to Sheets
            if analyzed_data:
                save_to_sheets(analyzed_data)
        else:
            print("No valid words found after cleaning.")