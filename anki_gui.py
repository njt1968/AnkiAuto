import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import os
import json
import threading
import requests  # <--- Make sure this is imported
from openai import OpenAI  # <--- New import
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 1. CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"

IMAGE_FOLDER = "final_images"
CSV_FILE = "ready_for_anki.csv"

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# --- 2. BACKEND LOGIC (AI & Images) ---

def generate_text_data(word, hint="None"):
    """Calls Gemini to get the linguistic data."""
    print(f"   ↳ 🧠 Asking Gemini about '{word}'...")
    
    prompt = f"""
    Task: Create a language flashcard for: "{word}" (Context: {hint}).
    
    Output a SINGLE JSON object (not a list) with these keys:
    - definition: Meaning of the word/phrase.
    - sentence: A natural sentence using it.
    - translation: English translation of that sentence.
    - scenario: A short visual description for an artist.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        # --- CLEANING THE RESPONSE ---
        raw_text = response.text.strip()
        
        # Remove markdown fences if present (e.g., ```json ... ```)
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]  # Remove first line
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3] # Remove last 3 chars

        parsed = json.loads(raw_text)

        # Handle case where AI returns a list [ {data} ] instead of {data}
        if isinstance(parsed, list):
            if len(parsed) > 0:
                return parsed[0] # Grab the first item
            else:
                return None # Empty list
        
        return parsed

    except Exception as e:
        print(f"❌ Error parsing AI response: {e}")
        print(f"Raw response was: {response.text}") # Helps debugging
        return None
# --- REPLACEMENT FUNCTIONS ---

def generate_image_real(scenario, filename):
    """
    REAL FUNCTION: Calls DALL-E 3 to generate the cartoon.
    """
    print(f"   ↳ 🎨 Painting with DALL-E 3: {scenario[:30]}...")
    
    # Initialize OpenAI Client (It automatically looks for OPENAI_API_KEY in .env)
    openai_client = OpenAI()

    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=f"A minimal, 2D vector art illustration. Flat colors, white background. No text. {scenario}",
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        
        # Download the image
        img_data = requests.get(image_url).content
        path = os.path.join(IMAGE_FOLDER, filename)
        with open(path, "wb") as f:
            f.write(img_data)
            
        return path

    except Exception as e:
        print(f"❌ Image Gen Failed: {e}")
        return generate_image_mock(f"ERROR: {e}", filename)
    
# Keep a backup mock just in case the API fails (e.g., billing issues)
def generate_image_mock(scenario, filename):
    img = Image.new('RGB', (400, 300), color=(200, 100, 100)) # Red for error
    d = ImageDraw.Draw(img)
    d.text((10, 10), "IMAGE GEN FAILED", fill=(255, 255, 255))
    path = os.path.join(IMAGE_FOLDER, filename)
    img.save(path)
    return path

# --- 3. THE GUI CLASS ---

class ReviewApp:
    def __init__(self, root, word_list):
        self.root = root
        self.root.title("Anki Card Reviewer")
        self.root.geometry("800x600")
        
        self.word_queue = word_list
        self.current_data = {}
        self.current_word = ""
        self.current_hint = ""
        
        # UI Layout
        self.setup_ui()
        
        # Start first card
        self.load_next_card()
    
    def setup_ui(self):
        # --- 1. BUTTONS (Pack these FIRST so they stick to the bottom) ---
        btn_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        btn_frame.pack(side="bottom", fill="x")

        # Define Buttons
        # Note: On Mac, 'bg' color might not show up on standard buttons. 
        # If buttons look plain, that is an OS limitation, but they will work.
        tk.Button(btn_frame, text="🔄 Regen Text", command=self.regen_text, bg="#ffcccb", width=15).pack(side="left", padx=20)
        tk.Button(btn_frame, text="🎨 Regen Image", command=self.regen_image, bg="#ffd700", width=15).pack(side="left", padx=10)
        tk.Button(btn_frame, text="✅ APPROVE", command=self.approve, bg="#90ee90", font=("Arial", 12, "bold")).pack(side="right", padx=20)

        # --- 2. MAIN CONTENT AREA (Fills the rest of the space) ---
        content_frame = tk.Frame(self.root)
        content_frame.pack(side="top", fill="both", expand=True)

        # Left Side: Image
        self.image_frame = tk.Frame(content_frame, bg="#ddd", width=400)
        self.image_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.img_label = tk.Label(self.image_frame, text="Loading Image...", bg="#ddd")
        self.img_label.pack(expand=True, fill="both")

        # Right Side: Text Fields
        self.text_frame = tk.Frame(content_frame)
        self.text_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Word Header
        self.lbl_word = tk.Label(self.text_frame, text="Target Word", font=("Arial", 18, "bold"), wraplength=300)
        self.lbl_word.pack(pady=(0, 10), anchor="w")

        # Editable Fields
        self.entries = {}
        fields = ["Definition", "Sentence", "Translation", "Scenario"]
        
        for f in fields:
            lbl = tk.Label(self.text_frame, text=f, font=("Arial", 9, "bold"), anchor="w", fg="#555")
            lbl.pack(fill="x", pady=(5, 0))
            
            # Use a slightly shorter height (2) to ensure it fits on smaller screens
            txt = tk.Text(self.text_frame, height=2, font=("Arial", 11), wrap="word", relief="flat", bg="#f9f9f9", highlightthickness=1, highlightbackground="#ccc")
            txt.pack(fill="x", pady=(0, 5))
            self.entries[f] = txt
            
    def load_next_card(self):
        if not self.word_queue:
            messagebox.showinfo("Done", "All words reviewed!")
            self.root.destroy()
            return

        raw = self.word_queue.pop(0)
        # Parse Hint
        if "(" in raw:
            self.current_word = raw.split("(")[0].strip()
            self.current_hint = raw.split("(")[1].replace(")", "").strip()
        else:
            self.current_word = raw
            self.current_hint = "None"
            
        self.lbl_word.config(text=self.current_word)
        
        # Trigger Generation in Thread (so UI doesn't freeze)
        threading.Thread(target=self.run_generation).start()

    def run_generation(self):
        # 1. Generate Text
        data = generate_text_data(self.current_word, self.current_hint)
        if not data: return
        
        # Update UI with Text
        self.root.after(0, lambda: self.fill_fields(data))
        
        # 2. Generate Image
        self.run_image_gen(data['scenario'])

    def run_image_gen(self, scenario):
        filename = f"{self.current_word.replace(' ', '_')}.png"
        path = generate_image_real(scenario, filename)
        
        # Update UI with Image
        self.root.after(0, lambda: self.show_image(path))
        self.current_data['image_path'] = path

    def fill_fields(self, data):
        self.current_data.update(data)
        for key, widget in self.entries.items():
            widget.delete("1.0", tk.END)
            widget.insert("1.0", data.get(key.lower(), ""))

    def show_image(self, path):
        load = Image.open(path)
        load = load.resize((400, 300))
        render = ImageTk.PhotoImage(load)
        self.img_label.config(image=render, text="")
        self.img_label.image = render

    def regen_text(self):
        self.run_generation()

    def regen_image(self):
        # Grab current scenario from the text box in case user edited it
        current_scenario = self.entries["Scenario"].get("1.0", tk.END).strip()
        threading.Thread(target=self.run_image_gen, args=(current_scenario,)).start()

    def approve(self):
        # Gather final data from text boxes (allows for manual edits)
        final_data = {
            "Target": self.current_word,
            "Definition": self.entries["Definition"].get("1.0", tk.END).strip(),
            "Sentence": self.entries["Sentence"].get("1.0", tk.END).strip(),
            "Translation": self.entries["Translation"].get("1.0", tk.END).strip(),
            "Scenario": self.entries["Scenario"].get("1.0", tk.END).strip(),
            "Image": f'<img src="{os.path.basename(self.current_data["image_path"])}">'
        }
        
        # Save to CSV
        file_exists = os.path.isfile(CSV_FILE)
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=final_data.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(final_data)
            
        print(f"Saved: {self.current_word}")
        self.load_next_card()

# --- 4. RUNNER ---

if __name__ == "__main__":
    # Load words from file
    if not os.path.exists("input_words.txt"):
        with open("input_words.txt", "w") as f:
            f.write("Gato (animal)\nBanco (seat)")
            
    with open("input_words.txt", "r") as f:
        words = [line.strip() for line in f if line.strip()]

    root = tk.Tk()
    app = ReviewApp(root, words)
    root.mainloop()