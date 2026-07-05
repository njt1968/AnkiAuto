import tkinter as tk
from tkinter import messagebox, simpledialog 
from PIL import Image, ImageTk
import os
import json
import threading
import time
import requests
import concurrent.futures
import shutil
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv
from google import genai
from google.genai import types
import azure.cognitiveservices.speech as speechsdk
import base64
import fal_client

# --- 1. CONFIGURATION LOADING ---
load_dotenv()


# LOAD SECRETS
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FAL_KEY = os.getenv("FAL_API_KEY")

# LOAD SETTINGS
CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "generation": {
        "image_mode": "mini",
        "azure_voice": "es-MX-JorgeNeural",
        "target_language": "Spanish (MX)",
        "image_quality": "medium",
        "cefr_lvl": "B2",
        "suggested_length": "20",
        "only_dalle": "n"
    },
    "paths": {
        "anki_media_folder": r"C:\Users\tutin\AppData\Roaming\Anki2\User 1\collection.media",
        "temp_folder": "temp_images",
        "output_csv": "ready_for_anki.csv"
    },
    "app_settings": {
        "sheet_name": "Anki_Inbox",
        "batch_limit": 50,
        "max_workers": 3
    }
}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"⚠️ {CONFIG_FILE} not found. Using defaults.")
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_FILE, "r") as f:
            user_config = json.load(f)
            # Merge with defaults
            for section, keys in DEFAULT_CONFIG.items():
                if section not in user_config:
                    user_config[section] = keys
                else:
                    for k, v in keys.items():
                        if k not in user_config[section]:
                            user_config[section][k] = v
            return user_config
    except Exception as e:
        print(f"❌ Error reading config.json: {e}")
        return DEFAULT_CONFIG

CFG = load_config()

# INITIALIZE CLIENTS
google_client = genai.Client(api_key=GOOGLE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# SHORTCUT VARIABLES
FINAL_FOLDER = CFG["paths"]["anki_media_folder"]
TEMP_FOLDER = CFG["paths"]["temp_folder"]
CSV_FILE = CFG["paths"]["output_csv"]
SHEET_NAME = CFG["app_settings"]["sheet_name"]
BATCH_LIMIT = CFG["app_settings"]["batch_limit"]
MAX_WORKERS = CFG["app_settings"]["max_workers"]
TARGET_LANGUAGE = CFG["generation"]["target_language"]
IMG_QUALITY = CFG["generation"]["image_quality"]
CEFR_LVL = CFG["generation"]["cefr_lvl"]
SUGGESTED_LENGTH = CFG["generation"]["suggested_length"]




# Ensure folders exist
for f in [FINAL_FOLDER, TEMP_FOLDER]:
    if not os.path.exists(f):
        os.makedirs(f)

# --- 2. GOOGLE SHEETS MANAGER ---
class SheetManager:
    def __init__(self, creds_file="credentials.json", sheet_name=SHEET_NAME):
        self.scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        self.creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, self.scope)
        self.client = gspread.authorize(self.creds)
        self.sheet = self.client.open(sheet_name).sheet1

    def fetch_pending_words(self, limit=BATCH_LIMIT):
        try:
            records = self.sheet.get_all_records()
            pending = []
            for i, row in enumerate(records):
                if len(pending) >= limit: break
                
                status = str(row.get("Status", "")).strip().lower()
                word = str(row.get("Word", "")).strip()
                
                if status not in ["done", "skipped"] and word:
                    pending.append({"text": word, "row_idx": i + 2})
            return pending
        except Exception as e:
            messagebox.showerror("Sheets Error", f"Could not read sheet: {e}")
            return []

    def mark_as_done(self, row_idx):
        try:
            self.sheet.update_cell(row_idx, 2, "Done")
        except Exception as e:
            print(f"❌ Failed to update sheet: {e}")
    def mark_as_skipped(self, row_idx):
        try:
            self.sheet.update_cell(row_idx, 2, "Skipped")
        except Exception as e:
            print(f"❌ Failed to update sheet: {e}")
                   

# --- 3. GENERATION LOGIC ---

# Update the arguments to accept 'instruction'
def generate_text_data(word, hint="None", instruction=None):
    # Base prompt
    base_prompt = f"""
    Task: Create a language flashcard for: "{word}" (Context: {hint}).
    Target Language: {TARGET_LANGUAGE}
    """

    # Add the user instruction if it exists
    if instruction:
        base_prompt += f"\nIMPORTANT USER INSTRUCTION: {instruction}\n"

    # Rest of the prompt remains the same
    base_prompt += f"""
    Output a SINGLE JSON object with these keys:
    - definition: STRICTLY just the definition IN TARGET LANGUAGE. No grammar notes.
    - sentence: A natural sentence using it in the Target Language at {CEFR_LVL} Level. Try not to exceed {SUGGESTED_LENGTH} words. 
    - translation: English translation of that sentence.
    - scenario: A vivid visual description for an artist IN ENGLISH. Describe lighting, subject, and environment.
    """
    
    try:
        response = google_client.models.generate_content(
            model="gemini-3.1-flash-lite", 
            contents=base_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        raw = response.text.strip()
        if raw.startswith("```"): raw = raw.split("\n", 1)[-1].rsplit("\n", 1)[0]
        parsed = json.loads(raw)
        if isinstance(parsed, list): parsed = parsed[0]
        return parsed
    except Exception as e:
        print(f"❌ Text Error ({word}): {e}")
        return None
    

def generate_image_fal(scenario, filename):
    # Optional: callback for real-time progress logs
    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"[fal-ai] {log['message']}")

    try:
        # Strictly following the docs you provided
        result = fal_client.subscribe(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": f"""
                2D vector illustration, flat design, SVG style, clean paths, no gradients.
                Minimalist, professional corporate illustration, thick strokes, bold outlines.
                White background. No text. 
                Scenario: {scenario}""",
                "image_size": "square",
                "num_inference_steps": 4,
                "num_images": 1,
                "enable_safety_checker": True,
                "output_format": "jpeg"
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        # The result schema shows an 'images' list containing a 'url'
        if result and "images" in result and len(result["images"]) > 0:
            image_url = result["images"][0]["url"]
            
            # Download the image to your local path
            img_response = requests.get(image_url)
            if img_response.status_code == 200:
                # Ensure TEMP_FOLDER is defined in your script
                path = os.path.join(TEMP_FOLDER, filename)
                with open(path, "wb") as f:
                    f.write(img_response.content)
                return path, None
            else:
                return None, f"Failed to download image: {img_response.status_code}"
        
        return None, "API call succeeded but no images were returned."

    except Exception as e:
        # This will catch any validation errors if the library schema changes
        print(f"Detailed Fal Error: {e}")
        return None, str(e)
            
def generate_image_fireworks(scenario, filename):
    try:
        url = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-1-schnell-fp8/text_to_image"
        
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "image/jpeg"
        }
        
        payload = {
            "prompt": f"""
            2D vector illustration, flat design, SVG style, clean paths, no gradients.
            Minimalist, professional corporate illustration, thick strokes, bold outlines.
            White background. No text. 
            Scenario: {scenario}""",
            "aspect_ratio": "1:1",
            "num_inference_steps": 10,
            "num_images": 1
        }

        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            path = os.path.join(TEMP_FOLDER, filename)
            with open(path, "wb") as f:
                f.write(response.content)
            return path, None
        else:
            print(f"API Error: {response.status_code}: {response.text}")
            return None, f"API Error: {response.status_code}"

    except Exception as e:
        return None, str(e)

def generate_image_dalle(scenario, filename):
    try:
        mode = CFG["generation"].get("image_mode", "standard").lower()
        
        safe_prompt = (
            f"Vector art illustration. White background. No text. "
            f"Object: {scenario}"
        )

        # --- MINI MODE (Bare Bones) ---
        if mode == "mini":
            # Strip ALL optional parameters to prevent 400 Errors
            response = openai_client.images.generate(
                model="gpt-image-1-mini", 
                prompt=safe_prompt,
                quality=f"{IMG_QUALITY}",
                n=1,
            )
            
            # Decode Base64
            if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                img_data = base64.b64decode(response.data[0].b64_json)
                path = os.path.join(TEMP_FOLDER, filename)
                with open(path, "wb") as f:
                    f.write(img_data)
                return path, None
            else:
                return None, "API returned no data"

        # --- STANDARD MODE (DALL-E 3) ---
        else:
            response = openai_client.images.generate(
                model="dall-e-3", 
                prompt=safe_prompt,
                size="1024x1024", 
                quality="standard",
                n=1
            )
            img_url = response.data[0].url
            img_data = requests.get(img_url).content
            path = os.path.join(TEMP_FOLDER, filename)
            with open(path, "wb") as f:
                f.write(img_data)
            return path, None

    except BadRequestError as e:
        print(f"⚠️ OpenAI Error: {e}")
        # If it's the Mini model, it's likely a parameter issue, not safety.
        if mode == "mini":
            return None, "Mini Model Error (Try Standard Mode)"
        return None, "Blocked: Content Filter"
        
    except Exception as e:
        print(f"❌ Image Error ({filename}): {e}")
        return None, f"Error: {str(e)[:20]}..."

def generate_image(scenario, safe_name, model):
    if model == "fal":
        return generate_image_fal(scenario, safe_name)
    else:
        return generate_image_dalle(scenario, safe_name)        


def generate_audio_azure(text, filename):
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        print("❌ CRITICAL: Azure Keys are missing from .env file!")
        return None

    try:
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = CFG["generation"]["azure_voice"]
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
        
        path = os.path.join(TEMP_FOLDER, filename)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return path
        return None
    except Exception as e:
        print(f"❌ Audio Exception ({filename}): {e}")
        return None

# --- 4. THE GUI APP ---

class ReviewApp:
    def __init__(self, root, sheet_manager):
        self.root = root
        self.sheet_mgr = sheet_manager
        
        mode = CFG["generation"].get("image_mode", "standard").upper()
        self.root.title(f"Anki Automator ({mode} MODE)")
        self.root.geometry("900x750")
        
        self.raw_data = self.sheet_mgr.fetch_pending_words()
        if not self.raw_data:
            messagebox.showinfo("Empty", "No pending words found!")
            root.destroy()
            return
            
        self.word_queue = [item["text"] for item in self.raw_data]
        self.cache = {} 
        self.current_word = None
        self.viewing_index = 0
        self.last_index = -1
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.after_id = None
        self.is_closing = False
        self.last_loaded_path = "" 

        self.setup_ui()
        self.start_prefetching()
        self.load_current_view()

    def setup_ui(self):
        top_frame = tk.Frame(self.root, bg="#ddd", height=40)
        top_frame.pack(side="top", fill="x")
        tk.Button(top_frame, text="🚪 Save & Exit", command=self.exit_app, bg="#ff6666", fg="white").pack(side="right", padx=10, pady=5)
        self.lbl_count = tk.Label(top_frame, text=f"Queue: {len(self.word_queue)} words", bg="#ddd")
        self.lbl_count.pack(side="left", padx=10)

        btn_frame = tk.Frame(self.root, bg="#f0f0f0", pady=15)
        btn_frame.pack(side="bottom", fill="x")

        self.btn_regen_text = tk.Button(btn_frame, text="🔄 Regen Text", command=self.regen_text, bg="#ffcccb", width=12)
        self.btn_regen_text.pack(side="left", padx=20)
        self.btn_regen_img = tk.Button(btn_frame, text="🎨 Regen Image", command=self.regen_image, bg="#ffd700", width=12)
        self.btn_regen_img.pack(side="left", padx=10)
        
        self.btn_skip = tk.Button(btn_frame, text="⏩ SKIP", command=self.skip_word, bg="#ffb366", width=12)
        self.btn_skip.pack(side="left", padx=10)
        
        self.lbl_status = tk.Label(btn_frame, text="Initializing...", bg="#f0f0f0", fg="gray", width=40, anchor="w")
        self.lbl_status.pack(side="left", padx=20)

        self.btn_approve = tk.Button(btn_frame, text="✅ APPROVE & NEXT", command=self.approve, bg="#90ee90", font=("Arial", 12, "bold"), padx=20)
        self.btn_approve.pack(side="right", padx=20)

        content = tk.Frame(self.root)
        content.pack(side="top", fill="both", expand=True)
        
        self.img_frame = tk.Frame(content, bg="#ddd", width=450)
        self.img_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.lbl_img = tk.Label(self.img_frame, text="Waiting...", bg="#ddd")
        self.lbl_img.pack(expand=True, fill="both")
        
        txt_frame = tk.Frame(content)
        txt_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # self.lbl_word = tk.Label(txt_frame, text="Loading...", font=("Arial", 22, "bold"), anchor="w")
        # self.lbl_word.pack(fill="x", pady=(0, 15))
        
        self.word_entry = tk.Entry(txt_frame, font=("Arial", 22, "bold"), bg="#f0f0f0", bd=0)
        self.word_entry.pack(fill="x", pady=(0, 15))
        # This saves the change when you click away or hit enter
        self.word_entry.bind("<FocusOut>", self.on_word_edited)
        self.word_entry.bind("<Return>", self.on_word_edited)
        
        
        self.entries = {}
        for f in ["Definition", "Sentence", "Translation", "Scenario"]:
            tk.Label(txt_frame, text=f, font=("Arial", 9, "bold"), anchor="w", fg="#555").pack(fill="x")
            box = tk.Text(txt_frame, height=3 if f != "Scenario" else 2, font=("Arial", 11), wrap="word", bg="#f9f9f9")
            box.pack(fill="x", pady=(0, 10))
            self.entries[f] = box

    def exit_app(self):
        if self.btn_approve.cget("state") == "disabled":
            if not messagebox.askyesno("Warning", "Save in progress. Exit anyway?"):
                return
        
        self.is_closing = True
        if self.after_id: self.root.after_cancel(self.after_id)
        try:
            for filename in os.listdir(TEMP_FOLDER):
                file_path = os.path.join(TEMP_FOLDER, filename)
                if os.path.isfile(file_path):
                    try: os.unlink(file_path)
                    except: pass
        except: pass
        self.root.destroy()

    def start_prefetching(self):
        self.update_status(f"🚀 Starting background workers...")
        for raw in self.word_queue:
            if "(" in raw:
                word = raw.split("(")[0].strip()
                hint = raw.split("(")[1].replace(")", "").strip()
            else:
                word = raw
                hint = "None"
            
            if word not in self.cache:
                self.cache[word] = {"status": "pending", "hint": hint}
            self.executor.submit(self.process_single_card, word, hint)

    def process_single_card(self, word, hint):
        if word not in self.cache: 
            return
        if "definition" not in self.cache[word]:
            self.update_status(f"📝 Writing text for '{word}'...")
            data = generate_text_data(word, hint)
            if data: self.cache[word].update(data)
        
        if "image_path" not in self.cache[word] and "image_error" not in self.cache[word]:
            scenario = self.cache[word].get("scenario", "")
            self.update_status(f"🎨 Painting '{word}'...")
            
            safe_name = "".join([c for c in word if c.isalnum()]) + f"_{int(time.time())}.png"
            
            # --- LOGIC: Use Fireworks for first 3, DALL-E for the rest ---
            # We find the position of the word in the original queue
            try:
                # Find index of the word by matching it back to the raw word_queue
                current_idx = next(i for i, raw in enumerate(self.word_queue) if raw.startswith(word))
                
                if current_idx < 3:
                    path, error = generate_image(scenario, safe_name, "fal")
                else:
                    path, error = generate_image(scenario, safe_name, "fal")
            except StopIteration:
                # Fallback to DALL-E if indexing fails
                path, error = generate_image(scenario, safe_name, "fal")
            # -------------------------------------------------------------

            if path: 
                self.cache[word]["image_path"] = path
                self.update_status(f"✨ Ready: '{word}'")
            elif error:
                self.cache[word]["image_error"] = error
                self.update_status(f"⚠️ Error: '{word}'")
                
    def load_current_view(self):
        if self.after_id: self.root.after_cancel(self.after_id)
        if self.is_closing: return

        if self.viewing_index >= len(self.word_queue):
            self.is_closing = True
            messagebox.showinfo("Done", "All cards reviewed!")
            self.root.destroy()
            return

        raw = self.word_queue[self.viewing_index]
        word = raw.split("(")[0].strip() if "(" in raw else raw
        
        # --- HEADER UPDATE ---
        if self.last_index != self.viewing_index:
            self.last_index = self.viewing_index
            self.current_word = word
            
            self.word_entry.delete(0, tk.END)
            self.word_entry.insert(0, word)
            
            self.lbl_count.config(text=f"Reviewing {self.viewing_index + 1} of {len(self.word_queue)}")
            for v in self.entries.values(): v.delete("1.0", tk.END)
            self.lbl_img.config(image="", text="Loading...")
            self.img_frame.config(bg="#ddd")
            self.last_loaded_path = ""

        # Use the word currently typed in the box for cache lookups
        lookup_word = self.word_entry.get().strip()
        data = self.cache.get(lookup_word, self.cache.get(word, {}))

        # --- TEXT UPDATE ---
        if "definition" in data:
            current_def = self.entries["Definition"].get("1.0", tk.END).strip()
            if current_def == "" or data.get("force_text_update", False):
                for k, v in self.entries.items():
                    val = data.get(k.lower(), "")
                    if v.get("1.0", tk.END).strip() != val:
                        v.delete("1.0", tk.END)
                        v.insert("1.0", val)
                if "force_text_update" in data: del data["force_text_update"]

        # --- IMAGE UPDATE (STRICT CHECKS) ---
        current_img_path = data.get("image_path")
        error_msg = data.get("image_error")
        current_lbl_text = self.lbl_img.cget("text")

        # Case A: Success (New Image)
        if current_img_path and getattr(self, "last_loaded_path", "") != current_img_path:
            self.show_image(current_img_path)
            self.last_loaded_path = current_img_path
            self.lbl_img.config(text="")
            self.img_frame.config(bg="#ddd")
            
        # Case B: Error (Only update if NOT already showing error)
        elif error_msg:
            # Check if we are ALREADY showing this specific error
            if error_msg not in current_lbl_text:
                self.lbl_img.config(image="", text=f"⚠️ {error_msg}\n\nChange Mode or Text", fg="red")
                self.img_frame.config(bg="#ffcccc")

        # Case C: Loading (Only update if NOT already loading)
        elif not current_img_path and not error_msg:
             if "Generating" not in current_lbl_text and "Loading" not in current_lbl_text:
                 self.lbl_img.config(image="", text="Generating...", fg="black")
                 self.img_frame.config(bg="#ddd")

        self.after_id = self.root.after(500, self.load_current_view)
        
    def approve(self):
        if "image_error" in self.cache.get(self.current_word, {}):
             messagebox.showerror("Blocked", "Image generation failed. Please regenerate before approving.")
             return
        
        if "image_path" not in self.cache.get(self.current_word, {}):
            messagebox.showwarning("Wait", "Image is still generating!")
            return

        # LOCK UI & STOP REFRESH LOOP
        self.btn_approve.config(state="disabled", text="Saving...")
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        # Pass the current index to the worker to ensure the correct row is updated
        idx_to_approve = self.viewing_index
        threading.Thread(target=self._approve_worker, args=(idx_to_approve,)).start()
        
    def _approve_worker(self, row_idx_in_queue):
        # final_sentence = self.entries["Sentence"].get("1.0", tk.END).strip()
        # safe_name = "".join([c for c in self.current_word if c.isalnum()]) + f"_{int(time.time())}.mp3"
        final_word = self.word_entry.get().strip() # Use the text currently in the box
        final_sentence = self.entries["Sentence"].get("1.0", tk.END).strip()
        safe_name = "".join([c for c in final_word if c.isalnum()]) + f"_{int(time.time())}.mp3"
        
        
        self.update_status(f"🎤 Generating Audio for '{self.current_word}'...")
        aud_path = generate_audio_azure(final_sentence, safe_name)
        aud_name = os.path.basename(aud_path) if aud_path else ""

        img_temp = self.cache[self.current_word]["image_path"]
        img_name = os.path.basename(img_temp)
        img_final = os.path.join(FINAL_FOLDER, img_name)
        
        try: shutil.move(img_temp, img_final)
        except: shutil.copy(img_temp, img_final)

        if aud_path:
            aud_final = os.path.join(FINAL_FOLDER, aud_name)
            try: shutil.move(aud_path, aud_final)
            except: shutil.copy(aud_path, aud_final)

        row_data = {
            "Target": final_word,
            "Definition": self.entries["Definition"].get("1.0", tk.END).strip(),
            "Sentence": final_sentence,
            "Translation": self.entries["Translation"].get("1.0", tk.END).strip(),
            "Scenario": self.entries["Scenario"].get("1.0", tk.END).strip(),
            "Image": f'<img src="{img_name}">',
            "Audio": f'[sound:{aud_name}]' if aud_name else ""
        }
        
        # --- CSV WRITE (COLON SEPARATOR, NO HEADER) ---
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=row_data.keys(), delimiter=':')
            writer.writerow(row_data)
        
        row_id = self.raw_data[row_idx_in_queue]["row_idx"]
        self.sheet_mgr.mark_as_done(row_id)

        print(f"✅ Approved: {self.current_word}")
        self.root.after(0, self._finish_approval)

    def _finish_approval(self):
        self.viewing_index += 1
        self.current_word = None
        for v in self.entries.values(): v.delete("1.0", tk.END)
        self.last_loaded_path = ""
        self.lbl_img.config(image="", text="Loading next...")
        self.btn_approve.config(state="normal", text="✅ APPROVE & NEXT")
        self.load_current_view()
    def skip_word(self):
        """Marks the current word as skipped in the sheet and moves to next."""
        if not self.current_word:
            return

        if messagebox.askyesno("Skip Word", f"Are you sure you want to skip '{self.current_word}'?"):
            self.btn_skip.config(state="disabled")
            threading.Thread(target=self._skip_worker).start()

    def _skip_worker(self):
        row_id = self.raw_data[self.viewing_index]["row_idx"]
        self.update_status(f"⏩ Skipping '{self.current_word}'...")
        
        # Update Google Sheets
        self.sheet_mgr.mark_as_skipped(row_id)
        
        print(f"⏩ Skipped: {self.current_word}")
        # Return to main thread to advance the UI
        self.root.after(0, self._finish_skip)

    def on_word_edited(self, event=None):
        new_word = self.word_entry.get().strip()
        
        if new_word and new_word != self.current_word:
            old_word = self.current_word
            
            # 1. Capture existing data before changing keys
            old_data = self.cache.get(old_word, {}).copy()
            
            # 2. Update tracking variable
            self.current_word = new_word 
            
            # 3. Transfer data to the new key
            # This ensures 'image_path' is carried over so the prefetcher sees it's 'done'
            self.cache[new_word] = old_data
            self.cache[new_word]["force_text_update"] = True
            
            self.update_status(f"✏️ Word updated. Image preserved.")
            
    def _finish_skip(self):
        self.btn_skip.config(state="normal")
        self.viewing_index += 1
        # Clear entries for next word
        for v in self.entries.values(): v.delete("1.0", tk.END)
        self.last_loaded_path = ""
        self.lbl_img.config(image="", text="Loading next...")
        self.load_current_view()
    # --- REGENERATION ---
    def regen_text(self):
        word = self.current_word
        hint = self.cache[word].get("hint", "None")
        
        # 1. Ask user for specific instructions
        instruction = simpledialog.askstring(
            "Regenerate Text", 
            f"Add specific instructions for '{word}'?\n(Leave empty for standard regen)",
            parent=self.root
        )
        
        # If user hit Cancel, abort
        if instruction is None: 
            return

        self.update_status(f"📝 Regenerating text for {word}...")
        self.btn_regen_text.config(state="disabled")
        
        # Pass the instruction to the worker
        threading.Thread(target=self._do_regen_text, args=(word, hint, instruction)).start()

    # Update the worker signature to accept instruction
    def _do_regen_text(self, word, hint, instruction):
        # Pass instruction to generation logic
        data = generate_text_data(word, hint, instruction)
        
        if data:
            data["force_text_update"] = True 
            self.cache[word].update(data)
            self.root.after(0, self._finish_text_regen)
        else:
            # Handle failure (optional: re-enable button)
            self.root.after(0, lambda: self.btn_regen_text.config(state="normal"))

    def _finish_text_regen(self):
        self.update_status("Text updated.")
        self.btn_regen_text.config(state="normal")

    def regen_image(self):
        scenario = self.entries["Scenario"].get("1.0", tk.END).strip()
        word = self.current_word
        self.update_status(f"🎨 Regenerating image for {word}...")
        self.lbl_img.config(image="", text="Regenerating...")
        self.img_frame.config(bg="#ddd") 
        
        self.btn_regen_img.config(state="disabled")
        self.btn_regen_text.config(state="disabled")
        threading.Thread(target=self._do_regen_image, args=(word, scenario)).start()

    def _do_regen_image(self, word, scenario):
        if "image_error" in self.cache[word]: del self.cache[word]["image_error"]
        old_path = self.cache[word].get("image_path")
        if old_path and os.path.exists(old_path):
            try: os.remove(old_path)
            except: pass
        
        safe_name = "".join([c for c in word if c.isalnum()]) + f"_{int(time.time())}.png"
        path, error = generate_image(scenario, safe_name, "fal")
        self.root.after(0, lambda: self.finish_regen(path, error, word))

    def finish_regen(self, path, error, word):
        if path:
            self.cache[word]["image_path"] = path
            self.show_image(path)
            self.last_loaded_path = path
            self.update_status("Regeneration complete.")
        elif error:
            self.cache[word]["image_error"] = error
            self.lbl_img.config(text=f"⚠️ {error}\nChange Text & Retry", fg="red")
            self.img_frame.config(bg="#ffcccc")
            self.update_status("❌ Image Blocked.")

        self.btn_regen_img.config(state="normal")
        self.btn_regen_text.config(state="normal")

    def show_image(self, path):
        try:
            load = Image.open(path)
            aspect = load.width / load.height
            new_w = 450
            new_h = int(new_w / aspect)
            load = load.resize((new_w, new_h), Image.Resampling.LANCZOS)
            render = ImageTk.PhotoImage(load)
            self.lbl_img.config(image=render)
            self.lbl_img.image = render
        except: pass

    def update_status(self, msg):
        self.root.after(0, lambda: self.lbl_status.config(text=msg))

if __name__ == "__main__":
    if not os.path.exists("credentials.json"):
        print("⚠️ Config not found. Using internal defaults.")
    root = tk.Tk()
    sheet_mgr = SheetManager(sheet_name=SHEET_NAME)
    app = ReviewApp(root, sheet_mgr)
    root.mainloop()