import tkinter as tk
from tkinter import messagebox
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
from openai import OpenAI
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 1. CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

google_client = genai.Client(api_key=GOOGLE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "gemini-2.0-flash"
FINAL_FOLDER = r"C:\Users\tutin\AppData\Roaming\Anki2\User 1\collection.media"
TEMP_FOLDER = "temp_images"
CSV_FILE = "ready_for_anki.csv"
SHEET_NAME = "Anki_Inbox"
MAX_WORKERS = 3 
BATCH_LIMIT = 50  # <--- Safety limit per session

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
        """Fetches rows where Status is NOT 'Done', up to a limit."""
        try:
            records = self.sheet.get_all_records()
            pending = []
            
            for i, row in enumerate(records):
                if len(pending) >= limit:
                    break 
                
                status = str(row.get("Status", "")).strip().lower()
                word = str(row.get("Word", "")).strip()
                
                if status != "done" and word:
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

# --- 3. GENERATION LOGIC ---

def generate_text_data(word, hint="None"):
    prompt = f"""
    Task: Create a language flashcard for: "{word}" (Context: {hint}).
    
    Output a SINGLE JSON object with these keys:
    - definition: STRICTLY just the definition. No grammar notes, no part-of-speech tags, no long explanations. Just the direct meaning.
    - sentence: A natural sentence using it.
    - translation: English translation of that sentence.
    - scenario: A short visual description for an artist. Do NOT describe any text, signs, words, or speech bubbles in the scene.
    """
    try:
        response = google_client.models.generate_content(
            model=MODEL_NAME, contents=prompt,
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

def generate_image_dalle(scenario, filename, forbidden_word):
    try:
        # STRONGER ANTI-TEXT PROMPT
        safe_prompt = (
            f"A minimal, 2D vector art illustration. Flat colors, white background. "
            f"CRITICAL RULE: The image must be completely text-free. "
            f"Do not include the word '{forbidden_word}'. "
            f"Do not include any text, letters, numbers, signs, labels, or speech bubbles of any kind. "
            f"SCENARIO: {scenario}"
        )
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=safe_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        img_url = response.data[0].url
        img_data = requests.get(img_url).content
        path = os.path.join(TEMP_FOLDER, filename)
        with open(path, "wb") as f:
            f.write(img_data)
        return path
    except Exception as e:
        print(f"❌ Image Error ({filename}): {e}")
        return None

# --- 4. THE GUI APP ---

class ReviewApp:
    def __init__(self, root, sheet_manager):
        self.root = root
        self.sheet_mgr = sheet_manager
        self.root.title("Anki Card Reviewer (Final)")
        self.root.geometry("900x750")
        
        # Load Data
        self.raw_data = self.sheet_mgr.fetch_pending_words()
        if not self.raw_data:
            messagebox.showinfo("Empty", "No pending words found in Sheets!")
            root.destroy()
            return
            
        self.word_queue = [item["text"] for item in self.raw_data]
        self.cache = {} 
        self.current_word = None
        self.viewing_index = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        self.after_id = None
        self.is_closing = False

        self.setup_ui()
        self.start_prefetching()
        self.load_current_view()

    def setup_ui(self):
        # --- TOP BAR (EXIT BUTTON) ---
        top_frame = tk.Frame(self.root, bg="#ddd", height=40)
        top_frame.pack(side="top", fill="x")
        
        tk.Button(top_frame, text="🚪 Save & Exit", command=self.exit_app, bg="#ff6666", fg="white", font=("Arial", 10, "bold")).pack(side="right", padx=10, pady=5)
        
        # Show count of batch
        self.lbl_count = tk.Label(top_frame, text=f"Queue: {len(self.word_queue)} words", bg="#ddd", font=("Arial", 10))
        self.lbl_count.pack(side="left", padx=10)

        # --- BOTTOM BAR (CONTROLS) ---
        btn_frame = tk.Frame(self.root, bg="#f0f0f0", pady=15)
        btn_frame.pack(side="bottom", fill="x")

        self.btn_regen_text = tk.Button(btn_frame, text="🔄 Regen Text", command=self.regen_text, bg="#ffcccb", width=12)
        self.btn_regen_text.pack(side="left", padx=20)
        
        self.btn_regen_img = tk.Button(btn_frame, text="🎨 Regen Image", command=self.regen_image, bg="#ffd700", width=12)
        self.btn_regen_img.pack(side="left", padx=10)
        
        self.lbl_status = tk.Label(btn_frame, text="Initializing...", bg="#f0f0f0", fg="gray", width=40, anchor="w")
        self.lbl_status.pack(side="left", padx=20)

        tk.Button(btn_frame, text="✅ APPROVE & NEXT", command=self.approve, bg="#90ee90", font=("Arial", 12, "bold"), padx=20).pack(side="right", padx=20)

        # --- MAIN CONTENT ---
        content = tk.Frame(self.root)
        content.pack(side="top", fill="both", expand=True)
        
        self.img_frame = tk.Frame(content, bg="#ddd", width=450)
        self.img_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.lbl_img = tk.Label(self.img_frame, text="Waiting...", bg="#ddd")
        self.lbl_img.pack(expand=True, fill="both")
        
        txt_frame = tk.Frame(content)
        txt_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.lbl_word = tk.Label(txt_frame, text="Loading...", font=("Arial", 22, "bold"), anchor="w")
        self.lbl_word.pack(fill="x", pady=(0, 15))
        
        self.entries = {}
        for f in ["Definition", "Sentence", "Translation", "Scenario"]:
            tk.Label(txt_frame, text=f, font=("Arial", 9, "bold"), anchor="w", fg="#555").pack(fill="x")
            box = tk.Text(txt_frame, height=3 if f != "Scenario" else 2, font=("Arial", 11), wrap="word", relief="flat", bg="#f9f9f9", highlightthickness=1)
            box.pack(fill="x", pady=(0, 10))
            self.entries[f] = box
    
    def exit_app(self):
        """Safely closes the app and cleans up temp files."""
        self.is_closing = True
        if self.after_id:
            self.root.after_cancel(self.after_id)
        
        # --- CLEANUP TEMP FOLDER ---
        print("🧹 Cleaning up temporary images...")
        try:
            for filename in os.listdir(TEMP_FOLDER):
                file_path = os.path.join(TEMP_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        except Exception as e:
            print(f"Cleanup warning: {e}")

        self.root.destroy()
        print("👋 Exited safely. Pending words saved for next time.")

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
        if "definition" not in self.cache[word]:
            self.update_status(f"📝 Writing text for '{word}'...")
            data = generate_text_data(word, hint)
            if data: self.cache[word].update(data)
        
        if "image_path" not in self.cache[word]:
            scenario = self.cache[word].get("scenario", "")
            self.update_status(f"🎨 Painting '{word}'...")
            safe_name = "".join([c for c in word if c.isalnum()]) + f"_{int(time.time())}.png"
            path = generate_image_dalle(scenario, safe_name, word)
            if path: 
                self.cache[word]["image_path"] = path
                self.update_status(f"✨ Finished '{word}'")

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
        self.current_word = word
        self.lbl_word.config(text=word)
        self.lbl_count.config(text=f"Reviewing {self.viewing_index + 1} of {len(self.word_queue)}")

        data = self.cache.get(word, {})

        # UPDATE TEXT FIELDS
        if "definition" in data:
            current_def = self.entries["Definition"].get("1.0", tk.END).strip()
            # Only update if field is empty OR forced by regen
            if current_def == "" or data.get("force_text_update", False):
                for k, v in self.entries.items():
                    v.delete("1.0", tk.END)
                    v.insert("1.0", data.get(k.lower(), ""))
                if "force_text_update" in data:
                    del data["force_text_update"] 

        # UPDATE IMAGE
        current_img_path = data.get("image_path")
        if current_img_path and getattr(self, "last_loaded_path", "") != current_img_path:
            self.show_image(current_img_path)
            self.last_loaded_path = current_img_path
            self.lbl_img.config(text="")
        elif not current_img_path:
             self.lbl_img.config(text="Generating...")

        self.after_id = self.root.after(500, self.load_current_view)

    def approve(self):
        if "image_path" not in self.cache[self.current_word]:
            messagebox.showwarning("Wait", "Image is still generating!")
            return

        temp_path = self.cache[self.current_word]["image_path"]
        filename = os.path.basename(temp_path)
        final_path = os.path.join(FINAL_FOLDER, filename)
        
        try:
            shutil.move(temp_path, final_path)
        except:
            shutil.copy(temp_path, final_path)

        data = {
            "Target": self.current_word,
            "Definition": self.entries["Definition"].get("1.0", tk.END).strip(),
            "Sentence": self.entries["Sentence"].get("1.0", tk.END).strip(),
            "Translation": self.entries["Translation"].get("1.0", tk.END).strip(),
            "Scenario": self.entries["Scenario"].get("1.0", tk.END).strip(),
            "Image": f'<img src="{filename}">'
        }
        
        file_exists = os.path.isfile(CSV_FILE)
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(data)
        
        row_id = self.raw_data[self.viewing_index]["row_idx"]
        threading.Thread(target=self.sheet_mgr.mark_as_done, args=(row_id,)).start()

        print(f"✅ Approved: {self.current_word}")
        
        self.viewing_index += 1
        for v in self.entries.values(): v.delete("1.0", tk.END)
        self.last_loaded_path = ""
        self.lbl_img.config(image="", text="Loading next...")
        self.load_current_view()

    # --- REGENERATION ---

    def regen_image(self):
        scenario = self.entries["Scenario"].get("1.0", tk.END).strip()
        word = self.current_word
        
        self.update_status(f"🎨 Regenerating image for {word}...")
        self.lbl_img.config(image="", text="Regenerating...")
        
        self.btn_regen_img.config(state="disabled")
        self.btn_regen_text.config(state="disabled")

        threading.Thread(target=self._do_regen_image, args=(word, scenario)).start()

    def _do_regen_image(self, word, scenario):
        # 1. DELETE OLD TEMP IMAGE (Garbage Collection)
        old_path = self.cache[word].get("image_path")
        if old_path and os.path.exists(old_path):
            try: 
                os.remove(old_path)
                print(f"🗑️ Deleted rejected image: {os.path.basename(old_path)}")
            except Exception as e:
                print(f"⚠️ Could not delete old image: {e}")

        # 2. GENERATE NEW ONE
        safe_name = "".join([c for c in word if c.isalnum()]) + f"_{int(time.time())}.png"
        path = generate_image_dalle(scenario, safe_name, word)
        
        if path:
            self.cache[word]["image_path"] = path
            self.root.after(0, lambda: self.finish_regen(path))

    def finish_regen(self, path):
        self.show_image(path)
        self.last_loaded_path = path
        self.update_status("Regeneration complete.")
        self.btn_regen_img.config(state="normal")
        self.btn_regen_text.config(state="normal")

    def regen_text(self):
        word = self.current_word
        hint = self.cache[word].get("hint", "None")
        
        self.update_status(f"📝 Regenerating text for {word}...")
        self.btn_regen_text.config(state="disabled")
        
        threading.Thread(target=self._do_regen_text, args=(word, hint)).start()

    def _do_regen_text(self, word, hint):
        data = generate_text_data(word, hint)
        if data:
            data["force_text_update"] = True 
            self.cache[word].update(data)
            self.root.after(0, self._finish_text_regen)

    def _finish_text_regen(self):
        self.update_status("Text updated.")
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
        print("❌ MISSING credentials.json! Please download from Google Cloud.")
    else:
        root = tk.Tk()
        sheet_mgr = SheetManager(sheet_name="Anki_Inbox")
        app = ReviewApp(root, sheet_mgr)
        root.mainloop()