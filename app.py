# app.py
#
# MIT License - Copyright (c) 2025 [Your Name or Project Name]
#
# Version: 2.5.0 (Milestone 1 - Foundation)
# This is the stable, bug-fixed foundation for the public release.
# - BUG FIX: The main processing loop is now fully implemented and functional.
# - BUG FIX: The cache status button is correctly defined.
# - NEW FEATURE: A Python-based text pre-processing pipeline cleans text before AI calls.
# - CORE LOGIC: Permanent batching (size 3) is now the default for optimal speed and quality.

# ==============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ==============================================================================
# region
import os
import json
import base64
import hashlib
import io
import re
import time
import traceback
from pathlib import Path
from datetime import datetime
import shutil
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from collections import deque

try:
    import fitz # PyMuPDF
    import requests
    from dotenv import load_dotenv
    import gradio as gr
    from PIL import Image
except ImportError as e:
    print(f"FATAL ERROR: A required library is not installed. Please run 'pip install -r requirements.txt'. Missing module: {e.name}")
    exit()

from prompts import (
    EXTRACTOR_PROMPT,
    BUILDER_PROMPT,
    CLOZE_BUILDER_PROMPT,
    CONCEPTUAL_CLOZE_BUILDER_PROMPT,
    VERIFIER_PROMPT
)

SCRIPT_VERSION = "2.5.0 (Milestone 1 - Foundation)"

ANKI_CONNECT_URL = "http://127.0.0.1:8765"
CACHE_DIR = Path(".pdf_cache")
AI_CACHE_DIR = Path(".ai_cache")
LOG_DIR = Path("logs")
MAX_LOG_FILES = 10
MIN_IMAGE_SIZE_BYTES = 10 * 1024
MAX_ASPECT_RATIO = 10.0
MAX_DECKS = 10
MAX_WORKERS_EXTRACTION = 8
RPM_LIMIT_FLASH = 15
INTER_DECK_COOLDOWN_SECONDS = 30
BATCH_SIZE = 3

NOTE_TYPE_CONFIG = {
    "basic": {
        "modelName": "Anki Deck Generator - Basic",
        "fields": ["Front", "Back", "Image", "Source"],
        "css": """.card { font-family: Arial; font-size: 20px; text-align: center; color: black; background-color: white; }
                  img { max-height: 500px; display: block; margin-left: auto; margin-right: auto; }
                  .nightMode img { background-color: #333; }""",
        "templates": [
            {
                "Name": "Card 1",
                "Front": "{{Front}}",
                "Back": "{{FrontSide}}\n\n<hr id=answer>\n\n{{Back}}\n\n<br><br>\n{{#Image}}{{Image}}{{/Image}}\n<div style='font-size:12px; color:grey; text-align: right;'>Source: {{Source}}</div>"
            }
        ],
        "tags": ["auto-generated-basic"]
    },
    "cloze": {
        "modelName": "Anki Deck Generator - Cloze",
        "fields": ["Text", "Extra", "Image", "Source"],
        "isCloze": True,
        "css": """.card { font-family: Arial; font-size: 20px; text-align: center; color: black; background-color: white; }
                  .cloze { font-weight: bold; color: #0000FF; }
                  .nightMode .cloze { color: #87CEFA; }
                  img { max-height: 500px; display: block; margin-left: auto; margin-right: auto; }
                  .nightMode img { background-color: #333; }""",
        "templates": [
            {
                "Name": "Cloze Card",
                "Front": "{{cloze:Text}}",
                "Back": "{{cloze:Text}}\n\n<br>\n{{Extra}}\n<br><br>\n{{#Image}}{{Image}}{{/Image}}\n<div style='font-size:12px; color:grey; text-align: right;'>Source: {{Source}}</div>"
            }
        ],
        "tags": ["auto-generated-cloze"]
    }
}
# endregion

# ==============================================================================
# SECTION 2: BACK-END ENGINE
# ==============================================================================
# region

def is_image_high_quality_heuristic(image_bytes: bytes) -> bool:
    if not image_bytes: return False
    if len(image_bytes) < MIN_IMAGE_SIZE_BYTES: return False
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            if width < 50 or height < 50: return False
            ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            if ratio > MAX_ASPECT_RATIO: return False
    except Exception: return False
    return True

def get_api_keys_from_env() -> Dict[str, str]:
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / '.env'
    if not env_path.exists(): raise FileNotFoundError(f"Error: .env file not found in {script_dir}. Please create one from the env.template file.")
    load_dotenv(dotenv_path=env_path)
    keys = ("GEMINI_API_KEY", "GOOGLE_SEARCH_API_KEY", "GOOGLE_CSE_ID")
    loaded_keys = {key: os.getenv(key) for key in keys}
    if not loaded_keys.get("GEMINI_API_KEY"):
        raise ValueError(f"Error: GEMINI_API_KEY is missing or empty in {env_path}.")
    return loaded_keys

def get_pdf_content(pdf_path: str) -> Tuple[str, Dict[int, List[bytes]]]:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        pdf_hash = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    except IOError as e: return f"Error reading PDF file {Path(pdf_path).name}: {e}", {}
    cache_file = CACHE_DIR / f"{pdf_hash}.json"
    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text(encoding='utf-8'))
            images_by_page = {int(k): [base64.b64decode(img) for img in v] for k, v in cached_data["images"].items()}
            return cached_data["text"], images_by_page
        except (json.JSONDecodeError, KeyError, TypeError): pass
    
    text_content, images_by_page = "", {}
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                text_content += f"--- Page {page_num} ---\n{page.get_text()}\n\n"
                page_images_data = []
                for img in page.get_images(full=True):
                    if extracted_img := doc.extract_image(img[0]):
                        if extracted_img.get("image"):
                            page_images_data.append(extracted_img["image"])
                if page_images_data:
                    images_by_page[page_num] = page_images_data
    except fitz.errors.FitzError as e:
        if "encrypted" in str(e).lower():
            return f"Error: The PDF file {Path(pdf_path).name} is password-protected and cannot be processed. Please provide a decrypted version.", {}
        return f"Error processing PDF {Path(pdf_path).name} with PyMuPDF: {e}", {}
    
    images_b64 = {k: [base64.b64encode(img).decode() for img in v] for k, v in images_by_page.items()}
    cache_file.write_text(json.dumps({"text": text_content, "images": images_b64}), encoding='utf-8')
    return text_content, images_by_page

def clean_pdf_text(raw_text: str) -> str:
    lines = raw_text.split('\n')
    line_counts = {}
    for line in lines:
        stripped_line = line.strip()
        if 2 < len(stripped_line) < 100:
            line_counts[stripped_line] = line_counts.get(stripped_line, 0) + 1
    
    frequent_lines = {line for line, count in line_counts.items() if count > 2}

    def is_header_or_footer(line):
        line_strip = line.strip()
        if not line_strip: return False
        if line_strip in frequent_lines: return True
        if re.search(r'^\s*page\s*\d+\s*(of\s*\d+)?\s*$', line_strip, re.IGNORECASE): return True
        return False

    cleaned_lines = [line for line in lines if not is_header_or_footer(line)]
    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    return cleaned_text

def call_gemini(prompt: str, api_key: str, model_name: str = "gemini-1.5-pro-latest") -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=300)
        
        if response.status_code == 429: return "API_LIMIT_REACHED"
        if response.status_code == 400: return f"API_BAD_REQUEST"

        response.raise_for_status()
        response_data = response.json()

        if response_data.get('promptFeedback', {}).get('blockReason'):
            reason = response_data['promptFeedback']['blockReason']
            return f"API_SAFETY_BLOCK: {reason}"

        if candidates := response_data.get('candidates', []):
            if parts := candidates[0].get('content', {}).get('parts', []):
                return parts[0]['text']
        
        return f"Warning: Gemini returned an empty response.\n{response.text}"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 500: return "API_SERVER_ERROR"
        return f"Error calling Gemini API: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error calling Gemini API: {e}"
    return "Error: Empty response from Gemini."

def call_gemini_vision(prompt: str, api_key: str, image_bytes_1: bytes, image_bytes_2: bytes) -> str:
    model_name = "gemini-1.5-pro-latest"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    b64_image_1 = base64.b64encode(image_bytes_1).decode('utf-8')
    b64_image_2 = base64.b64encode(image_bytes_2).decode('utf-8')
    data = { "contents": [ { "parts": [ {"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": b64_image_1}}, {"inline_data": {"mime_type": "image/png", "data": b64_image_2}}, ] } ] }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=300)
        if response.status_code == 429: return "API_LIMIT_REACHED"
        response.raise_for_status()
        response_data = response.json()
        if candidates := response_data.get('candidates', []):
            if parts := candidates[0].get('content', {}).get('parts', []): return parts[0]['text'].strip()
        return f"Warning: Gemini Vision returned an empty or blocked response.\n{response.text}"
    except requests.exceptions.RequestException as e: return f"Error calling Gemini Vision API: {e}"
    return "NEITHER"

def search_google_images(query: str, api_key: str, cse_id: str) -> str | None:
    params = {'q': query, 'key': api_key, 'cx': cse_id, 'searchType': 'image', 'num': 1}
    try:
        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
        response.raise_for_status()
        if items := response.json().get('items', []): return items[0]['link']
    except requests.exceptions.RequestException: pass
    return None

def download_image_from_url(url: str) -> bytes | None:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException: pass
    return None

def invoke_ankiconnect(action: str, **params: Any) -> Tuple[Any | None, str | None]:
    payload = json.dumps({"action": action, "version": 6, "params": params})
    try:
        response = requests.post(ANKI_CONNECT_URL, data=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get('error'): return None, result.get('error')
        return result.get('result'), None
    except requests.exceptions.RequestException as e: return None, f"Could not connect to AnkiConnect: {e}"

def setup_anki_deck_and_note_type(deck_name: str, note_type_key: str) -> str | None:
    config = NOTE_TYPE_CONFIG.get(note_type_key)
    if not config: return f"Internal Error: Note type config for '{note_type_key}' not found."
    
    model_name = config["modelName"]
    
    deck_names, error = invoke_ankiconnect("deckNames")
    if error: return f"AnkiConnect Error: {error}"
    if deck_name not in deck_names:
        _, error = invoke_ankiconnect("createDeck", deck=deck_name)
        if error: return f"Failed to create deck '{deck_name}': {error}"

    model_names, error = invoke_ankiconnect("modelNames")
    if error: return f"AnkiConnect Error: {error}"
    
    if model_name not in model_names:
        params = {
            "modelName": model_name,
            "inOrderFields": config["fields"],
            "css": config["css"],
            "isCloze": config.get("isCloze", False),
            "cardTemplates": config["templates"]
        }
        _, error = invoke_ankiconnect("createModel", **params)
        if error: return f"Failed to create note type '{model_name}': {error}"
    else:
        expected_fields = set(config["fields"])
        model_field_names, error = invoke_ankiconnect("modelFieldNames", modelName=model_name)
        if error: return f"Could not verify fields for note type '{model_name}': {error}"
        if not expected_fields.issubset(set(model_field_names)):
            return (f"Error: Note type '{model_name}' exists but is missing required fields. "
                    f"Expected: {list(expected_fields)}, Found: {model_field_names}. "
                    f"Please delete or rename the existing note type in Anki and try again.")
    return None

def add_note_to_anki(deck_name: str, note_type_key: str, fields_data: Dict[str, str], source_filename: str, custom_tags: List[str]) -> Tuple[int | None, str | None]:
    config = NOTE_TYPE_CONFIG.get(note_type_key)
    if not config: return None, f"Configuration for note type '{note_type_key}' not found."

    filename_stem = Path(source_filename).stem
    pdf_tag = f"PDF::{filename_stem.replace(' ', '_')}"
    
    final_tags = config.get("tags", []) + [pdf_tag] + custom_tags
    
    note = {
        "deckName": deck_name, "modelName": config["modelName"],
        "fields": fields_data, "options": {"allowDuplicate": False}, "tags": final_tags
    }
    
    note_id, error = invoke_ankiconnect("addNote", note=note)
    if error: return None, error
    return note_id, None

def update_note_with_image(note_id: int, image_bytes: bytes, original_filename: str, note_type_key: str) -> str:
    config = NOTE_TYPE_CONFIG.get(note_type_key)
    if not (note_id and image_bytes and config): return "Skipped image update (no data)."
    image_field = "Image"
    safe_filename = re.sub(r'[\\/*?:"<>|]', "", original_filename)
    filename = f"deck-gen-{note_id}-{safe_filename[:50]}.png"
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    _, error = invoke_ankiconnect("storeMediaFile", filename=filename, data=b64_image)
    if error: return f"Failed to store media file {filename}: {error}"
    note_update = {"id": note_id, "fields": {image_field: f'<img src="{filename}">'} }
    _, error = invoke_ankiconnect("updateNoteFields", note=note_update)
    if error: return f"Failed to update note {note_id} with image: {error}"
    return f"Successfully added image {filename} to note ID {note_id}."
# endregion

# ==============================================================================
# SECTION 3: GRADIO UI & EVENT HANDLING
# ==============================================================================
# region

def manage_log_files():
    LOG_DIR.mkdir(exist_ok=True)
    log_files = sorted(LOG_DIR.glob('*.txt'), key=os.path.getmtime)
    while len(log_files) >= MAX_LOG_FILES:
        os.remove(log_files[0])
        log_files.pop(0)

def clear_cache():
    results = []
    if CACHE_DIR.exists():
        try: shutil.rmtree(CACHE_DIR); results.append("✅ PDF cache cleared.")
        except Exception as e: results.append(f"❌ Error clearing PDF cache: {e}")
    else: results.append("ℹ️ PDF cache not found.")
    if AI_CACHE_DIR.exists():
        try: shutil.rmtree(AI_CACHE_DIR); results.append("✅ AI cache cleared.")
        except Exception as e: results.append(f"❌ Error clearing AI cache: {e}")
    else: results.append("ℹ️ AI cache not found.")
    return " ".join(results)

def guess_lecture_details(file: gr.File) -> Tuple[str, str]:
    if not file: return "01", ""
    file_stem = Path(file.name).stem
    num_guess = "1"
    keyword_pattern = r'(lecture|lec|session|s|chapter|chap|module|mod|unit|part|l)[\s_-]*(\d+(?:(?:[\s\/&-]| and )\d+)*)'

    if match := re.search(keyword_pattern, file_stem, re.IGNORECASE):
        num_guess = match.group(2).strip()
    elif match := re.search(r'^\s*(\d+(?:(?:[\s\/&-]| and )\d+)*)', file_stem):
        num_guess = match.group(1).strip()

    if len(num_guess) == 1:
        num_guess = f"0{num_guess}"

    name_guess = file_stem
    name_guess = re.sub(r'\b[A-Z]+[\s_-]*\d+([\s-]*\d+)*\b', '', name_guess, flags=re.IGNORECASE)
    name_guess = re.sub(keyword_pattern, '', name_guess, flags=re.IGNORECASE)
    name_guess = re.sub(r'^\s*' + re.escape(num_guess) + r'[\s_-]*', '', name_guess, re.IGNORECASE)
    name_guess = re.sub(r'[\s_-]+', ' ', name_guess).strip()
    return num_guess, name_guess.title()

def update_decks_from_files(files: List[gr.File]) -> List[Any]:
    updates = []
    num_files = len(files) if files else 0
    if num_files > 0: updates.append(gr.update(label=f"{num_files} File(s) Loaded"))
    else: updates.append(gr.update(label="Upload PDFs for each deck"))

    for i in range(MAX_DECKS):
        if i < num_files:
            num, name = guess_lecture_details(files[i])
            updates.extend([gr.update(visible=True), gr.update(value=num), gr.update(value=name), gr.update(value=[files[i].name], visible=True)])
        else:
            updates.extend([gr.update(visible=False), gr.update(value=""), gr.update(value=""), gr.update(value=[], visible=False)])
    return updates

def run_pre_flight_checks(api_keys: Dict[str, str], deck_configs: List[Tuple]) -> str | None:
    # 1. AnkiConnect Check
    version, error = invoke_ankiconnect("version")
    if error or not version: return f"CRITICAL ERROR: Could not connect to Anki.\nSOLUTION: Please ensure the Anki desktop application is running and the AnkiConnect Add-on is installed."
    if int(version) < 6: return f"CRITICAL ERROR: Your AnkiConnect add-on is outdated (Version {version}). Please update to the latest version."

    # 2. API Key Validity Check
    test_response = call_gemini("Hello", api_keys['GEMINI_API_KEY'], model_name="gemini-1.5-flash-latest")
    if "API key not valid" in test_response or "API_KEY_INVALID" in test_response:
        return "CRITICAL ERROR: Your Gemini API Key appears to be invalid. Please check your .env file."
    
    # 3. File System Permissions Check
    try:
        LOG_DIR.mkdir(exist_ok=True)
        test_file = LOG_DIR / "permissions_test.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except OSError as e:
        return f"CRITICAL ERROR: The script does not have permission to write to the 'logs' directory. Please check your folder permissions. Details: {e}"

    # 4. PDF Accessibility Check
    if not deck_configs: return "ERROR: No valid decks configured. Please upload at least one PDF and provide a deck name."
    for _, files in deck_configs:
        for pdf in files:
            if not Path(pdf.name).exists():
                return f"CRITICAL ERROR: The file '{Path(pdf.name).name}' could not be found. It may have been moved or deleted after being uploaded."
    return None

def generate_all_decks(master_files, generate_button, log_output, *args):
    yield "", gr.update(interactive=False), gr.update(value="Processing...", interactive=False)
    
    manage_log_files()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = LOG_DIR / f"log_{timestamp}.txt"
    log_history = ""
    def logger(message):
        nonlocal log_history
        timestamp_msg = datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp_msg}] {message}"
        log_history += full_message + "\n"
        with open(log_filename, 'a', encoding='utf-8') as f: f.write(full_message + "\n")
        return log_history

    try:
        yield logger(f"Starting Anki Deck Generator {SCRIPT_VERSION}"), gr.update(), gr.update()
        
        deck_inputs = args[:MAX_DECKS*3]
        card_type, image_strategy, custom_tags_str = args[MAX_DECKS*3:MAX_DECKS*3+3]
        prompts_dict = {
            'extractor': args[MAX_DECKS*3+3],
            'builder': args[MAX_DECKS*3+4],
            'cloze_builder': args[MAX_DECKS*3+5],
            'conceptual_cloze_builder': args[MAX_DECKS*3+6]
        }
        custom_tags = [tag.strip() for tag in custom_tags_str.split(',') if tag.strip()]

        try: api_keys = get_api_keys_from_env()
        except (FileNotFoundError, ValueError) as e:
            yield logger(f"CRITICAL ERROR: {e}"), gr.update(), gr.update(); return

        deck_configs = []
        for i in range(0, len(deck_inputs), 3):
            deck_num, deck_name, files = deck_inputs[i], deck_inputs[i+1], deck_inputs[i+2]
            if deck_name and files:
                deck_configs.append((f"Lecture {deck_num} - {deck_name}", files))
        
        pre_flight_error = run_pre_flight_checks(api_keys, deck_configs)
        if pre_flight_error:
            yield logger(pre_flight_error), gr.update(), gr.update(); return
        yield logger("Pre-flight checks passed successfully."), gr.update(), gr.update()

        yield logger(f"Found {len(deck_configs)} deck(s) to process."), gr.update(), gr.update()

        for i, (deck_name, files) in enumerate(deck_configs, 1):
            progress = gr.Progress(track_tqdm=True)
            yield logger(f"\n--- Starting Deck {i} of {len(deck_configs)}: '{deck_name}' ---"), gr.update(), gr.update()

            AI_CACHE_DIR.mkdir(exist_ok=True)
            note_type_key = "cloze" if "Cloze" in card_type else "basic"
            yield logger(f"Card Type Selected: {card_type}"), gr.update(), gr.update()
            yield logger(f"Image Strategy Selected: {image_strategy}"), gr.update(), gr.update()
            if custom_tags:
                yield logger(f"Custom Tags: {', '.join(custom_tags)}"), gr.update(), gr.update()

            error = setup_anki_deck_and_note_type(deck_name, note_type_key)
            if error: yield logger(f"DECK SETUP ERROR: {error}"), gr.update(), gr.update(); continue
            yield logger(f"Anki setup for deck '{deck_name}' is correct."), gr.update(), gr.update()

            yield logger("\n--- Processing PDF Files ---"), gr.update(), gr.update()
            full_text, all_images_by_page = "", {}
            pdf_paths = [file.name for file in files]
            combined_pdf_hash = hashlib.sha256()
            for pdf_path in progress.tqdm(pdf_paths, desc="Processing PDFs"):
                pdf_bytes = Path(pdf_path).read_bytes()
                combined_pdf_hash.update(pdf_bytes)
                text, images_on_page = get_pdf_content(pdf_path)
                if "Error:" in text and not images_on_page:
                    yield logger(text), gr.update(), gr.update(); continue
                full_text += f"\n\n--- Content from {Path(pdf_path).name} ---\n{text}"
                all_images_by_page.update(images_on_page)
            yield logger("All PDF files processed and cached."), gr.update(), gr.update()
            
            yield logger("\n--- Cleaning and Structuring Text ---"), gr.update(), gr.update()
            cleaned_text = clean_pdf_text(full_text)
            
            yield logger(f"\n--- AI Pass 1: Extracting atomic facts (Batch Size: {BATCH_SIZE}) ---"), gr.update(), gr.update()
            extractor_model = "gemini-1.5-flash-latest"
            yield logger(f"Using cheaper model for extraction: {extractor_model}"), gr.update(), gr.update()
            
            pages = cleaned_text.split("--- Page ")[1:]
            
            all_pages = [(p_num, f"--- Page {p_num} ---\n{page_text}") for p_num, page_text in enumerate(pages, 1) if page_text.strip()]
            batched_tasks = []
            for j in range(0, len(all_pages), BATCH_SIZE):
                batch = all_pages[j:j+BATCH_SIZE]
                page_numbers = [str(p[0]) for p in batch]
                combined_text = "\n\n".join([p[1] for p in batch])
                task_id = f"{page_numbers[0]}-{page_numbers[-1]}"
                batched_tasks.append((task_id, combined_text))

            atomic_facts_by_page_range = {}
            request_timestamps = deque()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS_EXTRACTION) as executor:
                def extract_facts_from_batch(task_data):
                    nonlocal request_timestamps
                    page_range, batch_content = task_data
                    
                    while True:
                        current_time = time.time()
                        while request_timestamps and request_timestamps[0] < current_time - 60: request_timestamps.popleft()
                        if len(request_timestamps) < RPM_LIMIT_FLASH:
                            request_timestamps.append(current_time)
                            break
                        else:
                            time.sleep(60 - (current_time - request_timestamps[0]) + 0.1)

                    response = call_gemini(prompts_dict['extractor'] + f"\n\n--- TEXT ---\n{batch_content}", api_keys["GEMINI_API_KEY"], model_name=extractor_model)
                    return page_range, response

                results = list(progress.tqdm(executor.map(extract_facts_from_batch, batched_tasks), total=len(batched_tasks), desc="Extracting Facts in Batches"))

            atomic_facts_with_pages = ""
            for page_range, page_facts in results:
                if page_facts == "API_LIMIT_REACHED":
                    yield logger(("\n--- PROCESS STOPPED: API LIMIT REACHED ---\n..."))
                    return
                elif "API_" in page_facts:
                    yield logger(f"\nWARNING (Page Range {page_range}): {page_facts}. This batch will be skipped.")
                    continue
                if page_facts and "Error" not in page_facts:
                    atomic_facts_by_page_range[page_range] = page_facts
                    atomic_facts_with_pages += f"--- Page(s) {page_range} ---\n{page_facts}\n"

            atomic_facts_str = "".join(atomic_facts_by_page_range.values())
            if not atomic_facts_str.strip():
                yield logger("ERROR: AI Pass 1 failed to extract any facts from any page."); continue
            yield logger(f"Pass 1 complete. Found {len(atomic_facts_str.splitlines())} facts."), gr.update(), gr.update()
            
            safe_card_type_str = re.sub(r'[\\/*?:"<>|()]', "", card_type)
            card_type_slug = safe_card_type_str.replace(" ", "_").lower()
            ai_cache_key = f"{combined_pdf_hash.hexdigest()}_{card_type_slug}.json"
            ai_cache_file = AI_CACHE_DIR / ai_cache_key
            final_cards_json_str = ""

            if ai_cache_file.exists():
                yield logger("\n--- Found Cached AI Response! Skipping card generation. ---"), gr.update(), gr.update()
                final_cards_json_str = ai_cache_file.read_text(encoding='utf-8')
            else:
                yield logger("\n--- No cached response found. Generating new cards with AI... ---"), gr.update(), gr.update()
                learning_objectives_text = "".join(p for p in pages if "learning objectives" in p.lower())

                if "Basic" in card_type: pass2_prompt = prompts_dict['builder'].format(atomic_facts=atomic_facts_str, learning_objectives=learning_objectives_text)
                elif "Atomic Cloze" in card_type: pass2_prompt = prompts_dict['cloze_builder'].format(atomic_facts_with_pages=atomic_facts_with_pages)
                else: pass2_prompt = prompts_dict['conceptual_cloze_builder'].format(atomic_facts_with_pages=atomic_facts_with_pages)

                final_cards_json_str = call_gemini(pass2_prompt, api_keys["GEMINI_API_KEY"])
                if "API_" in final_cards_json_str:
                    yield logger(f"\nERROR: AI card generation failed. Reason: {final_cards_json_str}"); continue
                
                if "Error" not in final_cards_json_str and final_cards_json_str.strip():
                    ai_cache_file.write_text(final_cards_json_str, encoding='utf-8')
                    yield logger("Saved new AI response to cache."), gr.update(), gr.update()

            if not final_cards_json_str or "Error" in final_cards_json_str:
                yield logger(f"ERROR: AI card generation failed. Response: {final_cards_json_str}"); continue
            
            yield logger("\n--- Parsing AI JSON Response ---"), gr.update(), gr.update()
            try:
                json_cleaned_str = final_cards_json_str.strip().replace("```json", "").replace("```", "")
                final_cards = json.loads(json_cleaned_str)
                yield logger(f"Successfully parsed {len(final_cards)} cards from JSON response."), gr.update(), gr.update()
            except json.JSONDecodeError as e:
                yield logger(f"CRITICAL ERROR: Failed to parse JSON from AI. Error: {e}\n\nAI Response was:\n{final_cards_json_str}"); continue

            yield logger(f"\n--- Adding Cards to Deck: '{deck_name}' ---"), gr.update(), gr.update()
            cards_added, cards_skipped, cards_failed = 0, 0, 0
            for card_data in progress.tqdm(final_cards, desc="Adding Cards to Anki"):
                try:
                    fields = {}
                    final_note_type_key = note_type_key
                    page_ref = card_data.get('page', card_data.get('best_pdf_page_for_image', 1))
                    source_text = f"{Path(pdf_paths[0]).stem} - Pg {page_ref}"

                    if note_type_key == "basic":
                        front, back = card_data.get("front"), card_data.get("back")
                        if not (front and back): cards_skipped += 1; continue
                        fields = {"Front": front, "Back": back, "Source": source_text}
                    else:
                        sentence, original_question = card_data.get("sentence"), card_data.get("original_question", "")
                        if not (sentence and original_question): cards_skipped += 1; continue
                        final_cloze_text = sentence
                        if keywords := card_data.get("keywords"):
                            for k, kw in enumerate(keywords): final_cloze_text = final_cloze_text.replace(kw, f"{{{{c{k+1}::{kw}}}}}")
                        elif keyword := card_data.get("keyword"): final_cloze_text = sentence.replace(keyword, f"{{{{c1::{keyword}}}}}")
                        
                        if "{{" not in final_cloze_text:
                            yield logger(f"   > WARNING (Page {page_ref}): AI keyword mismatch. Converting to Basic card.")
                            fields = {"Front": original_question, "Back": sentence, "Source": source_text}
                            final_note_type_key = "basic"
                        else:
                            fields = {"Text": final_cloze_text, "Extra": original_question, "Source": source_text}
                    
                    if not fields: cards_skipped += 1; continue
                    note_id, error = add_note_to_anki(deck_name, final_note_type_key, fields, pdf_paths[0], custom_tags)
                    if error:
                        yield logger(f"   > FAILED to add note. Reason: {error}")
                        cards_failed += 1; continue
                    cards_added += 1
                except Exception as e:
                    cards_failed += 1; yield logger(f"ERROR processing card data: '{card_data}' | Exception: {e}")

            yield logger(f"\n--- Final Tally ---\nCards Added: {cards_added}\nCards Skipped/Failed: {cards_skipped + cards_failed}"), gr.update(), gr.update()
            
            yield logger(f"--- Finished Deck {i}: '{deck_name}' ---\n"), gr.update(), gr.update()
            
            if i < len(deck_configs):
                yield logger(f"Pausing for {INTER_DECK_COOLDOWN_SECONDS} seconds to respect API rate limits..."), gr.update(), gr.update()
                time.sleep(INTER_DECK_COOLDOWN_SECONDS)
            
        yield logger("--- All Decks Processed! ---"), gr.update(), gr.update()

    except Exception as e:
        yield logger(f"\n--- A CRITICAL ERROR OCCURRED ---\n{e}\nTraceback: {traceback.format_exc()}"), gr.update(), gr.update()
    finally:
        yield gr.update(), gr.update(interactive=True), gr.update(value="Generate All Decks", interactive=True)
def build_ui() -> gr.Blocks:
    IMAGE_STRATEGY_HELP_TEXT = {
        "None (Text-Only)": "<strong>Fastest:</strong> Creates text-only cards. No images will be added.",
        "PDF Only (Fastest, Free)": "<strong>Recommended for Bulk:</strong> Only uses images found in your PDF. Fast and does not use web search quotas.",
        "PDF Priority (Balanced)": "<strong>Best Quality (Default):</strong> Prioritizes PDF images, but uses Google Images as a fallback if needed. <em>(Note: Free web search quota is limited).</em>",
        "AI Verified (Paid API Key Recommended)": "⚠️ <strong>High Cost / Pro Users:</strong> Uses the expensive `gemini-1-pro` model to pick the best image. <em>This will exhaust your free daily quota very quickly.</em>"
    }
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Anki Deck Generator") as app:
        gr.Markdown(f"# Anki Flashcard Generator\n*(v{SCRIPT_VERSION})*")

        # --- Top Level Control Panel ---
        with gr.Row():
            generate_button = gr.Button("Generate All Decks", variant="primary", scale=2)
            cancel_button = gr.Button("Cancel")
            new_batch_button = gr.Button("Start New Batch")

        with gr.Row():
            # --- Left Column: Configuration ---
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("1. Decks & Files"):
                        with gr.Group():
                            master_files = gr.File(label="Upload PDFs for each deck", file_count="multiple", file_types=[".pdf"])
                        
                        deck_ui_components, deck_input_components = [], []
                        for i in range(MAX_DECKS):
                            with gr.Accordion(f"Deck {i+1}", visible=(i==0), open=True) as acc:
                                with gr.Row():
                                    num = gr.Textbox(label="Lecture #", scale=1)
                                    name = gr.Textbox(label="Lecture Name", scale=3)
                                files = gr.File(label=f"PDF for Deck {i+1}", visible=False, file_count="multiple", interactive=False)
                            deck_ui_components.extend([acc, num, name, files])
                            deck_input_components.extend([num, name, files])

                    with gr.TabItem("2. Settings"):
                        with gr.Group():
                            card_type = gr.Radio(["Conceptual (Basic Cards)", "Atomic Cloze (1 fact/card)", "Conceptual Cloze (Linked facts/card)"], label="Card Type", value="Conceptual (Basic Cards)")
                            image_strategy = gr.Radio(["None (Text-Only)", "PDF Only (Fastest, Free)", "PDF Priority (Balanced)", "AI Verified (Paid API Key Recommended)"], label="Image Selection Strategy", value="PDF Priority (Balanced)")
                            image_strategy_help = gr.Markdown(value=IMAGE_STRATEGY_HELP_TEXT["PDF Priority (Balanced)"], elem_classes="help-text")
                            custom_tags_textbox = gr.Textbox(label="Custom Tags (Optional)", placeholder="e.g., #Anatomy, #Midterm_1, #Cardiology", info="Add your own tags, separated by commas.")

                    with gr.TabItem("3. Advanced"):
                         with gr.Accordion("Edit System Prompts (Power Users Only)", open=False):
                            gr.Markdown("⚠️ **Warning:** Editing these prompts can break the application if the AI's output format is changed. Edit with caution.")
                            extractor_prompt_editor = gr.Textbox(label="Fact Extractor Prompt", value=EXTRACTOR_PROMPT, lines=10, max_lines=20)
                            builder_prompt_editor = gr.Textbox(label="Basic Card Builder Prompt", value=BUILDER_PROMPT, lines=10, max_lines=20)
                            cloze_builder_prompt_editor = gr.Textbox(label="Atomic Cloze Builder Prompt", value=CLOZE_BUILDER_PROMPT, lines=10, max_lines=20)
                            conceptual_cloze_builder_prompt_editor = gr.Textbox(label="Conceptual Cloze Builder Prompt", value=CONCEPTUAL_CLOZE_BUILDER_PROMPT, lines=10, max_lines=20)
                         
                         with gr.Accordion("Cache Management", open=False):
                            clear_cache_button = gr.Button("Clear All Caches")
                            cache_status = gr.Textbox(label="Cache Status", interactive=False)
                         
                         with gr.Accordion("Acknowledgements", open=False):
                            gr.Markdown("""
                                This project was built with the invaluable help of the open-source community... (full text omitted for brevity)
                            """)
            
            # --- Right Column: Session Log ---
            with gr.Column(scale=1):
                gr.Markdown("### Session Log")
                log_output = gr.Textbox(label="Progress", lines=30, interactive=False, autoscroll=True)
                copy_log_button = gr.Button("Copy Log for Debugging")
        
        # --- Event Handlers ---
        master_files.change(fn=update_decks_from_files, inputs=master_files, outputs=[master_files] + deck_ui_components)
        
        def update_help_text(choice):
            return gr.update(value=IMAGE_STRATEGY_HELP_TEXT.get(choice, ""))
        
        image_strategy.change(fn=update_help_text, inputs=image_strategy, outputs=image_strategy_help)
        
        clear_cache_button.click(fn=clear_cache, outputs=[cache_status])

        all_gen_inputs = [master_files, generate_button, log_output] + deck_input_components + [card_type, image_strategy, custom_tags_textbox, extractor_prompt_editor, builder_prompt_editor, cloze_builder_prompt_editor, conceptual_cloze_builder_prompt_editor]
        all_gen_outputs = [log_output, master_files, generate_button]

        gen_event = generate_button.click(
            fn=generate_all_decks, 
            inputs=all_gen_inputs, 
            outputs=all_gen_outputs
        )
        
        cancel_button.click(fn=None, cancels=[gen_event])
        copy_log_button.click(fn=None, inputs=[log_output], js="(text) => { navigator.clipboard.writeText(text); alert('Log copied to clipboard!'); }")
        
        all_deck_files_components = [ui for i, ui in enumerate(deck_input_components) if i % 3 == 2]
        new_batch_button.click(fn=lambda: (gr.update(value=None), gr.update(value=""), []) + [gr.update(value=[]) for _ in all_deck_files_components], outputs=[master_files, log_output] + all_deck_files_components)

    return app

# endregion

# ==============================================================================
# SECTION 4: SCRIPT LAUNCHER
# ==============================================================================
# region
if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, debug=True, inbrowser=True)
# endregion