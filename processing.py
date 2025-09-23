# processing.py (V5.1 - Final Corrected Version)

import json, base64, hashlib, io, re, time, traceback
import regex
import markdown2
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import fitz, requests, gradio as gr
from PIL import Image

from prompts import EXTRACTOR_PROMPT, BUILDER_PROMPT, CLOZE_BUILDER_PROMPT, CONCEPTUAL_CLOZE_BUILDER_PROMPT
from utils import get_api_keys_from_env, manage_log_files, search_wikimedia_for_image, is_image_high_quality_heuristic, perform_ocr_on_image

# --- Configuration Constants ---
ANKI_CONNECT_URL = "http://127.0.0.1:8765"
MAX_WORKERS_EXTRACTION = 8
RPM_LIMIT_FLASH = 15
BATCH_SIZE = 3
INTER_DECK_COOLDOWN_SECONDS = 30
HTML_COLOR_MAP = {
    "positive_key_term": "#87CEFA", "negative_key_term": "#FF6347",
    "example": "#90EE90", "mnemonic_tip": "#FFD700"
}

NOTE_TYPE_CONFIG = {
    "basic": {
        "modelName": "Anki Deck Generator - Basic",
        "fields": ["Front", "Back", "Image", "Source"],
        "css": """.card { font-family: Arial; font-size: 20px; text-align: center; } img { max-height: 500px; } ul { display: inline-block; text-align: left; }""",
        "templates": [{"Name": "Card 1", "Front": "{{Front}}", "Back": "{{FrontSide}}\n\n<hr id=answer>\n\n{{Back}}\n\n<br><br>{{#Image}}{{Image}}{{/Image}}\n<div style='font-size:12px; color:grey;'>{{Source}}</div>"}]
    },
    "cloze": {
        "modelName": "Anki Deck Generator - Cloze",
        "fields": ["Text", "Extra", "Image", "Source"],
        "isCloze": True,
        "css": """.card { font-family: Arial; font-size: 20px; text-align: center; } .cloze { font-weight: bold; color: blue; } img { max-height: 500px; }""",
        "templates": [{"Name": "Cloze Card", "Front": "{{cloze:Text}}", "Back": "{{cloze:Text}}\n\n<br>{{Extra}}\n<br><br>{{#Image}}{{Image}}{{/Image}}\n<div style='font-size:12px; color:grey;'>{{Source}}</div>"}]
    }
}

# --- HTML Engine ---
def build_html_from_tags(text: str, enabled_colors: List[str]) -> str:
    """Converts custom tags and hyphens into final HTML."""
    tag_map = {
        "<pos>": ("positive_key_term", f"<font color='{HTML_COLOR_MAP['positive_key_term']}'><b>"), "</pos>": ("positive_key_term", "</b></font>"),
        "<neg>": ("negative_key_term", f"<font color='{HTML_COLOR_MAP['negative_key_term']}'><b>"), "</neg>": ("negative_key_term", "</b></font>"),
        "<ex>": ("example", f"<font color='{HTML_COLOR_MAP['example']}'>"), "</ex>": ("example", "</font>"),
        "<tip>": ("mnemonic_tip", f"<font color='{HTML_COLOR_MAP['mnemonic_tip']}'>"), "</tip>": ("mnemonic_tip", "</font>"),
    }
    for tag, (key, replacement) in tag_map.items():
        text = text.replace(tag, replacement if key in enabled_colors else "")
    
    # Convert Markdown lists to HTML line breaks for centered formatting
    html = markdown2.markdown(text, extras=["cuddled-lists"])
    # Remove list tags and convert list items to hyphenated lines
    html = html.replace("<ul>\n<li>", "- ").replace("</li>\n<li>", "<br>- ").replace("</li>\n</ul>", "")
    # Ensure any remaining newlines become line breaks
    return html.replace('\n', '<br>')

def get_pdf_content(pdf_path: str, pdf_cache_dir: Path) -> Tuple[str, Dict[int, List[bytes]], List[str]]:
    pdf_cache_dir.mkdir(exist_ok=True)
    ocr_log = []
    def is_text_meaningful(text: str, min_chars: int = 20) -> bool:
        alphanumeric_chars = re.sub(r'[^a-zA-Z0-9]', '', text)
        return len(alphanumeric_chars) >= min_chars
    try:
        pdf_hash = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    except IOError as e: 
        return f"Error reading PDF file {Path(pdf_path).name}: {e}", {}, []
    cache_file = pdf_cache_dir / f"{pdf_hash}.json"
    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text(encoding='utf-8'))
            images_by_page = {int(k): [base64.b64decode(img) for img in v] for k, v in cached_data["images"].items()}
            return cached_data["text"], images_by_page, []
        except (json.JSONDecodeError, KeyError, TypeError): pass
    text_content, images_by_page = "", {}
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                if not is_text_meaningful(page_text):
                    ocr_log.append(f"Page {page_num}: No selectable text found, falling back to OCR.")
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    page_text = perform_ocr_on_image(img_bytes)
                text_content += f"--- Page {page_num} ---\n{page_text}\n\n"
                page_images_data = []
                for img in page.get_images(full=True):
                    if extracted_img := doc.extract_image(img[0]):
                        if extracted_img.get("image"): page_images_data.append(extracted_img["image"])
                if page_images_data: images_by_page[page_num] = page_images_data
    except fitz.errors.FitzError as e:
        if "encrypted" in str(e).lower():
            return f"Error: The PDF file {Path(pdf_path).name} is password-protected and cannot be processed.", {}, []
        return f"Error processing PDF {Path(pdf_path).name}: {e}", {}, []
    images_b64 = {k: [base64.b64encode(img).decode() for img in v] for k, v in images_by_page.items()}
    cache_file.write_text(json.dumps({"text": text_content, "images": images_b64}), encoding='utf-8')
    return text_content, images_by_page, ocr_log

def clean_pdf_text(raw_text: str) -> str:
    lines = raw_text.split('\n')
    line_counts = {line.strip(): lines.count(line) for line in set(lines) if 2 < len(line.strip()) < 100}
    frequent_lines = {line for line, count in line_counts.items() if count > 2}
    def is_header_or_footer(line):
        line_strip = line.strip()
        if not line_strip: return False
        if line_strip in frequent_lines: return True
        if re.search(r'^\s*page\s*\d+\s*(of\s*\d+)?\s*$', line_strip, re.IGNORECASE): return True
        return False
    cleaned_lines = [line for line in lines if not is_header_or_footer(line)]
    cleaned_text = '\n'.join(cleaned_lines)
    return re.sub(r'\n{3,}', '\n\n', cleaned_text)

def call_gemini(prompt: str, api_key: str, model_name: str = "gemini-1.5-pro-latest") -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=300)
        if response.status_code == 429: return "API_LIMIT_REACHED"
        if response.status_code == 400: return "API_BAD_REQUEST"
        response.raise_for_status()
        response_data = response.json()
        if response_data.get('promptFeedback', {}).get('blockReason'):
            return f"API_SAFETY_BLOCK: {response_data['promptFeedback']['blockReason']}"
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

def invoke_ankiconnect(action: str, **params: Any) -> Tuple[Any | None, str | None]:
    payload = json.dumps({"action": action, "version": 6, "params": params})
    try:
        response = requests.post(ANKI_CONNECT_URL, data=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        if result.get('error'): return None, result.get('error')
        return result.get('result'), None
    except requests.exceptions.RequestException as e: return None, f"Could not connect to AnkiConnect: {e}"

def run_pre_flight_checks(api_keys: Dict[str, str], deck_configs: List[Tuple]) -> str | None:
    version, error = invoke_ankiconnect("version")
    if error or not version: return "CRITICAL ERROR: Could not connect to Anki.\nSOLUTION: Please ensure Anki is running and AnkiConnect is installed."
    if int(version) < 6: return f"CRITICAL ERROR: Your AnkiConnect add-on is outdated (Version {version}). Please update."
    test_response = call_gemini("Hello", api_keys['GEMINI_API_KEY'], model_name="gemini-1.5-flash-latest")
    if "API key not valid" in test_response or "API_KEY_INVALID" in test_response:
        return "CRITICAL ERROR: Your Gemini API Key appears to be invalid. Please check your .env file."
    if not deck_configs: return "ERROR: No valid decks configured. Please upload at least one PDF and provide a deck name."
    for _, files in deck_configs:
        for pdf in files:
            if not Path(pdf.name).exists():
                return f"CRITICAL ERROR: The file '{Path(pdf.name).name}' could not be found."
    return None

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
        params = {"modelName": model_name, "inOrderFields": config["fields"], "css": config["css"], "isCloze": config.get("isCloze", False), "cardTemplates": config["templates"]}
        _, error = invoke_ankiconnect("createModel", **params)
        if error: return f"Failed to create note type '{model_name}': {error}"
    else:
        expected_fields = set(config["fields"])
        model_field_names, error = invoke_ankiconnect("modelFieldNames", modelName=model_name)
        if error: return f"Could not verify fields for note type '{model_name}': {error}"
        if not expected_fields.issubset(set(model_field_names)):
            return f"Error: Note type '{model_name}' exists but is missing required fields. Expected: {list(expected_fields)}, Found: {model_field_names}."
    return None

def add_note_to_anki(deck_name: str, note_type_key: str, fields_data: Dict[str, str], source_filename: str, custom_tags: List[str]) -> Tuple[int | None, str | None]:
    config = NOTE_TYPE_CONFIG.get(note_type_key)
    if not config: return None, f"Configuration for note type '{note_type_key}' not found."
    filename_stem = Path(source_filename).stem
    pdf_tag = f"PDF::{filename_stem.replace(' ', '_')}"
    final_tags = config.get("tags", []) + [pdf_tag] + custom_tags
    note = {"deckName": deck_name, "modelName": config["modelName"], "fields": fields_data, "options": {"allowDuplicate": False}, "tags": final_tags}
    return invoke_ankiconnect("addNote", note=note)

class DeckProcessor:
    def __init__(self, deck_name, files, api_keys, logger, progress, card_type, image_strategy, enabled_colors, custom_tags, prompts_dict, cache_dirs, batch_size, max_workers, rpm_limit):
        self.deck_name = deck_name
        self.files = files
        self.api_keys = api_keys
        self.logger_func = logger
        self.progress = progress
        self.card_type = card_type
        self.image_strategy = image_strategy
        self.enabled_colors = enabled_colors
        self.custom_tags = custom_tags
        self.prompts_dict = prompts_dict
        self.pdf_cache_dir, self.ai_cache_dir = cache_dirs
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.rpm_limit = rpm_limit
        self.note_type_key = "cloze" if "Cloze" in self.card_type else "basic"
        self.full_text = ""
        self.all_images_by_page = {}
        self.pdf_paths = [file.name for file in self.files]
        self.combined_pdf_hash = hashlib.sha256()

    def log(self, message):
        self.logger_func(message) 

    def run(self):
        try:
            if not self._setup_anki(): return
            if not self._process_pdfs(): return
            atomic_facts_str, atomic_facts_with_pages = self._extract_facts()
            if atomic_facts_str is None: return
            final_cards_text = self._generate_cards(atomic_facts_str, atomic_facts_with_pages)
            if final_cards_text is None: return
            final_cards = self._parse_and_deduplicate(final_cards_text)
            if final_cards is None: return
            self._add_notes_to_anki(final_cards)
        except Exception as e:
            self.log(f"\n--- A CRITICAL ERROR OCCURRED IN DECK '{self.deck_name}' ---\n{e}\nTraceback: {traceback.format_exc()}")

    def _setup_anki(self):
        self.log(f"Card Type Selected: {self.card_type}")
        self.log(f"Image Strategy Selected: {self.image_strategy}")
        if self.custom_tags:
            self.log(f"Custom Tags: {', '.join(self.custom_tags)}")
        error = setup_anki_deck_and_note_type(self.deck_name, self.note_type_key)
        if error: self.log(f"DECK SETUP ERROR: {error}"); return False
        self.log(f"Anki setup for deck '{self.deck_name}' is correct.")
        return True

    def _process_pdfs(self):
        self.log("\n--- Processing PDF Files ---")
        total_ocr_pages = 0
        for pdf_path in self.progress.tqdm(self.pdf_paths, desc="Processing PDFs"):
            pdf_bytes = Path(pdf_path).read_bytes()
            self.combined_pdf_hash.update(pdf_bytes)
            text, images_on_page, ocr_log = get_pdf_content(pdf_path, self.pdf_cache_dir)
            if "Error:" in text and not images_on_page:
                self.log(text); return False
            if ocr_log:
                self.log(f"   > OCR performed on '{Path(pdf_path).name}':")
                for log_entry in ocr_log: self.log(f"     - {log_entry}")
                total_ocr_pages += len(ocr_log)
            self.full_text += f"\n\n--- Content from {Path(pdf_path).name} ---\n{text}"
            for page_num, images in images_on_page.items():
                if page_num not in self.all_images_by_page: self.all_images_by_page[page_num] = []
                self.all_images_by_page[page_num].extend(images)
        if total_ocr_pages > 0:
            self.log(f"   > OCR Summary: A total of {total_ocr_pages} page(s) were processed using OCR.")
        self.log("All PDF files processed and cached.")
        return True

    def _extract_facts(self):
        self.log("\n--- Cleaning and Structuring Text ---")
        cleaned_text = clean_pdf_text(self.full_text)
        self.log(f"\n--- AI Pass 1: Extracting atomic facts (Batch Size: {self.batch_size}) ---")
        extractor_model = "gemini-1.5-flash-latest"
        self.log(f"Using cheaper model for extraction: {extractor_model}")
        pages = cleaned_text.split("--- Page ")[1:]
        all_pages = [(p_num, f"--- Page {p_num} ---\n{page_text}") for p_num, page_text in enumerate(pages, 1) if page_text.strip()]
        batched_tasks = []
        for j in range(0, len(all_pages), self.batch_size):
            batch = all_pages[j:j+self.batch_size]
            page_numbers = [str(p[0]) for p in batch]
            combined_text = "\n\n".join([p[1] for p in batch])
            batched_tasks.append((f"{page_numbers[0]}-{page_numbers[-1]}", combined_text))
        atomic_facts_by_page_range = {}
        request_timestamps = deque()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            def extract_facts_from_batch(task_data):
                page_range, batch_content = task_data
                while True:
                    current_time = time.time()
                    while request_timestamps and request_timestamps[0] < current_time - 60: request_timestamps.popleft()
                    if len(request_timestamps) < self.rpm_limit:
                        request_timestamps.append(current_time); break
                    else: time.sleep(60 - (current_time - request_timestamps[0]) + 0.1)
                response = call_gemini(self.prompts_dict['extractor'] + f"\n\n--- TEXT ---\n{batch_content}", self.api_keys["GEMINI_API_KEY"], model_name=extractor_model)
                return page_range, response
            results = list(self.progress.tqdm(executor.map(extract_facts_from_batch, batched_tasks), total=len(batched_tasks), desc="Extracting Facts in Batches"))
        atomic_facts_with_pages = ""
        for page_range, page_facts in results:
            if page_facts == "API_LIMIT_REACHED":
                self.log(("\n--- PROCESS STOPPED: API LIMIT REACHED ---\n...")); raise Exception("API Limit Reached")
            elif "API_" in page_facts:
                self.log(f"\nWARNING (Page Range {page_range}): {page_facts}. This batch will be skipped."); continue
            if page_facts and "Error" not in page_facts:
                atomic_facts_by_page_range[page_range] = page_facts
                atomic_facts_with_pages += f"--- Page(s) {page_range} ---\n{page_facts}\n"
        atomic_facts_str = "".join(atomic_facts_by_page_range.values())
        if not atomic_facts_str.strip():
            self.log("ERROR: AI Pass 1 failed to extract any facts from any page."); return None, None
        self.log(f"Pass 1 complete. Found {len(atomic_facts_str.splitlines())} facts.")
        return atomic_facts_str, atomic_facts_with_pages

    def _generate_cards(self, atomic_facts_str, atomic_facts_with_pages):
        card_type_slug = self.card_type.replace(" ", "_").lower()
        ai_cache_key = f"{self.combined_pdf_hash.hexdigest()}_{card_type_slug}_v5.txt"
        ai_cache_file = self.ai_cache_dir / ai_cache_key
        if ai_cache_file.exists():
            self.log("\n--- Found Cached AI Response! Skipping card generation. ---")
            return ai_cache_file.read_text(encoding='utf-8')
        self.log("\n--- No cached response found. Generating new cards with AI... ---")
        if "Basic" in self.card_type:
            pass2_prompt = self.prompts_dict['builder'].format(atomic_facts=atomic_facts_str)
        elif "Atomic Cloze" in self.card_type:
            pass2_prompt = self.prompts_dict['cloze_builder'].format(atomic_facts_with_pages=atomic_facts_with_pages)
        else:
            pass2_prompt = self.prompts_dict['conceptual_cloze_builder'].format(atomic_facts_with_pages=atomic_facts_with_pages)
        final_cards_text = call_gemini(pass2_prompt, self.api_keys["GEMINI_API_KEY"])
        if "API_" in final_cards_text:
            self.log(f"\nERROR: AI card generation failed. Reason: {final_cards_text}"); return None
        if final_cards_text.strip():
            self.ai_cache_dir.mkdir(exist_ok=True)
            ai_cache_file.write_text(final_cards_text, encoding='utf-8')
            self.log("Saved new AI response to cache.")
        return final_cards_text

    def _parse_and_deduplicate(self, raw_text_response):
        if not raw_text_response or "Error" in raw_text_response:
            self.log(f"ERROR: AI card generation failed."); return None
        self.log("\n--- Parsing AI Response ---")
        final_cards = []
        card_blocks = raw_text_response.strip().split('\n\n')
        for block in card_blocks:
            card = None
            if "|||" in block:
                parts = block.split('|||')
                if self.card_type == "Conceptual (Basic Cards)" and len(parts) == 4:
                    front, back, query, page_str = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                    if front and back and query:
                        try: page = int(page_str)
                        except (ValueError, TypeError): page = 1
                        card = {"front": front, "back_text": back, "image_search_query": query, "best_page_for_image": page}
                elif "Cloze" in self.card_type and len(parts) == 3:
                    oq, sentence, query = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    if oq and sentence and query:
                        card = {"original_question": oq, "sentence_html": sentence, "image_search_query": query, "page": 1}
            if card: final_cards.append(card)
        if not final_cards:
            self.log(f"CRITICAL ERROR: No valid cards could be parsed from the AI's response."); return None
        self.log(f"Successfully parsed {len(final_cards)} cards.")
        unique_cards = list({(card.get('front') or card.get('original_question')): card for card in final_cards}.values())
        if len(unique_cards) < len(final_cards):
            self.log(f"   > INFO: Found and removed {len(final_cards) - len(unique_cards)} duplicate card(s).")
        return unique_cards

    def _find_best_image_for_card(self, card_data: Dict[str, Any], page_ref_int: int) -> str | None:
        if self.image_strategy == "None (Text-Only)":
            return None

        # Attempt to find a suitable image in the PDF first for relevant strategies
        if self.image_strategy in ["PDF Only (Fastest, Free)", "PDF Priority (Smart)"]:
            page_images = self.all_images_by_page.get(page_ref_int, [])
            if page_images:
                high_quality_images = [img for img in page_images if is_image_high_quality_heuristic(img)]
                if high_quality_images:
                    self.log(f"   > Found high-quality image in PDF on page {page_ref_int}.")
                    best_image = max(high_quality_images, key=len)
                    b64_image = base64.b64encode(best_image).decode('utf-8')
                    return f'<img src="data:image/jpeg;base64,{b64_image}">'
        
        # If no suitable PDF image was found, fall back to web search if applicable
        if self.image_strategy in ["Wikimedia (Educational, Free)", "PDF Priority (Smart)"]:
            if self.image_strategy == "PDF Priority (Smart)":
                self.log(f"   > No suitable PDF image found. Falling back to web search.")
            search_query = card_data.get("image_search_query") or card_data.get("front")
            self.log(f"   > Searching Wikimedia for '{search_query}'...")
            image_html = search_wikimedia_for_image(search_query)
            if image_html:
                self.log(f"   > Found and added image from web.")
                return image_html
        
        return None

    def _add_notes_to_anki(self, final_cards):
        self.log(f"\n--- Adding Cards to Deck: '{self.deck_name}' ---")
        cards_added, cards_skipped, cards_failed = 0, 0, 0
        for card_data in self.progress.tqdm(final_cards, desc="Adding Cards to Anki"):
            try:
                fields, final_note_type_key = {}, self.note_type_key
                page_ref_raw = card_data.get('best_page_for_image') or card_data.get('page', 1)
                page_ref_str = str(page_ref_raw).strip()
                page_ref_int = int(page_ref_str.split('-')[0])
                source_text = f"{Path(self.pdf_paths[0]).stem} - Pg {page_ref_str}"
                image_html = self._find_best_image_for_card(card_data, page_ref_int)
                if self.note_type_key == "basic":
                    front, back_text = card_data.get("front"), card_data.get("back_text")
                    if front and back_text:
                        back_html = build_html_from_tags(back_text, self.enabled_colors)
                        fields = {"Front": front, "Back": back_html, "Source": source_text, "Image": image_html or ""}
                    else: cards_skipped += 1; continue
                else: # Cloze Cards
                    sentence_html = card_data.get("sentence_html")
                    original_question = card_data.get("original_question")
                    if not (sentence_html and original_question): cards_skipped += 1; continue
                    if "{{" not in sentence_html:
                        self.log(f"   > WARNING (Page {page_ref_str}): AI failed to generate a cloze. Converting to Basic card.")
                        fields = {"Front": original_question, "Back": sentence_html, "Source": source_text, "Image": image_html or ""}
                        final_note_type_key = "basic"
                    else:
                        fields = {"Text": sentence_html, "Extra": original_question, "Source": source_text, "Image": image_html or ""}
                if not fields: cards_skipped += 1; continue
                _, error = add_note_to_anki(self.deck_name, final_note_type_key, fields, self.pdf_paths[0], self.custom_tags)
                if error:
                    if "duplicate" in error: cards_skipped += 1
                    else: self.log(f"   > FAILED to add note. Reason: {error}"); cards_failed += 1
                    continue
                cards_added += 1
            except Exception as e:
                cards_failed += 1; self.log(f"ERROR processing card data: '{card_data}' | Exception: {e}")
        self.log(f"\n--- Final Tally ---\nCards Added: {cards_added}\nCards Skipped/Failed: {cards_skipped + cards_failed}")
def generate_all_decks(max_decks: int, master_files, generate_button, log_output, *args):
    log_history = ""
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    def logger(message):
        nonlocal log_history
        timestamp_msg = datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp_msg}] {message}"
        if log_history is None: log_history = ""
        log_history += full_message + "\n"
        print(full_message)
        return log_history
    final_ui_state = [gr.update(), gr.update(interactive=True), gr.update(value="Generate All Decks")]
    log_file_path = None
    try:
        yield logger(f"Starting Anki Deck Generator..."), gr.update(interactive=False), gr.update(value="Processing...")
        from app import LOG_DIR, MAX_LOG_FILES, PDF_CACHE_DIR, AI_CACHE_DIR
        manage_log_files(LOG_DIR, MAX_LOG_FILES)
        log_file_path = LOG_DIR / f"session_log_{session_timestamp}.txt"
        cache_dirs = (PDF_CACHE_DIR, AI_CACHE_DIR)
        deck_inputs_per_deck = 2
        num_core_settings = 8
        expected_arg_len = (max_decks * deck_inputs_per_deck) + num_core_settings
        if len(args) != expected_arg_len:
            yield logger(f"CRITICAL ERROR: Argument mismatch. Expected {expected_arg_len} but received {len(args)}."), *final_ui_state[1:]
            return
        deck_inputs = args[:max_decks * deck_inputs_per_deck]
        other_args = args[max_decks * deck_inputs_per_deck:]
        card_type, image_strategy, enabled_colors, custom_tags_str, extractor_prompt, builder_prompt, cloze_builder_prompt, conceptual_cloze_builder_prompt = other_args
        
        # CORRECTED: The prompts_dict is now simple and correct.
        prompts_dict = {'extractor': extractor_prompt, 'builder': builder_prompt, 'cloze_builder': cloze_builder_prompt, 'conceptual_cloze_builder': conceptual_cloze_builder_prompt}
        
        custom_tags = [tag.strip() for tag in custom_tags_str.split(',') if tag.strip()]
        api_keys = get_api_keys_from_env()
        deck_configs = []
        for i in range(0, len(deck_inputs), deck_inputs_per_deck):
            deck_title, files = deck_inputs[i], deck_inputs[i+1]
            if deck_title and files:
                deck_configs.append((deck_title, files))
        pre_flight_error = run_pre_flight_checks(api_keys, deck_configs)
        if pre_flight_error:
            yield logger(pre_flight_error), *final_ui_state[1:]
            return
        yield logger("Pre-flight checks passed successfully."), gr.update(), gr.update()
        yield logger(f"Found {len(deck_configs)} deck(s) to process."), gr.update(), gr.update()
        for i, (deck_name, files) in enumerate(deck_configs, 1):
            progress = gr.Progress(track_tqdm=True)
            yield logger(f"\n--- Starting Deck {i} of {len(deck_configs)}: '{deck_name}' ---"), gr.update(), gr.update()
            def processor_logger_wrapper(message):
                logger(message)
            processor = DeckProcessor(
                deck_name=deck_name, files=files, api_keys=api_keys, 
                logger=processor_logger_wrapper, progress=progress, card_type=card_type, 
                image_strategy=image_strategy, enabled_colors=enabled_colors, 
                custom_tags=custom_tags, prompts_dict=prompts_dict, 
                cache_dirs=cache_dirs, batch_size=BATCH_SIZE, 
                max_workers=MAX_WORKERS_EXTRACTION, rpm_limit=RPM_LIMIT_FLASH
            )
            processor.run()
            yield log_history, gr.update(), gr.update()
            yield logger(f"--- Finished Deck {i}: '{deck_name}' ---\n"), gr.update(), gr.update()
            if i < len(deck_configs):
                yield logger(f"Pausing for {INTER_DECK_COOLDOWN_SECONDS} seconds..."), gr.update(), gr.update()
                time.sleep(INTER_DECK_COOLDOWN_SECONDS)
        logger("--- All Decks Processed! ---")
    except Exception as e:
        error_message = f"\n--- A CRITICAL UNHANDLED ERROR OCCURRED ---\n{e}\nTraceback: {traceback.format_exc()}"
        logger(error_message)
        print(error_message)
    finally:
        if log_file_path and log_history:
            try:
                with open(log_file_path, "w", encoding="utf-8") as f:
                    f.write(log_history)
                logger(f"Session log saved to: {log_file_path}")
            except Exception as e:
                logger(f"CRITICAL: Failed to save log file. Error: {e}")
        final_ui_state[0] = log_history
        yield tuple(final_ui_state)