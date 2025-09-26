import json, hashlib, re, time, traceback
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import fitz, requests, gradio as gr
import pytesseract
from PIL import Image
import markdown2

from utils import get_api_keys_from_env, manage_log_files
from image_finder import ImageFinder, PDFImageSource, WikimediaSource, NLMOpenISource, OpenverseSource, FlickrSource
# --- Correctly import the new manager ---
from model_manager import GeminiModelManager

# --- Define only the necessary constants ---
MAX_WORKERS_EXTRACTION = 8
BATCH_SIZE = 3

# --- Configuration Constants ---
ANKI_CONNECT_URL = "http://127.0.0.1:8765"
INTER_DECK_COOLDOWN_SECONDS = 30
HTML_COLOR_MAP = {
    "positive_key_term": "#87CEFA", "negative_key_term": "#FF6347",
    "example": "#90EE90", "mnemonic_tip": "#FFD700"
}
NOTE_TYPE_CONFIG = {
    "basic": {
        "modelName": "Anki Deck Generator - Basic",
        "fields": ["Front", "Back", "Image", "Source"],
        "css": """.card { font-family: Arial; font-size: 20px; text-align: center; } 
                 img { max-height: 500px; } 
                 ul { display: inline-block; text-align: left; }""",
        "templates": [{"Name": "Card 1", "Front": "{{Front}}", "Back": "{{FrontSide}}\n\n<hr id=answer>\n\n{{Back}}\n\n<br><br>{{#Image}}{{Image}}{{/Image}}\n<div style='font-size:12px; color:grey;'>{{Source}}</div>"}],
        "function_tool": {
            "name": "create_anki_card",
            "description": "Creates a single Anki card based on a conceptual chunk of facts.",
            "parameters": {
                # --- CORRECTION: Types must be lowercase ---
                "type": "object",
                "properties": {
                    "Front": {"type": "string", "description": "The specific, 2nd or 3rd-order question for the card's front."},
                    "Back": {"type": "string", "description": "The detailed answer, formatted with hyphenated bullet points and custom tags."},
                    "Page_numbers": {"type": "array", "items": {"type": "integer"}, "description": "A JSON array of unique integer page numbers from the source facts."},
                    "Search_Query": {"type": "string", "description": "A concise, 2-4 word search query for finding a relevant medical diagram."}
                }, "required": ["Front", "Back", "Page_numbers", "Search_Query"]
            }
        }
    },
    "cloze": {
        "modelName": "Anki Deck Generator - Cloze",
        "fields": ["Text", "Extra", "Image", "Source"],
        "isCloze": True,
        "css": """.card { font-family: Arial; font-size: 20px; text-align: center; } .cloze { font-weight: bold; color: blue; } img { max-height: 500px; }""",
        "templates": [{"Name": "Cloze Card", "Front": "{{cloze:Text}}", "Back": "{{cloze:Text}}\n\n<br>{{Extra}}\n<br><br>{{#Image}}{{Image}}{{/Image}}\n<div style='font-size:12px; color:grey;'>{{Source}}</div>"}],
        "function_tool": {
            "name": "create_cloze_card",
            "description": "Creates a single Anki cloze-deletion card from a fact.",
            "parameters": {
                # --- CORRECTION: Types must be lowercase ---
                "type": "object",
                "properties": {
                    "Context_Question": {"type": "string", "description": "A simple question that provides context for the cloze sentence."},
                    "Sentence_HTML": {"type": "string", "description": "The full sentence containing the cloze deletion in the format {{c1::keyword}}."},
                    "Source_Page": {"type": "string", "description": "The source page number(s) for this fact, as a string (e.g., 'Page 5')."},
                    "Search_Query": {"type": "string", "description": "A concise, 2-4 word search query for finding a relevant diagram."},
                    "Simple_Search_Query": {"type": "string", "description": "A broader, 1-3 word fallback query with only the main keywords."}
                }, "required": ["Context_Question", "Sentence_HTML", "Source_Page", "Search_Query", "Simple_Search_Query"]
            }
        }
    }
} 


# --- HTML Engine ---
def build_html_from_tags(text: str, enabled_colors: List[str]) -> str:
    tag_map = {
        "<pos>": ("positive_key_term", f"<font color='{HTML_COLOR_MAP['positive_key_term']}'><b>"), "</pos>": ("positive_key_term", "</b></font>"),
        "<neg>": ("negative_key_term", f"<font color='{HTML_COLOR_MAP['negative_key_term']}'><b>"), "</neg>": ("negative_key_term", "</b></font>"),
        "<ex>": ("example", f"<font color='{HTML_COLOR_MAP['example']}'>"), "</ex>": ("example", "</font>"),
        "<tip>": ("mnemonic_tip", f"<font color='{HTML_COLOR_MAP['mnemonic_tip']}'>"), "</tip>": ("mnemonic_tip", "</font>"),
    }
    for tag, (key, replacement) in tag_map.items():
        text = text.replace(tag, replacement if key in enabled_colors else "")

    # Let markdown2 handle list creation and line breaks properly.
    # The "cuddled-lists" extra ensures that lists are tight.
    # The "break-on-newline" extra will convert single newlines to <br> tags correctly.
    html = markdown2.markdown(text, extras=["cuddled-lists", "break-on-newline"])
    
    # This removes any lingering paragraph tags that markdown2 might add, which cause extra space.
    html = html.replace("<p>", "").replace("</p>", "").strip()
    return html

# --- PDF Processing ---
def get_pdf_content(pdf_path: str, pdf_cache_dir: Path) -> Tuple[str, List[str]]:
    pdf_cache_dir.mkdir(exist_ok=True)
    ocr_log = []
    tesseract_warning_issued = False

    def is_text_meaningful(text: str, min_chars: int = 30) -> bool:
        """Checks if a string contains a minimum number of alphanumeric characters."""
        alphanumeric_chars = re.sub(r'[^a-zA-Z0-9]', '', text)
        return len(alphanumeric_chars) >= min_chars

    try:
        pdf_hash = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    except IOError as e:
        return f"Error reading PDF file {Path(pdf_path).name}: {e}", [f"IOError: {e}"]

    cache_file = pdf_cache_dir / f"{pdf_hash}_ocr.txt"
    if cache_file.exists():
        # A bit of a hack to let the user know OCR was used on a cached file
        cached_text = cache_file.read_text(encoding='utf-8')
        if "--- OCR Log ---" in cached_text:
             ocr_log.append("Using cached text that was generated with OCR on a previous run.")
        return cached_text, ocr_log

    text_content = ""
    page_ocr_logs = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text().strip()

                if not is_text_meaningful(page_text):
                    page_ocr_logs.append(f"   > Page {page_num}: Low text quality. Attempting OCR fallback.")
                    try:
                        pix = page.get_pixmap(dpi=300)
                        img_bytes = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_bytes))
                        ocr_text = pytesseract.image_to_string(pil_image)
                        if is_text_meaningful(ocr_text):
                             page_text = ocr_text
                             page_ocr_logs.append(f"     - OCR SUCCESS: Extracted {len(ocr_text)} characters.")
                        else:
                             page_ocr_logs.append(f"     - OCR FAILED: Tesseract found no meaningful text.")

                    except Exception as ocr_error:
                        if not tesseract_warning_issued:
                            page_ocr_logs.append("  [CRITICAL OCR WARNING] Failed to run Tesseract OCR.")
                            page_ocr_logs.append("  SOLUTION: Ensure Tesseract is installed and in your system PATH.")
                            page_ocr_logs.append(f"  (Underlying Error: {ocr_error})")
                            tesseract_warning_issued = True
                        page_text = ""

                text_content += f"--- Page {page_num} ---\n{page_text}\n\n"

    except fitz.errors.FitzError as e:
        return f"Error processing PDF {Path(pdf_path).name}: {e}", [f"FitzError: {e}"]

    if page_ocr_logs:
        ocr_log.append("\n--- OCR Processing Log ---")
        ocr_log.extend(page_ocr_logs)
        # Embed the log in the cache file itself for future reference
        text_content += "\n--- OCR Log ---\n" + "\n".join(page_ocr_logs)

    cache_file.write_text(text_content, encoding='utf-8')
    return text_content, ocr_log

def clean_pdf_text(raw_text: str) -> str:
    cleaned_text = re.sub(r'\n{3,}', '\n\n', raw_text)
    return cleaned_text

# --- Gemini API Call with Function Calling Support ---
def call_gemini(prompt: str, api_key: str, model_name: str, tools: Optional[List[Dict]] = None) -> Any:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    if tools:
        payload["tools"] = tools
        payload["tool_config"] = {"function_calling_config": {"mode": "ANY"}}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        if response.status_code == 429: return "API_LIMIT_REACHED"
        response.raise_for_status()

        response_data = response.json()
        if response_data.get('promptFeedback', {}).get('blockReason'):
            return f"API_SAFETY_BLOCK: {response_data['promptFeedback']['blockReason']}"

        candidate = response_data.get('candidates', [{}])[0]
        content = candidate.get('content', {})
        parts = content.get('parts', [{}])

        function_calls = [part['functionCall'] for part in parts if 'functionCall' in part]
        if function_calls:
            return function_calls

        if 'text' in parts[0]:
            return parts[0]['text']

        return f"Warning: Gemini returned an empty or unexpected response.\n{response.text}"

    except requests.exceptions.RequestException as e:
        return f"Error calling Gemini API: {e}"
    return "Error: Empty response from Gemini."

# --- AnkiConnect ---
def invoke_ankiconnect(action: str, **params: Any) -> Tuple[Any | None, str | None]:
    try:
        response = requests.post(ANKI_CONNECT_URL, json={"action": action, "version": 6, "params": params}, timeout=30)
        response.raise_for_status()
        result = response.json()
        return (result.get('result'), result.get('error'))
    except requests.exceptions.RequestException as e:
        return None, f"Could not connect to AnkiConnect: {e}"

def run_pre_flight_checks(api_keys: Dict[str, str], deck_configs: List[Tuple]) -> Optional[str]:
    version, error = invoke_ankiconnect("version")
    if error or not version: return "CRITICAL ERROR: Could not connect to Anki.\nSOLUTION: Please ensure Anki is running and AnkiConnect is installed."
    if int(version) < 6: return f"CRITICAL ERROR: Your AnkiConnect add-on is outdated (Version {version}). Please update."
    test_response = call_gemini("Hello", api_keys['GEMINI_API_KEY'], model_name="gemini-1.5-flash-latest")
    if "API key not valid" in str(test_response) or "API_KEY_INVALID" in str(test_response):
        return "CRITICAL ERROR: Your Gemini API Key appears to be invalid. Please check your .env file."
    if not deck_configs: return "ERROR: No valid decks configured. Please upload at least one PDF and provide a deck name."
    for _, files in deck_configs:
        for pdf in files:
            if not Path(pdf.name).exists():
                return f"CRITICAL ERROR: The file '{Path(pdf.name).name}' could not be found."
    return None

def setup_anki_deck_and_note_type(deck_name: str, note_type_key: str) -> Optional[str]:
    config = NOTE_TYPE_CONFIG.get(note_type_key)
    if not config: return f"Internal Error: Note type config for '{note_type_key}' not found."
    model_name = config["modelName"]

    _, error = invoke_ankiconnect("createDeck", deck=deck_name)
    if error and "exists" not in error: return f"Failed to create deck '{deck_name}': {error}"

    model_names, error = invoke_ankiconnect("modelNames")
    if error: return f"AnkiConnect Error: {error}"

    if model_name not in model_names:
        params = {"modelName": model_name, "inOrderFields": config["fields"], "css": config["css"], "isCloze": config.get("isCloze", False), "cardTemplates": config["templates"]}
        _, error = invoke_ankiconnect("createModel", **params)
        if error: return f"Failed to create note type '{model_name}': {error}"
    return None

def add_note_to_anki(deck_name: str, note_type_key: str, fields_data: Dict[str, str], source_filename: str, custom_tags: List[str]) -> Tuple[int | None, str | None]:
    config = NOTE_TYPE_CONFIG[note_type_key]
    pdf_tag = f"PDF::{Path(source_filename).stem.replace(' ', '_')}"
    note = {
        "deckName": deck_name, "modelName": config["modelName"], "fields": fields_data,
        "options": {"allowDuplicate": False}, "tags": [pdf_tag] + custom_tags
    }
    return invoke_ankiconnect("addNote", note=note)

# --- Main Deck Processor Class ---
class DeckProcessor:
    def __init__(self, deck_name, files, api_keys, logger, progress, card_type, image_sources_config, enabled_colors, custom_tags, prompts_dict, cache_dirs, clip_model):
        self.deck_name = deck_name
        self.files = files
        self.api_keys = api_keys
        self.logger_func = logger
        self.progress = progress
        self.card_type = card_type
        self.enabled_colors = enabled_colors
        self.custom_tags = custom_tags
        self.prompts_dict = prompts_dict
        self.pdf_cache_dir, self.ai_cache_dir = cache_dirs
        self.clip_model = clip_model['model'] if clip_model else None
        self.note_type_key = "cloze" if "Cloze" in self.card_type else "basic"
        self.full_text = ""
        self.pdf_paths = [file.name for file in self.files]
        self.combined_pdf_hash = hashlib.sha256(b''.join(Path(p).read_bytes() for p in self.pdf_paths)).hexdigest()
        self.pro_model = None
        self.flash_model = None
        self.rpm_limit_flash = None
        self.pdf_images_cache = [] 
        if self.clip_model:
            strategies = [PDFImageSource(), WikimediaSource(), NLMOpenISource(), OpenverseSource(api_key=self.api_keys.get("OPENVERSE_API_KEY"), api_key_name="OPENVERSE_API_KEY"), FlickrSource(api_key=self.api_keys.get("FLICKR_API_KEY"), api_key_name="FLICKR_API_KEY")]
            self.image_finder = ImageFinder([s for s in strategies if s.name in image_sources_config])
        else:
            self.logger_func("WARNING: CLIP Model not loaded. Image searching will be disabled.")
            self.image_finder = None

    def log(self, message): self.logger_func(message)

    def run(self):
        try:
            manager = GeminiModelManager(self.api_keys["GEMINI_API_KEY"])
            optimal_models = manager.get_optimal_models()
            if not optimal_models:
                self.log("CRITICAL ERROR: Could not determine optimal Gemini models. Stopping deck generation.")
                return
            self.pro_model = optimal_models['pro_model_name']
            self.flash_model = optimal_models['flash_model_name']
            self.rpm_limit_flash = optimal_models['flash_model_rpm']

            if not self._setup_anki(): return
            if not self._process_pdfs(): return
            if self.image_finder:
                self.log("\n--- Pre-caching images from PDF(s) ---")
                # Create a temporary instance just to use its helper method
                pdf_image_extractor = PDFImageSource()
                for pdf_path in self.pdf_paths:
                    extracted = pdf_image_extractor._extract_images_and_context(pdf_path)
                    if extracted:
                        self.pdf_images_cache.extend(extracted)
                self.log(f"Found and cached {len(self.pdf_images_cache)} images from source PDF(s).")
            # ------------------------------------
            
            structured_facts = self._extract_facts()
            if not structured_facts:
                self.log(f"\n--- SKIPPING DECK: '{self.deck_name}' ---")
                self.log("   > Reason: The AI could not extract any meaningful facts from the source PDF(s).")
                return
            final_cards_data = self._generate_cards(structured_facts)
            if final_cards_data is None: return
            final_cards = self._parse_and_deduplicate(final_cards_data)
            if final_cards is None: return
            self._add_notes_to_anki(final_cards)
        except Exception as e:
            self.log(f"\n--- A CRITICAL ERROR OCCURRED IN DECK '{self.deck_name}' ---\n{e}\nTraceback: {traceback.format_exc()}")

    def _setup_anki(self):
        self.log(f"Card Type Selected: {self.card_type}")
        if self.custom_tags: self.log(f"Custom Tags: {', '.join(self.custom_tags)}")
        error = setup_anki_deck_and_note_type(self.deck_name, self.note_type_key)
        if error: self.log(f"DECK SETUP ERROR: {error}"); return False
        self.log(f"Anki setup for deck '{self.deck_name}' is correct.")
        return True

    def _process_pdfs(self):
        self.log("\n--- Processing PDF Files ---")
        for pdf_path in self.progress.tqdm(self.pdf_paths, desc="Processing PDFs"):
            text, ocr_log = get_pdf_content(pdf_path, self.pdf_cache_dir)
            if "Error:" in text:
                self.log(text)
                return False
            self.full_text += f"\n\n--- Content from {Path(pdf_path).name} ---\n{text}"
            if ocr_log:
                for line in ocr_log:
                    self.log(line)
        self.log("All PDF files processed and cached.")
        return True

    def _extract_facts(self) -> Optional[List[Dict[str, Any]]]:
        self.log("\n--- Cleaning and Structuring Text ---")
        cleaned_text = clean_pdf_text(self.full_text)
        self.log(f"\n--- AI Pass 1: Extracting atomic facts in parallel (Batch Size: {BATCH_SIZE}) ---")
        extractor_model = self.flash_model

        page_pattern = re.compile(r'--- Page (\d+) ---\n(.*?)(?=--- Page \d+ ---|\Z)', re.DOTALL)
        all_pages = [(int(num), content) for num, content in page_pattern.findall(cleaned_text) if len(content.strip()) > 50]

        if not all_pages:
            self.log("ERROR: No pages with sufficient text content found after cleaning."); return None
        
        batched_tasks = []
        for i in range(0, len(all_pages), BATCH_SIZE):
            batch = all_pages[i:i + BATCH_SIZE]
            combined_text = "\n\n".join([f"--- Page {p[0]} ---\n{p[1]}" for p in batch])
            page_numbers_in_batch = [p[0] for p in batch]
            batched_tasks.append((page_numbers_in_batch, combined_text))

        request_timestamps = deque()
        lock = threading.Lock()
        all_extracted_facts = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS_EXTRACTION) as executor:
            def extract_facts_from_batch(task_data):
                page_nums, batch_content = task_data
                page_range_str = str(page_nums[0]) if len(page_nums) == 1 else f"{page_nums[0]}-{page_nums[-1]}"
                with lock:
                    while True:
                        current_time = time.time()
                        while request_timestamps and request_timestamps[0] <= current_time - 60:
                            request_timestamps.popleft()
                        if len(request_timestamps) < self.rpm_limit_flash:
                            request_timestamps.append(current_time)
                            break
                        wait_time = request_timestamps[0] - (current_time - 60) + 0.1
                        time.sleep(wait_time)
                prompt = self.prompts_dict['extractor'] + "\n\n--- TEXT ---\n" + batch_content
                response = call_gemini(prompt, self.api_keys["GEMINI_API_KEY"], model_name=extractor_model)
                was_successful = False
                if isinstance(response, str):
                    match = re.search(r'\[.*\]', response, re.DOTALL)
                    if match:
                        json_string = match.group(0)
                        try:
                            facts_in_batch = json.loads(json_string)
                            if isinstance(facts_in_batch, list):
                                with lock:
                                    all_extracted_facts.extend(facts_in_batch)
                                was_successful = True
                        except json.JSONDecodeError:
                            self.log(f"\nWARNING (Batch {page_range_str}): AI returned a string that looked like JSON but failed to parse.")
                if not was_successful:
                    self.log(f"\nWARNING (Batch {page_range_str}): AI call failed or returned a response without valid JSON. Reason: {response}")

            list(self.progress.tqdm(executor.map(extract_facts_from_batch, batched_tasks), total=len(batched_tasks), desc="Extracting Facts in Batches"))

        if not all_extracted_facts:
            self.log("ERROR: AI Pass 1 failed to extract any facts from any batch."); return None
        
        validated_facts = []
        for i, item in enumerate(all_extracted_facts):
            if isinstance(item, dict) and 'fact' in item and 'page_number' in item:
                try: validated_facts.append({"fact": str(item['fact']), "page_number": int(item['page_number'])})
                except (ValueError, TypeError): self.log(f"WARNING: Discarding malformed fact object at index {i}: {item}")
        
        self.log(f"Pass 1 complete. Found and validated {len(validated_facts)} facts across all batches.")
        return validated_facts

    def _generate_cards(self, facts_json_input: List[Dict[str, Any]]) -> Any:
        card_type_slug = self.card_type.replace(" ", "_").lower()
        facts_hash = hashlib.sha256(json.dumps(facts_json_input, sort_keys=True).encode()).hexdigest()[:16]
        ai_cache_key = f"{self.combined_pdf_hash}_{card_type_slug}_{facts_hash}_v6_fc.json"
        ai_cache_file = self.ai_cache_dir / ai_cache_key
        
        if ai_cache_file.exists():
            self.log("\n--- Found Cached AI Response! Skipping card generation. ---")
            return json.loads(ai_cache_file.read_text(encoding='utf-8'))

        self.log("\n--- No cached response found. Generating new cards with AI... ---")

        prompt, tools = None, None

        if "Basic" in self.card_type:
            prompt = self.prompts_dict['builder'].format(atomic_facts_json=json.dumps(facts_json_input, indent=2))
            tools = [{"function_declarations": [NOTE_TYPE_CONFIG['basic']['function_tool']]}]
        elif "Cloze" in self.card_type:
            atomic_facts_with_pages = ""
            # This logic correctly handles potentially non-sequential page numbers
            pages_to_facts = {}
            for item in facts_json_input:
                pg = item['page_number']
                if pg not in pages_to_facts: pages_to_facts[pg] = []
                pages_to_facts[pg].append(item['fact'])
            
            for page_num in sorted(pages_to_facts.keys()):
                atomic_facts_with_pages += f"--- Page(s) {page_num} ---\n" + "\n".join(pages_to_facts[page_num]) + "\n"

            if "Atomic" in self.card_type:
                prompt = self.prompts_dict['cloze_builder'].format(atomic_facts_with_pages=atomic_facts_with_pages)
            else:
                prompt = self.prompts_dict['conceptual_cloze_builder'].format(atomic_facts_with_pages=atomic_facts_with_pages)
            tools = [{"function_declarations": [NOTE_TYPE_CONFIG['cloze']['function_tool']]}]

        if not prompt or not tools:
            self.log(f"ERROR: Could not determine prompt for card type '{self.card_type}'"); return None

        card_generation_result = call_gemini(prompt, self.api_keys["GEMINI_API_KEY"], model_name=self.pro_model, tools=tools)

        if isinstance(card_generation_result, str) and "API_" in card_generation_result:
            self.log(f"\nERROR: AI card generation failed. Reason: {card_generation_result}"); return None

        if card_generation_result:
            self.ai_cache_dir.mkdir(exist_ok=True) # Add this line
            ai_cache_file.write_text(json.dumps(card_generation_result, indent=2), encoding='utf-8')
            self.log("Saved new AI response to cache.")

        return card_generation_result

    def _parse_and_deduplicate(self, card_function_calls: List[Dict]) -> Optional[List[Dict]]:
        if isinstance(card_function_calls, dict):
            card_function_calls = [card_function_calls]

        if not isinstance(card_function_calls, list):
            self.log("CRITICAL ERROR: AI response was not a list or a single function call object. AI may have failed to follow instructions."); return None

        self.log("\n--- Parsing AI Function Call Response ---")
        final_cards = []
        for call in card_function_calls:
            # The AI response for function calls is nested under 'functionCall'
            call = call.get('functionCall', call)
            args = call.get("args", {})
            card = None

            if call.get("name") == "create_anki_card":
                page_nums = sorted(list(set(args.get("Page_numbers", [1]))))
                card = {
                    "type": "basic", "front": args.get("Front"), "back_text": args.get("Back"),
                    "image_search_query": args.get("Search_Query"), "page_numbers": page_nums
                }
            elif call.get("name") == "create_cloze_card":
                card = {
                    "type": "cloze", "original_question": args.get("Context_Question"),
                    "sentence_html": args.get("Sentence_HTML"), "image_search_query": args.get("Search_Query"),
                    "page_numbers": [int(re.search(r'\d+', args.get("Source_Page", "1")).group())]
                }

            if card: final_cards.append(card)

        if not final_cards:
            self.log("CRITICAL ERROR: No valid card creation calls were parsed from the AI's response."); return None

        self.log(f"Successfully parsed {len(final_cards)} cards.")

        unique_cards_dict = {}
        for card in final_cards:
            key = card.get('front') or card.get('sentence_html')
            if key and key not in unique_cards_dict:
                unique_cards_dict[key] = card
        unique_cards = list(unique_cards_dict.values())

        if len(unique_cards) < len(final_cards):
            self.log(f"   > INFO: Found and removed {len(final_cards) - len(unique_cards)} duplicate card(s).")
        return unique_cards

    def _add_notes_to_anki(self, final_cards: List[Dict]):
        self.log(f"\n--- Adding Cards to Deck: '{self.deck_name}' ---")
        cards_added, cards_skipped, cards_failed = 0, 0, 0
        
        # tqdm gives us the progress bar in the UI
        for card_data in self.progress.tqdm(final_cards, desc="Adding Cards to Anki"):
            try:
                image_html = None
                
                # Get the complete list of pages this card was sourced from.
                # This will be used for the accurate source text on the card itself.
                full_source_page_numbers = card_data.get("page_numbers", [])

                # Check if an image search is warranted
                if self.image_finder and card_data.get("image_search_query"):
                    
                    # 1. Create a list of all possible search queries for the multi-query search.
                    queries = [
                        card_data.get("image_search_query"),
                        card_data.get("simple_search_query")
                    ]
                    # Filter out any empty or None values to get the final list.
                    search_queries = [q for q in queries if q]

                    # 2. Apply the heuristic to create a "focused" list of pages for the fast Tier 1 search.
                    if len(full_source_page_numbers) > 3:
                        focused_search_pages = full_source_page_numbers[:3]
                    else:
                        focused_search_pages = full_source_page_numbers

                    # 3. Call the image finder with all the necessary arguments for the tiered search.
                    image_html = self.image_finder.find_best_image(
                        query_texts=search_queries,
                        clip_model=self.clip_model,
                        pdf_path=self.pdf_paths[0],
                        pdf_images_cache=self.pdf_images_cache,
                        focused_search_pages=focused_search_pages,
                        full_source_pages=full_source_page_numbers
                    )

                # Use the complete, accurate list of pages for the source text.
                page_str = f"Pgs {', '.join(map(str, full_source_page_numbers))}"
                source_text = f"{Path(self.pdf_paths[0]).stem} - {page_str}"

                fields, final_note_type_key = {}, None

                # Build the fields for a "Basic" card
                if card_data['type'] == 'basic':
                    final_note_type_key = 'basic'
                    fields = {
                        "Front": card_data["front"],
                        "Back": build_html_from_tags(card_data["back_text"], self.enabled_colors),
                        "Source": source_text,
                        "Image": image_html or ""
                    }
                # Build the fields for a "Cloze" card
                elif card_data['type'] == 'cloze':
                    sentence_html = card_data["sentence_html"]
                    if not sentence_html or "{{" not in sentence_html:
                        self.log(f"   > WARNING: AI failed to generate a cloze. Converting to Basic card.")
                        final_note_type_key = "basic"
                        fields = {"Front": card_data["original_question"], "Back": sentence_html or " ", "Source": source_text, "Image": image_html or ""}
                    else:
                        final_note_type_key = "cloze"
                        fields = {"Text": sentence_html, "Extra": card_data["original_question"], "Source": source_text, "Image": image_html or ""}

                if not fields:
                    cards_skipped += 1
                    continue

                # Add the finalized note to Anki
                _, error = add_note_to_anki(self.deck_name, final_note_type_key, fields, self.pdf_paths[0], self.custom_tags)
                if error:
                    if "duplicate" in error:
                        cards_skipped += 1
                    else:
                        self.log(f"   > FAILED to add note. Reason: {error}")
                        cards_failed += 1
                else:
                    cards_added += 1
            except Exception as e:
                cards_failed += 1
                # Log a truncated version of the card data to avoid flooding the log
                self.log(f"ERROR processing card data: '{str(card_data)[:500]}...' | Exception: {e}")

        self.log(f"\n--- Final Tally ---\nCards Added: {cards_added}\nCards Skipped/Failed: {cards_skipped + cards_failed}")

# --- Main Generator Function (CORRECTED) ---
def generate_all_decks(max_decks: int, *args):
    # This robustly unpacks all arguments passed from the UI
    master_files, generate_button, log_output, clip_model, *remaining_args = args
    
    log_history = ""
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    def logger(message):
        nonlocal log_history
        timestamp_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
        log_history += timestamp_msg
        print(timestamp_msg.strip())
        return log_history

    final_ui_state = [gr.update(), gr.update(interactive=True), gr.update(value="Generate All Decks")]
    log_file_path = None
    try:
        yield logger("Starting Anki Deck Generator..."), gr.update(interactive=False), gr.update(value="Processing...")
        from app import LOG_DIR, MAX_LOG_FILES, PDF_CACHE_DIR, AI_CACHE_DIR
        manage_log_files(LOG_DIR, MAX_LOG_FILES)
        log_file_path = LOG_DIR / f"session_log_{session_timestamp}.txt"
        cache_dirs = (PDF_CACHE_DIR, AI_CACHE_DIR)

        # Correctly separate deck inputs from other settings
        deck_inputs_flat = remaining_args[:max_decks * 2]
        settings_and_prompts = remaining_args[max_decks * 2:]
        
        card_type, image_sources, enabled_colors, custom_tags_str, *prompts = settings_and_prompts
        
        prompts_dict = {'extractor': prompts[0], 'builder': prompts[1], 'cloze_builder': prompts[2], 'conceptual_cloze_builder': prompts[3]}
        
        custom_tags = [tag.strip() for tag in custom_tags_str.split(',') if tag.strip()]
        api_keys = get_api_keys_from_env()
        
        deck_configs = []
        for i in range(0, len(deck_inputs_flat), 2):
            deck_title, files = deck_inputs_flat[i], deck_inputs_flat[i+1]
            if deck_title and files: deck_configs.append((deck_title, files))

        if pre_flight_error := run_pre_flight_checks(api_keys, deck_configs):
            yield logger(pre_flight_error), *final_ui_state[1:]
            return
            
        yield logger("Pre-flight checks passed."), gr.update(), gr.update()
        
        for i, (deck_name, files) in enumerate(deck_configs, 1):
            progress = gr.Progress(track_tqdm=True)
            yield logger(f"\n--- Starting Deck {i} of {len(deck_configs)}: '{deck_name}' ---"), gr.update(), gr.update()
            
            processor = DeckProcessor(
                deck_name=deck_name, files=files, api_keys=api_keys, 
                logger=logger, progress=progress, card_type=card_type, 
                image_sources_config=image_sources, enabled_colors=enabled_colors, 
                custom_tags=custom_tags, prompts_dict=prompts_dict, 
                cache_dirs=cache_dirs, clip_model=clip_model
            )
            processor.run()
            yield log_history, gr.update(), gr.update()
            yield logger(f"--- Finished Deck {i}: '{deck_name}' ---\n"), gr.update(), gr.update()
            if i < len(deck_configs):
                time.sleep(INTER_DECK_COOLDOWN_SECONDS)
                
        logger("--- All Decks Processed! ---")
    except Exception as e:
        logger(f"\n--- A CRITICAL UNHANDLED ERROR OCCURRED ---\n{e}\nTraceback: {traceback.format_exc()}")
    finally:
        if log_file_path and log_history:
            log_file_path.write_text(log_history, encoding="utf-8")
            logger(f"Session log saved to: {log_file_path}")
        final_ui_state[0] = log_history
        yield tuple(final_ui_state)