# test_suite.py (Version 8.2 - Final Bug Fixes)
import os
import sys
import json
import base64
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

# --- Constants ---
LOG_DIR = Path("test_logs")
MAX_LOG_FILES = 10

# --- Setup Imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from app import (
        get_api_keys_from_env, get_pdf_content, invoke_ankiconnect, 
        guess_lecture_details, call_gemini, call_gemini_vision
    )
    from prompts import (
        BUILDER_PROMPT, CLOZE_BUILDER_PROMPT, CONCEPTUAL_CLOZE_BUILDER_PROMPT, 
        AUDITOR_PROMPT, VERIFIER_PROMPT
    )
except ImportError as e:
    print(f"[FATAL] Could not import from app.py or prompts.py. Error: {e}")
    sys.exit(1)

# --- Logging Class ---
class Logger:
    def __init__(self, filepath, original_stream):
        self.terminal = original_stream
        self.log_file = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    log_files = sorted(LOG_DIR.glob('*.txt'), key=os.path.getmtime)
    while len(log_files) >= MAX_LOG_FILES:
        os.remove(log_files[0])
        log_files.pop(0)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = LOG_DIR / f"test_log_{timestamp}.txt"
    
    sys.stdout = Logger(log_filename, sys.stdout)
    sys.stderr = Logger(log_filename, sys.stderr)
    print(f"--- Logging test output to {log_filename} ---")

# --- Mock Objects and Helper Functions ---
class MockGradioFile:
    def __init__(self, name):
        self.name = name

def run_test(name, func, is_live=False):
    print(f"- Running Test: {name}... ", end="")
    if is_live: print("[LIVE API CALL]... ", end="")
    sys.stdout.flush()
    try:
        result, error = func()
        if error:
            print(f"[FAIL]\n   -> Error: {error}")
            return False
        else:
            print("[PASS]")
            return True
    except Exception as e:
        import traceback
        print(f"[FAIL]\n   -> An unexpected exception occurred: {e}")
        traceback.print_exc(file=sys.stdout)
        return False

def run_hybrid_audit(card_type, builder_prompt, prompt_input, api_key, source_text):
    card_gen_prompt = builder_prompt.format(**prompt_input)
    card_json_str = call_gemini(card_gen_prompt, api_key)
    try:
        json_cleaned_str = card_json_str.strip().replace("```json", "").replace("```", "")
        card_to_audit = json.loads(json_cleaned_str)[0]
    except (json.JSONDecodeError, IndexError):
        return None, f"Failed to parse card generation AI's response. Response was:\n{card_json_str}"

    auditor_input = AUDITOR_PROMPT.format(card_type=card_type, source_text=source_text, card_to_audit=json.dumps(card_to_audit, indent=2))
    audit_result_str = call_gemini(auditor_input, api_key)
    try:
        json_cleaned_audit_str = audit_result_str.strip().replace("```json", "").replace("```", "")
        audit_result = json.loads(json_cleaned_audit_str)
    except json.JSONDecodeError:
        return None, f"Failed to parse Auditor AI's JSON response. Response was:\n{audit_result_str}"
        
    if "passed" not in audit_result or "reason" not in audit_result:
        return None, f"Auditor AI response is malformed. Response: {audit_result}"
    if audit_result["passed"]:
        return True, None
    else:
        return None, f"AI-Auditor rejected the card. Reason: {audit_result['reason']}"

MOCK_GOOGLE_IMAGE_BYTES = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")

# --- Test Definitions ---
def test_environment_check():
    env_path = Path(".env")
    if not env_path.exists(): return None, ".env file not found."
    try:
        keys = get_api_keys_from_env()
        if all(keys.values()): return True, None
    except Exception as e: return None, str(e)
    return None, "One or more API keys are missing values."

def test_pdf_parsing():
    pdf_path = Path("test_assets") / "sample.pdf"
    if not pdf_path.exists(): return None, f"'{pdf_path}' not found."
    text_content, _ = get_pdf_content(str(pdf_path))
    if "Error" in text_content or not text_content.strip(): return None, "PDF parsing failed."
    return True, None

def test_ankiconnect_check():
    _, error = invoke_ankiconnect("version")
    if error: return None, f"AnkiConnect connection failed. Is Anki running? Error: {error}"
    return True, None

def test_smart_naming():
    test_cases = {
        "Lec 04 - Microbiology Part 2.pdf": ("04", "Microbiology"),
        "Lec 4 - Immunology.pdf": ("04", "Immunology"),
        "11_Cardiology.pdf": ("11", "Cardiology")
    }
    for filename, expected in test_cases.items():
        num, name = guess_lecture_details(MockGradioFile(name=filename))
        if (num, name) != expected: return None, f"For '{filename}', expected {expected} but got {(num, name)}."
    return True, None

def test_ai_quality_basic_cards():
    api_keys = get_api_keys_from_env()
    pdf_text, _ = get_pdf_content(str(Path("test_assets") / "sample.pdf"))
    prompt_input = {"atomic_facts": pdf_text[:5000], "learning_objectives": ""}
    return run_hybrid_audit("Conceptual Basic", BUILDER_PROMPT, prompt_input, api_keys["GEMINI_API_KEY"], pdf_text[:5000])

def test_ai_quality_atomic_cloze():
    api_keys = get_api_keys_from_env()
    pdf_text, _ = get_pdf_content(str(Path("test_assets") / "sample.pdf"))
    prompt_input = {"atomic_facts_with_pages": pdf_text[:5000]}
    return run_hybrid_audit("Atomic Cloze", CLOZE_BUILDER_PROMPT, prompt_input, api_keys["GEMINI_API_KEY"], pdf_text[:5000])

def test_ai_quality_conceptual_cloze():
    api_keys = get_api_keys_from_env()
    pdf_text, _ = get_pdf_content(str(Path("test_assets") / "sample.pdf"))
    prompt_input = {"atomic_facts_with_pages": pdf_text[:5000]}
    return run_hybrid_audit("Conceptual Cloze", CONCEPTUAL_CLOZE_BUILDER_PROMPT, prompt_input, api_keys["GEMINI_API_KEY"], pdf_text[:5000])

@patch('app.download_image_from_url')
def test_ai_image_verification(mock_download):
    mock_download.return_value = MOCK_GOOGLE_IMAGE_BYTES
    api_keys = get_api_keys_from_env()
    _, images_by_page = get_pdf_content(str(Path("test_assets") / "sample.pdf"))
    
    # --- BUG FIX ---
    # We now select the FIRST image [0] from the list of images on page 2.
    pdf_image_bytes = images_by_page.get(2, [None])[0]
    
    if not pdf_image_bytes: return None, "sample.pdf in test_assets has no image on page 2."
    prompt = VERIFIER_PROMPT.format(card_front="Heart Valves", card_back="Valves of the heart")
    decision = call_gemini_vision(prompt, api_keys["GEMINI_API_KEY"], pdf_image_bytes, MOCK_GOOGLE_IMAGE_BYTES)
    if decision not in ["PDF_IMAGE", "GOOGLE_IMAGE", "NEITHER"]: return None, f"Invalid decision: '{decision}'."
    if decision != "PDF_IMAGE": return None, f"Incorrect choice. Expected PDF_IMAGE, got {decision}."
    return True, None

if __name__ == "__main__":
    setup_logging()
    print("--- Starting Anki Deck Generator Health Check (v8.2) ---")
    
    offline_tests = [
        ("Environment Check (.env)", test_environment_check, False),
        ("PDF Parsing (sample.pdf)", test_pdf_parsing, False),
        ("AnkiConnect Check", test_ankiconnect_check, False),
        ("Smart Naming Function", test_smart_naming, False),
    ]
    
    live_tests = [
        ("AI Quality - Basic Cards", test_ai_quality_basic_cards, True),
        ("AI Quality - Atomic Cloze", test_ai_quality_atomic_cloze, True),
        ("AI Quality - Conceptual Cloze", test_ai_quality_conceptual_cloze, True),
        ("AI Image Verification Logic", test_ai_image_verification, True),
    ]

    print("\n--- Running Offline Integrity Tests ---")
    offline_results = [run_test(name, func, is_live) for name, func, is_live in offline_tests]
    
    print("\n--- Running Live AI Quality Assurance Audits ---")
    live_results = [run_test(name, func, is_live) for name, func, is_live in live_tests]
    
    print("\n--- Health Check Complete ---")
    if all(offline_results) and all(live_results):
        print("✅ All systems, logic, and AI quality checks are nominal.")
    else:
        print("❌ One or more tests failed. Please review the errors above.")