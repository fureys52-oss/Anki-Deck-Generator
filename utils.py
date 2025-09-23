# utils.py

import os
import shutil
import re
import requests
import base64
import io
from pathlib import Path
from typing import Dict, Tuple, List, Any

from dotenv import load_dotenv
import gradio as gr
from PIL import Image
import pytesseract

# --- Global Constants for Image Heuristics ---
MIN_IMAGE_SIZE_BYTES = 5 * 1024
MAX_ASPECT_RATIO = 3.0
MAX_IMAGE_DIMENSION = 1000

# --- File and Cache Management ---
def manage_log_files(log_dir: Path, max_logs: int):
    log_dir.mkdir(exist_ok=True)
    log_files = sorted(log_dir.glob('*.txt'), key=os.path.getmtime)
    while len(log_files) >= max_logs:
        os.remove(log_files[0])
        log_files.pop(0)

def clear_cache(pdf_cache_dir: Path, ai_cache_dir: Path) -> str:
    results = []
    if pdf_cache_dir.exists():
        try:
            shutil.rmtree(pdf_cache_dir)
            results.append("✅ PDF cache cleared.")
        except Exception as e:
            results.append(f"❌ Error clearing PDF cache: {e}")
    else:
        results.append("ℹ️ PDF cache not found.")
    
    if ai_cache_dir.exists():
        try:
            shutil.rmtree(ai_cache_dir)
            results.append("✅ AI cache cleared.")
        except Exception as e:
            results.append(f"❌ Error clearing AI cache: {e}")
    else:
        results.append("ℹ️ AI cache not found.")
    
    pdf_cache_dir.mkdir(exist_ok=True)
    ai_cache_dir.mkdir(exist_ok=True)
    
    return " ".join(results)

# --- Configuration and API Keys ---
def get_api_keys_from_env() -> Dict[str, str]:
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / '.env'
    if not env_path.exists():
        raise FileNotFoundError(f"Error: .env file not found in {script_dir}. Please create one from the env.template file.")
    load_dotenv(dotenv_path=env_path)
    keys = ("GEMINI_API_KEY", "GOOGLE_SEARCH_API_KEY", "GOOGLE_CSE_ID")
    loaded_keys = {key: os.getenv(key) for key in keys}
    if not loaded_keys.get("GEMINI_API_KEY"):
        raise ValueError(f"Error: GEMINI_API_KEY is missing or empty in {env_path}.")
    return loaded_keys

# --- PDF and UI Helpers ---
def guess_lecture_details(file: gr.File) -> Tuple[str, str]:
    if not file: return "01", ""
    file_stem = Path(file.name).stem
    num_guess = "1"
    keyword_pattern = r'(lecture|lec|session|s|chapter|chap|module|mod|unit|part|l)[\s_-]*(\d+(?:(?:[\s\/&-]| and )\d+)*)'
    if match := re.search(keyword_pattern, file_stem, re.IGNORECASE):
        num_guess = match.group(2).strip()
    elif match := re.search(r'^\s*(\d+(?:(?:[\s\/&-]| and )\d+)*)', file_stem):
        num_guess = match.group(1).strip()
    if len(num_guess) == 1: num_guess = f"0{num_guess}"
    name_guess = file_stem
    name_guess = re.sub(r'\b[A-Z]+[\s_-]*\d+([\s-]*\d+)*\b', '', name_guess, flags=re.IGNORECASE)
    name_guess = re.sub(keyword_pattern, '', name_guess, flags=re.IGNORECASE)
    name_guess = re.sub(r'^\s*' + re.escape(num_guess) + r'[\s_-]*', '', name_guess, re.IGNORECASE)
    name_guess = re.sub(r'[\s_-]+', ' ', name_guess).strip()
    return num_guess, name_guess.title()

def update_decks_from_files(files: List[gr.File], max_decks: int) -> List[Any]:
    updates = []
    num_files = len(files) if files else 0
    if num_files > 0: updates.append(gr.update(label=f"{num_files} File(s) Loaded"))
    else: updates.append(gr.update(label="Upload PDFs to assign to decks below"))
    
    for i in range(max_decks):
        if i < num_files:
            num, name = guess_lecture_details(files[i])
            file_name = Path(files[i].name).name
            deck_title_str = f"L{num} - {name}"
            accordion_label_str = f"Deck {i+1} - {file_name}"
            updates.extend([
                gr.update(visible=True, label=accordion_label_str), 
                gr.update(value=deck_title_str), 
                gr.update(value=[files[i].name], visible=False)
            ])
        else:
            updates.extend([
                gr.update(visible=False, label=f"Deck {i+1}"), 
                gr.update(value=""), 
                gr.update(value=[], visible=False)
            ])
    return updates

# --- Image Processing and Web Search ---
def is_image_high_quality_heuristic(image_bytes: bytes) -> bool:
    if not image_bytes: return False
    if len(image_bytes) < MIN_IMAGE_SIZE_BYTES: return False
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            if width < 50 or height < 50: return False
            ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            if ratio > MAX_ASPECT_RATIO: return False
    except Exception: 
        return False
    return True

def optimize_image(image_bytes: bytes) -> bytes | None:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, (0, 0), image.convert("RGBA"))
            image = background
        
        image = image.convert("RGB")

        if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
        
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='JPEG', quality=85, optimize=True)
        return byte_arr.getvalue()
    except Exception as e:
        print(f"Error optimizing image: {e}")
        return None

def download_image(url: str) -> bytes | None:
    try:
        headers = {'User-Agent': 'AnkiDeckGenerator/1.0'}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def search_wikimedia_for_image(query: str) -> str | None:
    WIKIMEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
    headers = {'User-Agent': 'AnkiDeckGenerator/1.0'}
    search_query = f"{query} english diagram"
    params = {"action": "query", "format": "json", "list": "search", "srsearch": search_query, "srnamespace": "6", "srlimit": "5"}
    
    try:
        response = requests.get(WIKIMEDIA_API_URL, params=params, timeout=10, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("query", {}).get("search"): return None

        candidate_titles = [r["title"] for r in data["query"]["search"] if any(r["title"].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.svg'])]
        if not candidate_titles: return None

        image_params = {"action": "query", "format": "json", "titles": "|".join(candidate_titles), "prop": "imageinfo", "iiprop": "url|size|mime"}
        image_response = requests.get(WIKIMEDIA_API_URL, params=image_params, timeout=10, headers=headers)
        image_response.raise_for_status()
        image_data = image_response.json().get("query", {}).get("pages", {})
        
        all_images_info = [page["imageinfo"][0] for page in image_data.values() if "imageinfo" in page]
        if not all_images_info: return None
        
        sorted_candidates = sorted(all_images_info, key=lambda x: x["size"], reverse=True)

        for candidate in sorted_candidates:
            image_bytes = download_image(candidate["url"])
            if not image_bytes: continue

            is_valid = False
            if candidate["mime"] == "image/svg+xml":
                if len(image_bytes) > 2048:
                    is_valid = True
            else:
                if is_image_high_quality_heuristic(image_bytes):
                    is_valid = True
            
            if is_valid:
                optimized_bytes = optimize_image(image_bytes)
                if optimized_bytes:
                    b64_image = base64.b64encode(optimized_bytes).decode('utf-8')
                    return f'<img src="data:image/jpeg;base64,{b64_image}">'
        
    except requests.RequestException as e:
        print(f"Error searching Wikimedia: {e}")
    
    return None

def perform_ocr_on_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return ""