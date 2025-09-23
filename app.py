# app.py

from pathlib import Path
from ui import build_ui

# --- Global Configuration ---
SCRIPT_VERSION = "3.0.0 RC1"
PDF_CACHE_DIR = Path(".pdf_cache")
AI_CACHE_DIR = Path(".ai_cache")
LOG_DIR = Path("logs")
MAX_LOG_FILES = 10
MAX_DECKS = 10

# ==============================================================================
# SECTION: SCRIPT LAUNCHER
# ==============================================================================
if __name__ == "__main__":
    cache_dirs = (PDF_CACHE_DIR, AI_CACHE_DIR)
    
    app = build_ui(
        version=SCRIPT_VERSION,
        max_decks=MAX_DECKS,
        cache_dirs=cache_dirs,
        log_dir=LOG_DIR,
        max_log_files=MAX_LOG_FILES
    )
    
    app.launch(server_name="127.0.0.1", debug=True, inbrowser=True)