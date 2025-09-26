# model_manager.py (Corrected)

import requests
from typing import List, Dict, Any, Optional

# Updated based on the user-provided screenshot of current free-tier models.
KNOWN_MODELS = [
    # Pro Models (for high-quality generation)
    {"name": "gemini-2.5-pro", "role": "pro", "quality_rank": 5},
    {"name": "gemini-1.5-pro", "role": "pro", "quality_rank": 4}, # Kept for future-proofing

    # Flash Models (for high-volume extraction)
    {"name": "gemini-2.5-flash-lite", "role": "flash", "rpm": 15, "rpd": 1000},
    {"name": "gemini-2.5-flash", "role": "flash", "rpm": 10, "rpd": 250},
    {"name": "gemini-2.0-flash-001", "role": "flash", "rpm": 15, "rpd": 200},
]

class GeminiModelManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.available_models = self._list_available_models()

    def _list_available_models(self) -> List[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            return [model['name'].replace('models/', '') for model in data.get('models', [])]
        except requests.RequestException as e:
            print(f"[Model Manager] CRITICAL: Could not fetch model list from Google API: {e}")
            return []

    def get_optimal_models(self) -> Optional[Dict[str, Any]]:
        if not self.available_models:
            print("[Model Manager] No available models found. Cannot select optimal configuration.")
            return None

        print("\n--- Model Discovery ---")
        print(f"Found {len(self.available_models)} available models from API.")

        available_pro_models = [m for m in KNOWN_MODELS if m['role'] == 'pro' and m['name'] in self.available_models]
        if not available_pro_models:
            print("[Model Manager] CRITICAL: No known 'Pro' model is currently available via the API.")
            return None
        
        best_pro_model = sorted(available_pro_models, key=lambda x: x['quality_rank'], reverse=True)[0]
        
        available_flash_models = [m for m in KNOWN_MODELS if m['role'] == 'flash' and m['name'] in self.available_models]
        if not available_flash_models:
            print("[Model Manager] CRITICAL: No known 'Flash' model is currently available via the API.")
            return None
            
        best_flash_model = sorted(available_flash_models, key=lambda x: (x['rpm'], x['rpd']), reverse=True)[0]

        result = {
            "pro_model_name": best_pro_model['name'],
            "flash_model_name": best_flash_model['name'],
            "flash_model_rpm": best_flash_model['rpm']
        }
        
        print(f"Selected PRO model for card generation: {result['pro_model_name']}")
        print(f"Selected FLASH model for fact extraction: {result['flash_model_name']} (RPM Limit: {result['flash_model_rpm']})")
        print("-----------------------\n")

        return result