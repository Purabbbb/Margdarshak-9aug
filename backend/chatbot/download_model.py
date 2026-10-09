"""
download_model.py
-----------------
Prepares and verifies the lightweight ML intent classifier model.
No external network downloads or heavy weights required.
"""

from pathlib import Path
from chatbot.intent_classifier import get_classifier

def download_model():
    print("Initializing and caching lightweight ML intent classifier...")
    clf = get_classifier()
    test_res = clf.classify("hello")
    print(f"Lightweight classifier initialized successfully: {test_res}")

if __name__ == "__main__":
    download_model()