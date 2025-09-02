# backend/memory_store.py
# Simple persistent key-value memory using JSON. Replace with SQLite/FAISS for production.
import json
from pathlib import Path

MEM_PATH = Path(__file__).parent.parent / 'memory.json'

def load_memory():
    if MEM_PATH.exists():
        return json.loads(MEM_PATH.read_text(encoding='utf-8'))
    return {}

def save_memory(mem):
    MEM_PATH.write_text(json.dumps(mem, indent=2), encoding='utf-8')

def remember(key, value):
    mem = load_memory()
    mem[key] = value
    save_memory(mem)

def recall(key):
    mem = load_memory()
    return mem.get(key)
