# backend/server.py
"""
Robust model wrapper:
- auto-detects .gguf model files (Desktop then repo/model/)
- exposes ask(...) and ask_stream(...) for streaming tokens
- stores simple conversation history to backend/history.json
- graceful mocked fallback if llama-cpp-python or model missing
"""
import json
import os
from pathlib import Path
from collections import deque

# Try import; handle missing gracefully
try:
    from llama_cpp import Llama
except Exception:
    Llama = None

MAX_HISTORY = 16
SYSTEM_PROMPT = "You are a helpful, concise assistant that runs locally."
SEARCH_PATHS = [
    Path.home() / "Desktop",
    Path(__file__).resolve().parent.parent / "model"
]
HISTORY_FILE = Path(__file__).resolve().parent / "history.json"

def find_model_file(ext=".gguf"):
    for base in SEARCH_PATHS:
        try:
            if not base.exists():
                continue
            for p in sorted(base.iterdir()):
                if p.is_file() and p.suffix.lower() == ext:
                    return str(p)
        except PermissionError:
            continue
    return None

class Conversation:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.history = deque(maxlen=MAX_HISTORY)

    def build_messages(self, user_message):
        msgs = [{"role": "system", "content": self.system_prompt}]
        for u, a in self.history:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def append(self, user, assistant):
        self.history.append((user, assistant))

conv = Conversation()

_llm = None
_model_path = None

def init_model(path: str = None):
    """
    Initialize the Llama model. Auto-detects first .gguf if path is None.
    """
    global _llm, _model_path
    if Llama is None:
        print("[server] llama-cpp-python not installed; model calls will be mocked.")
        return
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        _model_path = str(p)
    else:
        _model_path = find_model_file()
        if _model_path is None:
            print("[server] No .gguf model found in search paths:", ", ".join(str(x) for x in SEARCH_PATHS))
            return
    try:
        print(f"[server] Loading model from {_model_path} ...")
        _llm = Llama(model_path=_model_path)
        print("[server] Model loaded.")
    except Exception as e:
        _llm = None
        print("[server] Failed to initialize model:", e)

def _extract_from_choice_chunk(choice):
    """
    When streaming with llama-cpp-python you may receive small dicts.
    This helper attempts to extract incremental text tokens.
    """
    # chat style delta (OpenAI-like)
    delta = choice.get("delta") if isinstance(choice, dict) else None
    if isinstance(delta, dict):
        # sometimes delta can have 'content'
        content = delta.get("content")
        if content:
            return content
    # older/other shapes: check message.content
    msg = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(msg, dict):
        c = msg.get("content")
        if c:
            return c
    # check 'text'
    text = choice.get("text") if isinstance(choice, dict) else None
    if text:
        return text
    return ""

def ask(user_message: str, max_tokens: int = 256, temperature: float = 0.2):
    """
    Non-streaming blocking ask: returns final assistant text.
    If model unavailable, returns a mocked reply.
    """
    prompt_messages = conv.build_messages(user_message)
    if _llm is None:
        mocked = "Mocked assistant reply — model not loaded. Place a .gguf model on your Desktop or repo/model/."
        conv.append(user_message, mocked)
        return mocked

    # Prefer chat completion API
    try:
        resp = _llm.create_chat_completion(messages=prompt_messages,
                                          max_tokens=max_tokens,
                                          temperature=temperature)
        # response likely dict with choices -> message -> content
        choices = resp.get("choices", [])
        if choices:
            first = choices[0]
            # chat style:
            message = first.get("message")
            if isinstance(message, dict) and "content" in message:
                text = message["content"].strip()
            else:
                # fallback older 'text'
                text = first.get("text", "").strip()
        else:
            text = ""
    except AttributeError:
        # fallback older create API
        try:
            resp = _llm.create(prompt="\n".join([m["content"] for m in prompt_messages]),
                                max_tokens=max_tokens,
                                temperature=temperature)
            if isinstance(resp, dict):
                choices = resp.get("choices", [])
                text = choices[0].get("text", "").strip() if choices else ""
            else:
                text = str(resp)
        except Exception as e:
            text = f"[error] model inference failed: {e}"
    except Exception as e:
        text = f"[error] model inference failed: {e}"

    conv.append(user_message, text)
    return text

def ask_stream(user_message: str, max_tokens: int = 256, temperature: float = 0.2):
    """
    Generator yielding incremental assistant text fragments.
    Yields strings. If model missing, yields a mocked reply then stops.
    """
    prompt_messages = conv.build_messages(user_message)
    if _llm is None:
        mocked = "Mocked assistant reply — model not loaded. Place a .gguf model on your Desktop or repo/model/."
        conv.append(user_message, mocked)
        yield mocked
        return

    # Try streaming chat completion
    try:
        stream = _llm.create_chat_completion(messages=prompt_messages,
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            stream=True)
        # stream is an iterator/generator of dict-like events
        partial = ""
        for event in stream:
            # Each event often has 'choices' list; accumulate parts
            choices = event.get("choices", [])
            for ch in choices:
                piece = _extract_from_choice_chunk(ch)
                if piece:
                    partial += piece
                    yield piece
        # finalization: yield nothing more (consumer should handle)
        # append full text to history
        conv.append(user_message, partial)
    except AttributeError:
        # create_chat_completion not supported with stream=True -> fallback to blocking ask
        text = ask(user_message, max_tokens=max_tokens, temperature=temperature)
        yield text
    except Exception as e:
        yield f"[error] streaming failed: {e}"

# History persistence utilities
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(history_list):
    try:
        HISTORY_FILE.write_text(json.dumps(history_list, indent=2), encoding="utf-8")
    except Exception as e:
        print("[server] failed saving history:", e)

def add_history_item(title, messages):
    """
    messages: list of dicts: {'role':'user'/'assistant','content':...}
    """
    data = load_history()
    data.insert(0, {"title": title, "messages": messages})
    # cap to 200 items
    data = data[:200]
    save_history(data)

if __name__ == "__main__":
    # quick local test
    init_model()
    print(ask("Hello, test streaming please."))
    print("Streamed:")
    for p in ask_stream("Say hi and explain yourself briefly."):
        print(p, end="", flush=True)
