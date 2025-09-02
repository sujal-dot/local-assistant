# Local Assistant (Mistral 7B) — Starter Repo
This repository is a scaffold to build an offline ChatGPT-style assistant using Mistral-7B.
It includes:
- backend/: model wrapper, memory store, speech hooks
- frontend/: PyQt6 GUI skeleton with styles + placeholder assets
- packaging/: notes for building installers
- A minimal working flow (requires you to supply the model file, e.g. mistral-7b.gguf)

**Important**
- Download or place your model under `model/` (not included here).
- This scaffold uses `llama-cpp-python` style APIs as an example; adjust to your inference binding.
 
 **To Run**
 cd /Users/sujal/Desktop/local-assistant
python -m frontend.gui