# SAM3 Playground

Fun with SAM3

## Setup

### Prerequisites
1. Create and activate a virtual environment with uv:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. Export your Hugging Face token that has access to SAM3 model files:
   ```bash
   export HF_TOKEN=your_token_here
   ```

3. Install transformers from git:
   ```bash
   uv pip install git+https://github.com/huggingface/transformers.git
   ```

4. Run `uv sync`

5. Install project dependencies (editable):
   ```bash
   uv pip install -e .
   ```

## Quickstart
- Start UI: `uvicorn sam3.ui:app --reload --app-dir src`


## UI info

Most of it was created by gemini 3