# AI Agent Arena — 10 Challenge Streamlit POC

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## LLM behavior

The app first attempts to load `Qwen/Qwen2.5-1.5B-Instruct`.

If Hugging Face/model access is unavailable, the app automatically falls back
to an offline TF-IDF/extractive answer generator. Therefore the POC can still
be demonstrated without a Hugging Face login or model download.

## Security

This POC executes submitted Python code with `exec()`. This is suitable only
for a controlled local/demo environment. A public production platform must
use an isolated Docker/Kubernetes/microVM sandbox.
