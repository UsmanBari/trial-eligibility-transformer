"""Start the browser app:  python -m app  [options]   then open http://127.0.0.1:8000

Defaults load the final fine-tuned model (Qwen2.5-1.5B-Instruct + LoRA v2) from HuggingFace.
On a CPU-only laptop this needs ~7 GB free RAM in float32, or ~3.5 GB with --dtype bfloat16.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="Trial eligibility screener (browser app)")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="usmanbari2005/trialmatch-qwen25-1p5b-lora-v2",
                    help="LoRA adapter (HF id or local folder); 'none' for the zero-shot base model")
    ap.add_argument("--draft-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--draft-adapter", default="usmanbari2005/trialmatch-qwen25-0p5b-draft-lora")
    ap.add_argument("--load-draft", action="store_true", help="load the draft model at start (else on first use)")
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"],
                    help="default: float16 on GPU, float32 on CPU")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open-browser", action="store_true", help="open the page in the default browser")
    a = ap.parse_args()

    import uvicorn
    from app.server import create_app
    from trialmatch.inference import Screener

    screener = Screener(a.base_model, None if a.adapter.lower() == "none" else a.adapter,
                        draft_model=a.draft_model, draft_adapter=a.draft_adapter or None,
                        dtype=a.dtype, load_draft=a.load_draft)
    url = f"http://{a.host}:{a.port}"
    print(f"Open {url}  (the model loads in the background; the page shows progress)")
    if a.open_browser:
        import threading
        import webbrowser
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(screener), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
