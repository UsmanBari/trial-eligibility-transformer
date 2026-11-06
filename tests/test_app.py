"""Browser-app backend tests: input parsing, API contract, decoders agree, error handling.

A tiny random Qwen2 model and a word-level tokenizer with a chat template are built in a
temp folder, so this runs on CPU in seconds with no download.
"""
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from trialmatch.inference import InputError, Screener, parse_criteria, parse_patient_note  # noqa: E402


def _tiny_model_dir(path: Path, layers: int, seed: int, words: set):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    specials = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]
    vocab = {w: i for i, w in enumerate(specials + sorted(words))}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tk.decoder = decoders.WordPiece(prefix="##")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", unk_token="<unk>",
                                  eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"])
    tok.chat_template = ("{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
                         "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}")
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      eos_token_id=vocab["<|im_end|>"], pad_token_id=0, tie_word_embeddings=False)
    m = Qwen2ForCausalLM(cfg)
    m.generation_config.eos_token_id = vocab["<|im_end|>"]
    m.save_pretrained(path)
    tok.save_pretrained(path)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient
    import trialmatch.prompting as P
    from app.server import DEMO, create_app
    d = tmp_path_factory.mktemp("tiny")
    text = " ".join([P.SYSTEM_PROMPT, P.LABEL_GUIDE["inclusion"], P.LABEL_GUIDE["exclusion"],
                     P.user_message([{"id": 0, "text": "x"}], "inclusion", "y"),
                     P.user_message([{"id": 0, "text": "x"}], "exclusion", "y"),
                     DEMO["note"], DEMO["criteria"], "system user assistant 0 1 2 3 4 5 6 7 8 9"])
    words = set(re.findall(r"\w+|[^\w\s]", text))
    _tiny_model_dir(d / "target", 2, 0, words)
    _tiny_model_dir(d / "draft", 1, 1, words)
    s = Screener(str(d / "target"), None, draft_model=str(d / "draft"), dtype="float32", max_new_tokens=24)
    s.load()
    assert s.status == "ready", s.error
    app = create_app(s, results_dir=ROOT / "results", processed_dir=d / "none", load_in_background=False)
    c = TestClient(app)
    c.screener = s
    return c


# ------------------------------------------------------------------ parsing
def test_parse_free_text_note():
    s = parse_patient_note("A 62-year-old man. He smokes 20 cigarettes a day! BP 148/92 mmHg.\nNo heart failure.")
    assert [x["id"] for x in s] == [0, 1, 2, 3] and s[1]["text"] == "He smokes 20 cigarettes a day!"


def test_parse_numbered_note_keeps_numbering():
    s = parse_patient_note("0. A 58-year-old woman.\n1. She has 2. children.\n2. No diabetes.")
    assert [x["text"] for x in s] == ["A 58-year-old woman.", "She has 2. children.", "No diabetes."]


def test_parse_criteria_headings_and_bullets():
    c = parse_criteria("Inclusion Criteria:\n- Age 18-65\n2) Signed consent\nExclusion criteria\n"
                       "* Pregnancy\nExclusion of patients with dementia\nInclusion: BMI > 30")
    assert [(x["criterion_type"], x["criterion_text"]) for x in c] == [
        ("inclusion", "Age 18-65"), ("inclusion", "Signed consent"), ("exclusion", "Pregnancy"),
        ("exclusion", "Exclusion of patients with dementia"), ("inclusion", "BMI > 30")]


@pytest.mark.parametrize("text", ["", "- Age 18-65", "Inclusion Criteria:\n\n"])
def test_parse_criteria_errors(text):
    with pytest.raises(InputError):
        parse_criteria(text)


def test_parse_note_errors():
    with pytest.raises(InputError):
        parse_patient_note("   ")
    with pytest.raises(InputError):
        parse_patient_note("x" * 30_000)


# ------------------------------------------------------------------ API
def test_health_and_examples(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ready" and h["model"]["n_layers"] == 2 and "kv" in h["decoders"]
    ex = client.get("/api/examples").json()["examples"]
    assert ex[0]["id"] == "demo" and "Inclusion Criteria" in ex[0]["criteria"]


def test_screen_demo(client):
    from app.server import DEMO
    r = client.post("/api/screen", json={"note": DEMO["note"], "criteria_text": DEMO["criteria"]})
    assert r.status_code == 200, r.text
    res = r.json()
    assert len(res["results"]) == 6 and len(res["sentences"]) == 6
    allowed = {"inclusion": {"included", "not included", "not enough information", "not applicable", "invalid"},
               "exclusion": {"excluded", "not excluded", "not enough information", "not applicable", "invalid"}}
    for x in res["results"]:
        assert x["label"] in allowed[x["criterion_type"]]
        assert all(0 <= e < 6 for e in x["evidence"])
    assert res["overall"]["status"] in {"likely_eligible", "likely_ineligible", "needs_review"}
    assert res["timing"]["shared_prefix_tokens"] > 20          # note prefix encoded once


def test_all_decoders_give_identical_output(client):
    from app.server import DEMO
    outs = {}
    for dec in ("kv", "nocache", "spec_lookup", "spec_draft"):
        r = client.post("/api/screen", json={"note": DEMO["note"], "criteria_text": DEMO["criteria"],
                                             "decoder": dec, "k": 3})
        assert r.status_code == 200, r.text
        outs[dec] = [x["raw_output"] for x in r.json()["results"]]
    assert outs["kv"] == outs["nocache"] == outs["spec_lookup"] == outs["spec_draft"]


def test_compare_endpoint(client):
    from app.server import DEMO
    r = client.post("/api/compare", json={"note": DEMO["note"], "criterion_type": "inclusion",
                                          "criterion_text": "Age 40 to 75 years", "include_draft": True})
    rows = r.json()["rows"]
    assert [x["decoder"] for x in rows] == ["nocache", "kv", "spec_lookup", "spec_draft"]
    assert all(x["same_output_as_first"] for x in rows)


def test_structured_criteria_and_export(client):
    r = client.post("/api/screen", json={"note": "A 30-year-old woman. She is pregnant.",
                                         "criteria": [{"type": "exclusion", "text": "Pregnancy"}]})
    res = r.json()
    md = client.post("/api/export", json={"result": res, "format": "markdown"})
    assert md.status_code == 200 and "| 1 | exclusion | Pregnancy |" in md.text and "research prototype" in md.text
    html = client.post("/api/export", json={"result": res, "format": "html"})
    assert html.status_code == 200 and html.text.startswith("<!doctype html>") and "Pregnancy" in html.text
    assert client.post("/api/export", json={"result": {"x": 1}}).status_code == 400


@pytest.mark.parametrize("body,code", [
    ({"note": "", "criteria_text": "Inclusion Criteria:\n- Age > 18"}, 400),
    ({"note": "A man.", "criteria_text": "Age > 18"}, 400),
    ({"note": "A man.", "criteria_text": "Inclusion Criteria:\n- Age > 18", "decoder": "magic"}, 400),
    ({"criteria_text": "Inclusion Criteria:\n- Age > 18"}, 422),
])
def test_input_errors_are_reported(client, body, code):
    r = client.post("/api/screen", json=body)
    assert r.status_code == code
    assert ("error" in r.json()) or ("detail" in r.json())


def test_not_ready_returns_503(client):
    s = client.screener
    s.status = "loading"
    try:
        r = client.post("/api/screen", json={"note": "A man.", "criteria_text": "Inclusion Criteria:\n- Male"})
        assert r.status_code == 503 and "loading" in r.json()["detail"]
    finally:
        s.status = "ready"


def test_benchmarks_and_ui(client):
    b = client.get("/api/benchmarks").json()
    assert set(b) >= {"evaluation", "kv", "attention", "speculative", "figures"}
    if (ROOT / "results" / "ft_v2_test_greedy").exists():
        assert any(r["tag"] == "ft_v2_test_greedy" for r in b["evaluation"])
    page = client.get("/")
    assert page.status_code == 200 and "TrialMatch" in page.text
    assert client.get("/static/app.js").status_code == 200
