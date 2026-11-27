"""Inline transcripts into the viewer so it works as a single file.

    python scripts/build_viewer.py viz/data/*.json
writes viz/dist/replay.html (standalone page) and viz/dist/replay_body.html
(body-only variant used for publishing as a claude.ai Artifact).
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(path: str) -> dict:
    """UTF-8, or the Windows code page for transcripts written by older versions on Windows."""
    raw = open(path, "rb").read()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252", errors="replace"))


def slim(t: dict) -> dict:
    t = dict(t)
    t.pop("action_log", None)      # not needed for replay; keeps the file small
    return t


if __name__ == "__main__":
    files = [f for f in sys.argv[1:] if not f.endswith("_llm.json")]   # skip raw model replies
    if not files:
        d = os.path.join(ROOT, "viz", "data")
        idx = json.load(open(os.path.join(d, "index.json"), encoding="utf-8"))
        files = [os.path.join(d, it["file"]) for it in idx]
    data = [slim(load(f)) for f in files]
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    src = open(os.path.join(ROOT, "viz", "index.html"), encoding="utf-8").read()
    body = src.replace("/*__EMBED__*/[]", blob)
    out = os.path.join(ROOT, "viz", "dist")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "replay_body.html"), "w", encoding="utf-8").write(body)
    page = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<style>[hidden]{display:none!important}body{margin:0}</style></head><body>\n'
            + body + "\n</body></html>\n")
    open(os.path.join(out, "replay.html"), "w", encoding="utf-8").write(page)
    print(f"embedded {len(data)} transcripts, {len(page) / 1e6:.2f} MB")
