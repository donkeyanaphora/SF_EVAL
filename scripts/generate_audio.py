# synthesize_medical40_chunks_simple.py
#
# Reads synthetic_sentences.json  ->  writes WAVs + manifest.jsonl
# Make sure OPENAI_API_KEY is in your environment (or .env).

import json, uuid, time, re, sys, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv

# ── PATHS & CONSTANTS ──────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]        # project root
JSON_IN   = ROOT / "data" / "input"  / "specialize-2.json"
OUT_DIR   = ROOT / "data" / "output"
MANIFEST  = OUT_DIR / "manifest.jsonl"

MODEL     = "gpt-4o-mini-tts"
VOICE     = "nova"
FORMAT    = "wav"
JOBS      = 6                       # parallel threads
RETRIES   = 3
INSTRUCT  = "You are a doctor recording a medical report please ignore punctuation and just read fluidly while being sure to pronounce specialized terms correctly."

# ── setup ──────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv()                       # pull OPENAI_API_KEY
client = OpenAI()

_strip_nan = re.compile(r"\s*nan\s*$", re.IGNORECASE)
clean = lambda txt: _strip_nan.sub("", txt).strip()

def tts_job(rec: dict, idx: int):
    uid = str(uuid.uuid4())
    wav_path = OUT_DIR / f"{uid}.{FORMAT}"
    if wav_path.exists():                    # resume-safe
        return

    text = clean(rec["text"])
    for attempt in range(RETRIES + 1):
        try:
            rsp = client.audio.speech.create(
                model=MODEL,
                voice=VOICE,
                input=text,
                instructions=INSTRUCT,
                response_format=FORMAT,
            )
            rsp.write_to_file(wav_path)
            print("✓", wav_path.name)

            row = {
                "uuid": uid,
                "file": wav_path.name,
                "chunk_id": rec.get("chunk_id"),
                "orig_id":  rec.get("orig_id"),
                "label":    rec.get("label"),
                "text":     text,
            }
            with MANIFEST.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            return

        except (RateLimitError, APIError) as e:
            if attempt < RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception:
            print(f"\nFAILED idx={idx}\n{text}\n", file=sys.stderr)
            traceback.print_exc()
            return

def main():
    records = json.loads(JSON_IN.read_text("utf-8"))
    with ThreadPoolExecutor(max_workers=JOBS) as pool:
        futs = [pool.submit(tts_job, rec, i)
                for i, rec in enumerate(records, 1)]
        for f in as_completed(futs):
            if exc := f.exception():
                print("unrecoverable error:", exc, file=sys.stderr)
                sys.exit(1)
    print("\nAll done – audio files and manifest at", OUT_DIR.resolve())

if __name__ == "__main__":
    main()