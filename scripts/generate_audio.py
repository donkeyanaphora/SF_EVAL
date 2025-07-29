import json, uuid, time, sys, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv

# ── paths & constants ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
JSON_IN = ROOT / "data" / "input"  / "synthetic_sentences.json"
OUT_DIR = ROOT / "data" / "output"
MANIFEST = OUT_DIR / "manifest.jsonl"

MODEL = "tts-1-hd"
VOICE = "nova"
FORMAT = "wav"
JOBS = 6
RETRIES = 3

load_dotenv(ROOT / ".env") 
client = OpenAI()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helper: span validation (unchanged) ─────────────────────────────────────
# def validate_spans(block: list[dict], cat: str) -> None:
#     for idx, row in enumerate(block, 1):
#         txt = row["text"]
#         for term in row["terms"]:
#             if txt[term["start"]:term["end"]] != term["text"]:
#                 raise ValueError(
#                     f"[{cat} #{idx}] span mismatch: "
#                     f'"{term["text"]}" not at {term["start"]}:{term["end"]}'
#                 )
#     print(f"{cat} spans OK")

# ── helper: TTS for a single sentence ───────────────────────────────────────
def synth(category: str, idx: int, sentence: str) -> None:
    uid = str(uuid.uuid4())
    wav = OUT_DIR / f"{uid}.{FORMAT}"
    if wav.exists():
        return  # skip if already done

    for attempt in range(RETRIES + 1):
        try:
            rsp = client.audio.speech.create(
                model=MODEL,
                voice=VOICE,
                input=sentence,
                response_format=FORMAT,
            )
            # new helper (SDK ≥ 1.23)
            rsp.write_to_file(wav)
            print("saved", wav.name)

            # append one manifest line
            with MANIFEST.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "uuid": uid,
                    "file": wav.name,
                    "category": category,
                    "index": idx,
                    "text": sentence
                }) + "\n")
            return

        except (RateLimitError, APIError) as e:
            if attempt < RETRIES:
                wait = 2 ** attempt
                print(f"retry {category} #{idx} in {wait}s – {e}")
                time.sleep(wait)
                continue
            raise

        except Exception:
            print(f"\nFAILED {category} #{idx}\n{sentence}\n", file=sys.stderr)
            traceback.print_exc()
            return

# ── main batch driver (with threads) ────────────────────────────────────────
def main() -> None:
    data = json.loads(JSON_IN.read_text("utf-8"))

    # for cat in ("doctor_sentences", "claim_sentences"):
    #     validate_spans(data[cat], cat)

    tasks = []
    with ThreadPoolExecutor(max_workers=JOBS) as pool:
        for cat in ("radiology_dictations", "study_sentences"):
            for i, row in enumerate(data[cat], 1):
                tasks.append(pool.submit(synth, cat, i, row["text"]))

        for fut in as_completed(tasks):
            if (exc := fut.exception()):
                print("\nBatch stopped because of an unrecoverable error.")
                sys.exit(1)

    print("All sentences processed; audio + manifest at:", OUT_DIR)

# ── entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except ValueError as ve:
        print("\nSPAN VALIDATION FAILED:", ve)
        sys.exit(1)
    except Exception as e:
        print("\nUnexpected error:", e)
        sys.exit(1)