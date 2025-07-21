import json, sys, traceback, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

ROOT = Path(__file__).resolve().parents[1]  
print(ROOT)
JSON_IN = ROOT / "data" / "input"  / "synthetic_sentences.json"
OUT_DIR = ROOT / "data" / "output"

MODEL = "tts-1-hd"
VOICE = "nova"
FORMAT = "wav"
JOBS = 6
RETRY = 3

load_dotenv(ROOT / ".env") 
client = OpenAI()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_spans(block: list[dict], cat: str) -> None:
    """Raise ValueError if any term span is incorrect."""
    for idx, row in enumerate(block, 1):
        txt = row["text"]
        for term in row["terms"]:
            if txt[term["start"] : term["end"]] != term["text"]:
                raise ValueError(
                    f"[{cat} #{idx}] span mismatch: "
                    f'"{term["text"]}" not at {term["start"]}:{term["end"]}'
                )
    print(f"{cat} spans OK")


def synth(category, idx, sentence):
    out_file = OUT_DIR / f"{category[:3]}_{idx:02}.{FORMAT}"
    if out_file.exists():
        return

    for attempt in range(RETRY + 1):
        try:
            resp = client.audio.speech.create(
                model=MODEL,
                voice=VOICE,
                input=sentence,
                response_format=FORMAT  # works with SDK ≤ 1.9x
            )
            resp.stream_to_file(str(out_file))
            print(f"saved {out_file.name}")
            return

        except (RateLimitError, APIError) as e:
            if attempt < RETRY:
                wait = 2 ** attempt
                print(f"retry {category} #{idx} in {wait}s – {e}")
                time.sleep(wait)
                continue
            raise

        except Exception:
            # any other exception: log sentence + traceback, then skip
            print(f"\nFAILED {category} #{idx}\n{sentence}\n", file=sys.stderr)
            traceback.print_exc()
            return

def main():
    data = json.loads(JSON_IN.read_text("utf-8"))

    # VALIDATE ALL SPANS FIRST
    for cat in ("doctor_sentences", "claim_sentences"):
        validate_spans(data[cat], cat)

    # ONLY IF VALIDATION PASSES, DO TTS
    tasks = []
    with ThreadPoolExecutor(max_workers=JOBS) as pool:
        for cat in ("doctor_sentences", "claim_sentences"):
            for i, row in enumerate(data[cat], 1):
                tasks.append(pool.submit(synth, cat, i, row["text"]))

        # surface the first exception, if any
        for fut in as_completed(tasks):
            exc = fut.exception()
            if exc:
                print("\nBatch stopped because of an unrecoverable error.")
                sys.exit(1)

    print("All sentences processed; audio files are in", OUT_DIR)

if __name__ == "__main__":
    try:
        main()
    except ValueError as ve:
        print("\nSPAN VALIDATION FAILED:", ve)
        sys.exit(1)
    except Exception as e:
        print("\nUnexpected error:", e)
        sys.exit(1)