# ingest.py
import os, glob, shutil, pickle, hashlib, json, time, pandas as pd
import gdown
from sentence_transformers import SentenceTransformer

FOLDER_ID = "1l9HIUYCAKmK9P3KW_Sh90s72FsAThFBh"
TEMP_DIR = "_temp_drive_pull"
STORE_PATH = "store.pkl"
MANIFEST_PATH = "ingestion_manifest.json"
REQUIRED_COLS = {"topic", "question", "answer", "source_url"}
MAX_RETRIES = 3

def pull_folder_tree(folder_id, temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            try:
                gdown.download_folder(url, output=temp_dir, quiet=False, use_cookies=False, remaining_ok=True)
            except TypeError:
                # installed gdown version doesn't support remaining_ok — fall back
                gdown.download_folder(url, output=temp_dir, quiet=False, use_cookies=False)
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Drive pull failed after {MAX_RETRIES} attempts: {e}")
            print(f"Pull attempt {attempt} failed: {e}. Retrying...")
            time.sleep(2 ** attempt)

def load_all_csvs(temp_dir):
    files = glob.glob(os.path.join(temp_dir, "**", "*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError("No CSVs found anywhere in the folder tree.")

    frames, bad_files = [], []
    for f in files:
        try:
            df = pd.read_csv(f)
            missing = REQUIRED_COLS - set(df.columns)
            if missing:
                bad_files.append((f, f"missing columns: {missing}"))
                continue
            df["_source_file"] = os.path.relpath(f, temp_dir)
            frames.append(df)
        except Exception as e:
            bad_files.append((f, str(e)))

    if bad_files:
        print("REJECTED FILES (fix and re-run):")
        for f, reason in bad_files:
            print(f"  - {f}: {reason}")

    if not frames:
        raise ValueError("Every CSV failed validation. Nothing to ingest.")

    combined = pd.concat(frames, ignore_index=True)
    return combined.dropna(subset=["question", "answer"]), len(files)

def check_against_manifest(current_count, current_files):
    if not os.path.exists(MANIFEST_PATH):
        return  # first run — nothing to compare against
    with open(MANIFEST_PATH) as f:
        last = json.load(f)
    last_count = last.get("file_count", 0)
    if current_count < last_count:
        missing = set(last.get("files", [])) - set(current_files)
        raise RuntimeError(
            f"REGRESSION: last successful pull had {last_count} files, this one found {current_count}. "
            f"Missing: {sorted(missing)[:10]}. Aborting before store.pkl is touched. "
            f"If files were genuinely deleted on purpose, delete {MANIFEST_PATH} and re-run."
        )

def write_manifest(count, files):
    with open(MANIFEST_PATH, "w") as f:
        json.dump({"file_count": count, "files": sorted(files)}, f, indent=2)

def fact_hash(row):
    return hashlib.md5(f"{row['topic']}|{row['question']}|{row['answer']}".encode()).hexdigest()

def rebuild_store(folder_id=FOLDER_ID, store_path=STORE_PATH, model_name="all-MiniLM-L6-v2"):
    pull_folder_tree(folder_id, TEMP_DIR)
    df, total_files = load_all_csvs(TEMP_DIR)
    current_files = [os.path.relpath(f, TEMP_DIR) for f in glob.glob(os.path.join(TEMP_DIR, "**", "*.csv"), recursive=True)]

    check_against_manifest(total_files, current_files)  # abort here if truncation detected

    df["_hash"] = df.apply(fact_hash, axis=1)
    before = len(df)
    df = df.drop_duplicates(subset="_hash", keep="last")
    if before != len(df):
        print(f"Dropped {before - len(df)} duplicate facts.")

    existing = {}
    if os.path.exists(store_path):
        with open(store_path, "rb") as f:
            old = pickle.load(f)
        existing = {fact["_hash"]: emb for fact, emb in zip(old["facts"], old["embeddings"])}

    model = SentenceTransformer(model_name)
    new_hashes = [h for h in df["_hash"] if h not in existing]
    if new_hashes:
        new_rows = df[df["_hash"].isin(new_hashes)]
        new_embs = model.encode(new_rows["question"].tolist(), show_progress_bar=True)
        existing.update(dict(zip(new_rows["_hash"], new_embs)))

    embeddings = [existing[h] for h in df["_hash"]]
    with open(store_path, "wb") as f:
        pickle.dump({"facts": df.to_dict(orient="records"), "embeddings": embeddings}, f)

    write_manifest(total_files, current_files)  # only commit on success
    shutil.rmtree(TEMP_DIR)
    print(f"Store rebuilt: {len(df)} facts from {total_files} files | {len(new_hashes)} new | {len(df)-len(new_hashes)} reused.")

if __name__ == "__main__":
    rebuild_store()