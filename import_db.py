import os
import zipfile
import requests
from pathlib import Path
from music_similarity_engine import MusicDatabase
import time

FMA_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
DATA_DIR = Path("data/fma")
AUDIO_DIR = DATA_DIR / "fma_small"
ZIP_PATH = DATA_DIR / "fma_small.zip"

def download_fma_small():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if ZIP_PATH.exists():
        print("✅ Zip file already exists.")
        return

    print(f"⬇️  Downloading FMA Small dataset (~1.6GB)...")
    with requests.get(FMA_URL, stream=True) as r:
        r.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("✅ Download complete.")

def unzip_dataset():
    if AUDIO_DIR.exists():
        print("✅ Audio folder already exists.")
        return
    
    print("📦 Unzipping dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("✅ Unzipping complete.")

def add_songs_to_db():
    print("🎵 Adding songs to MusicDatabase...")
    db = MusicDatabase()

    mp3_files = sorted(AUDIO_DIR.glob("**/*.mp3"))
    print(f"🎧 Found {len(mp3_files)} tracks.")

    added = 0
    for mp3_file in mp3_files[:500]:  # change this limit as needed
        song_id = mp3_file.stem
        metadata = {
            "filename": mp3_file.name,
            "source": "FMA Small"
        }

        success = db.add_song(song_id, str(mp3_file), metadata)
        if success:
            print(f"✅ Added: {song_id}")
            added += 1
        else:
            print(f"⚠️  Skipped: {song_id}")

        time.sleep(0.05)  # optional small delay

    print(f"📊 Finished. Total songs added: {added}")

def main():
    print("🚀 Starting FMA import process...")
    download_fma_small()
    unzip_dataset()
    add_songs_to_db()

if __name__ == "__main__":
    main()
