import os
import gdown
import zipfile

FILE_ID = "1CJFqqyYQ8eDTUJJrd7qPx5-8Jl66lVcw"
OUTPUT = "data/oulad.zip"
DATA_DIR = "data/oulad"

os.makedirs(DATA_DIR, exist_ok=True)

url = f"https://drive.google.com/uc?id={FILE_ID}"

print("Downloading dataset from Google Drive...")
gdown.download(url, OUTPUT, quiet=False)

print("Extracting dataset...")
with zipfile.ZipFile(OUTPUT, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)

os.remove(OUTPUT)

print("Done. Dataset ready.")