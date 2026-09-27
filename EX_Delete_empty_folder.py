from pathlib import Path

ROOT_DIR = Path(r"GSV_Downloader/images")

folders = sorted(
    [p for p in ROOT_DIR.rglob("*") if p.is_dir()],
    key=lambda p: len(p.parts),
    reverse=True
)

for folder in folders:
    if not any(folder.iterdir()):
        print(f"Deleting empty folder: {folder}")
        folder.rmdir()

print("Done.")