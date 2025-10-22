import zipfile
from pathlib import Path


PAGES_DIR = Path("pages")
INGESTION_DIR = Path("ingestion_program")
SCORING_DIR = Path("scoring_program")
PHASE_DATA = Path("dev_phase")

BUNDLE_FILES = [
    "competition.yaml",
    "logo.png",
    "solution/submission.py",
]


if __name__ == "__main__":

    with zipfile.ZipFile("bundle.zip", mode='w') as bundle:

        for f in BUNDLE_FILES:
            print(f)
            bundle.write(f)
        for dirpath in [INGESTION_DIR, SCORING_DIR, PAGES_DIR, PHASE_DATA]:
            for f in dirpath.rglob("*"):
                if not f.is_file():
                    continue
                if f.name.startswith('.') or f.name.endswith('.pyc'):
                    continue
                print(f)
                bundle.write(f)
