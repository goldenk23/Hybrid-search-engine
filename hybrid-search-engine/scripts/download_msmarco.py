"""
Download the MS MARCO passage ranking dataset.

By default this script fetches only the small qrels files used for evaluation.
Pass ``--include-collection`` to also download and extract the full
``collection.tsv`` file expected by the search pipeline.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data" / "msmarco"
MSMARCO_BASE = "https://msmarco.z22.web.core.windows.net/msmarcoranking"
COLLECTION_ARCHIVE = "collectionandqueries.tar.gz"
COLLECTION_URL = f"{MSMARCO_BASE}/{COLLECTION_ARCHIVE}"
COLLECTION_FILENAME = "collection.tsv"

DOWNLOADS = {
    "qrels.train.tsv": f"{MSMARCO_BASE}/qrels.train.tsv",
    "qrels.dev.small.tsv": f"{MSMARCO_BASE}/qrels.dev.small.tsv",
}


class DownloadProgressBar(tqdm):
    """Show a progress bar while downloading."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_file(url: str, output_path: Path) -> bool:
    """Download a file with progress output and basic error handling."""
    if output_path.exists():
        print(f"  OK Already exists: {output_path.name}")
        return True

    try:
        print(f"  -> Downloading: {output_path.name} from {url}")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as progress:
            urllib.request.urlretrieve(url, filename=str(output_path), reporthook=progress.update_to)
        print(f"  OK Downloaded: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  FAIL Failed to download {output_path.name}: {exc}")
        return False


def extract_collection(archive_path: Path, output_dir: Path) -> bool:
    """Extract collection.tsv from the official MS MARCO archive."""
    collection_path = output_dir / COLLECTION_FILENAME
    if collection_path.exists():
        print(f"  OK Already extracted: {COLLECTION_FILENAME}")
        return True

    try:
        print(f"  -> Extracting: {COLLECTION_FILENAME}")
        with tarfile.open(archive_path, "r:gz") as archive:
            member = archive.getmember(COLLECTION_FILENAME)
            archive.extract(member, path=output_dir)
        print(
            f"  OK Extracted: {COLLECTION_FILENAME} "
            f"({collection_path.stat().st_size / 1024 / 1024:.1f} MB)"
        )
        return True
    except KeyError:
        print(f"  FAIL Archive does not contain {COLLECTION_FILENAME}")
        return False
    except Exception as exc:
        print(f"  FAIL Failed to extract {COLLECTION_FILENAME}: {exc}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download MS MARCO dataset files.")
    parser.add_argument(
        "--include-collection",
        action="store_true",
        help="Download and extract the full collection.tsv archive.",
    )
    return parser.parse_args()


def main() -> None:
    """Download and prepare MS MARCO dataset files."""
    try:
        args = parse_args()
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("MS MARCO Passage Ranking Dataset - Downloader")
        print("=" * 70)
        print(f"\nTarget directory: {DATA_DIR}")
        if args.include_collection:
            print("This will download the qrels files and the full collection archive.\n")
        else:
            print("This will download only the small qrels files.\n")

        success_count = 0
        failed_files: list[str] = []

        print("Downloading relevance judgments (qrels)...\n")
        for filename, url in DOWNLOADS.items():
            output_path = DATA_DIR / filename
            if download_file(url, output_path):
                success_count += 1
            else:
                failed_files.append(filename)

        if args.include_collection:
            print("\nDownloading full passages collection...\n")
            archive_path = DATA_DIR / COLLECTION_ARCHIVE
            if download_file(COLLECTION_URL, archive_path):
                success_count += 1
                if extract_collection(archive_path, DATA_DIR):
                    success_count += 1
                else:
                    failed_files.append(COLLECTION_FILENAME)
            else:
                failed_files.append(COLLECTION_ARCHIVE)

        print("\n" + "=" * 70)
        total_expected = len(DOWNLOADS) + (2 if args.include_collection else 0)
        if failed_files:
            print("Download completed with issues:")
            print(f"  OK Successful: {success_count}/{total_expected}")
            print(f"  FAIL Failed: {', '.join(failed_files)}")
            print("\nTroubleshooting:")
            print("  1. Check your internet connection")
            print("  2. Try again in a few minutes")
            print("  3. Manual download: https://github.com/microsoft/MSMARCO-Passage-Ranking")
        else:
            print("OK All requested files downloaded successfully.")

        print(f"\nFiles saved to: {DATA_DIR}")
        if args.include_collection:
            print(f"Collection path: {DATA_DIR / COLLECTION_FILENAME}")
        print("=" * 70)

    except Exception as exc:
        print(f"\nFAIL Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()