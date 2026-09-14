"""Fetching the deposited inputs, and the one switch that regenerates them.

Two kinds of input sit behind this module, and the notebook treats them the
same way: look for what is already on disk, fetch it if it is missing, and
regenerate it only when told to.

- The natural binder reference archive holds the frozen RCSB search results and
  the gzipped structures. The notebook still runs the analysis over them, so
  the measurement code is exercised on every fresh clone rather than only when
  someone goes looking for it. Freezing the search results is also what makes
  the set stable: the RCSB recomputes its sequence-identity clusters as the PDB
  grows, so a live query drifts while a recorded one does not.
- The design campaign archive holds the per-design tables. Those cannot be
  regenerated cheaply, because designing needs GPUs and the loop is not
  seed-reproducible, so a rerun produces different designs rather than the same
  ones.

Downloads try S3 first and fall back to Zenodo, because Zenodo can be slow.
Neither needs an account: the S3 reads are unsigned.

Following the pattern in `Arcadia-Science/2025-phylogenetic-analysis-of-msa-pairformer`,
the downloaded archive is kept after extraction rather than deleted. It is the
marker that says this data was fetched rather than computed, and because it is
written only after the extraction succeeds, an interrupted download leaves no
marker and the next run fetches again.

``FORCE_RERUN=1`` is the other path. It deletes everything derivable, skips the
fetch, and regenerates from source: the RCSB query goes to the live API and the
campaigns go to Modal. That needs a Modal account and costs real GPU time, so
:func:`require_modal` runs before anything is deleted.
"""

from __future__ import annotations
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from analysis import paths

PUBLIC_S3_BUCKET = "2026-esmfold2-secondary-structure-steering"
ZENODO_RECORD = "PENDING"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

FORCE_RERUN_ENV = "FORCE_RERUN"

REFERENCE_ARCHIVE = "natural-binder-inputs.zip"
DESIGN_ARCHIVE = "design-campaigns.zip"


def force_rerun() -> bool:
    """Whether the caller asked for everything to be regenerated from source."""
    return os.environ.get(FORCE_RERUN_ENV, "") == "1"


def _download_from_s3(key: str, destination: Path) -> bool:
    """Unsigned public read. False if the object is not there or S3 is down."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    try:
        client.download_file(PUBLIC_S3_BUCKET, key, str(destination))
    except (ClientError, BotoCoreError):
        return False
    return True


def _download_from_zenodo(key: str, destination: Path) -> None:
    url = f"{ZENODO_BASE_URL}/{key}?download=1"
    with urllib.request.urlopen(url, timeout=600) as response:
        destination.write_bytes(response.read())


@dataclass(frozen=True)
class DataArtifact:
    """One deposited archive, fetched on demand and kept as its own marker."""

    filename: str
    extract_dir: Path
    description: str

    @property
    def archive_path(self) -> Path:
        return self.extract_dir / self.filename

    def is_fetched(self) -> bool:
        return self.archive_path.is_file()

    def fetch(self) -> Path:
        """Download and extract, unless the archive is already here.

        The archive is written to its final name only after extraction
        succeeds, so a half-finished download never looks like a complete one.
        """
        if self.is_fetched():
            return self.archive_path
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        partial = self.archive_path.with_suffix(".partial")
        print(f"fetching {self.description} ({self.filename})...", flush=True)
        if not _download_from_s3(self.filename, partial):
            print("  S3 unavailable, falling back to Zenodo", flush=True)
            try:
                _download_from_zenodo(self.filename, partial)
            except Exception as error:  # noqa: BLE001
                partial.unlink(missing_ok=True)
                raise FileNotFoundError(
                    f"could not fetch {self.filename} from S3 bucket "
                    f"{PUBLIC_S3_BUCKET} or Zenodo record {ZENODO_RECORD}.\n"
                    f"  cause: {error}\n"
                    f"If the deposit is not minted yet, the archive has to be "
                    f"placed at {self.archive_path} by hand, or regenerated "
                    f"with {FORCE_RERUN_ENV}=1 and a Modal account."
                ) from error
        with zipfile.ZipFile(partial) as archive:
            archive.extractall(self.extract_dir)
        partial.rename(self.archive_path)
        print(f"  extracted into {self.extract_dir}", flush=True)
        return self.archive_path


    def package(self, members: list[Path]) -> Path:
        """Zip ``members`` into the archive, for uploading to the deposit.

        Paths are stored relative to ``extract_dir``, so extracting the result
        reproduces the layout the notebook expects. Already-compressed members
        are stored rather than deflated, which is most of what goes in here.
        """
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            self.archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for member in members:
                compression = (
                    zipfile.ZIP_STORED
                    if member.suffix in (".gz", ".zip", ".parquet")
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(
                    member,
                    arcname=str(member.relative_to(self.extract_dir)),
                    compress_type=compression,
                )
        return self.archive_path


REFERENCE_INPUTS = DataArtifact(
    filename=REFERENCE_ARCHIVE,
    extract_dir=paths.REFERENCE,
    description="the frozen RCSB search results and structures",
)
DESIGN_TABLES = DataArtifact(
    filename=DESIGN_ARCHIVE,
    extract_dir=paths.CAMPAIGNS,
    description="the per-design campaign tables",
)


def reference_members() -> list[Path]:
    """What the reference archive holds: the frozen search results and structures.

    Deliberately not the composition table. The table is what the notebook
    derives from these, and depositing it too would let a reader load a result
    without ever running the measurement that produced it.
    """
    from analysis import rcsb

    members = [paths.reference_table(rcsb.SEARCH_HITS)]
    members.extend(sorted(rcsb.CIF_CACHE.glob("*.cif.gz")))
    return [member for member in members if member.is_file()]


def design_members() -> list[Path]:
    """What the design archive holds: the per-campaign tables.

    Not the predicted structures. They are 82 MB across the grid and nothing
    in the notebook reads them, because composition is measured once at
    collection time and stored beside the campaign.
    """
    members: list[Path] = []
    for campaign_dir in sorted(paths.CAMPAIGNS.glob("*__*")):
        for name in (
            "composition.parquet",
            "all_results.parquet",
            "step_trace.parquet",
            "protocol.json",
            "summary.json",
        ):
            member = campaign_dir / name
            if member.is_file():
                members.append(member)
    return members


def require_modal() -> None:
    """Raise unless Modal could actually regenerate the designs.

    Called before anything is deleted. Deleting first and discovering the
    credentials are missing afterwards would leave no data and no way to make
    it, which is the one genuinely unrecoverable ordering here.
    """
    try:
        import modal.config
    except ImportError as error:
        raise RuntimeError(
            f"{FORCE_RERUN_ENV}=1 regenerates the design campaigns on Modal, "
            f"which is not installed. Install it with\n"
            f"    pip install -r requirements-campaign.txt\n"
            f"or unset {FORCE_RERUN_ENV} to use the deposited data."
        ) from error

    config = modal.config.config
    if not (config.get("token_id") and config.get("token_secret")):
        raise RuntimeError(
            f"{FORCE_RERUN_ENV}=1 regenerates the design campaigns on Modal, "
            f"but no Modal token is configured. Run\n"
            f"    modal token new\n"
            f"then deploy the app with\n"
            f"    modal deploy src/analysis/modal_app.py\n"
            f"or unset {FORCE_RERUN_ENV} to use the deposited data."
        )


def clear_generated() -> list[Path]:
    """Delete everything the notebook can regenerate, and say what went.

    Leaves the archives alone if they happen to be present; ``fetch`` is
    skipped under ``FORCE_RERUN`` anyway, and deleting a download nobody asked
    to replace would be gratuitous.
    """
    removed: list[Path] = []
    for path in sorted(paths.REFERENCE.glob("*")):
        if path.name in (".gitkeep", REFERENCE_ARCHIVE):
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    for path in sorted(paths.CAMPAIGNS.glob("*")):
        if path.name in (".gitkeep", DESIGN_ARCHIVE):
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    return removed
