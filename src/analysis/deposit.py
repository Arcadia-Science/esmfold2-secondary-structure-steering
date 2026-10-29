"""Package, fetch, and extract preserved reference and campaign inputs."""

from __future__ import annotations
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from analysis import paths

PUBLIC_S3_BUCKET = "2026-esmfold2-secondary-structure-steering"
ZENODO_RECORD = "23542792"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

REFERENCE_ARCHIVE = "natural-binder-inputs.zip"
DESIGN_ARCHIVE = "design-campaigns.zip"


def _download_from_s3(key: str, destination: Path) -> bool:
    """Download a public S3 object; return False on handled download errors."""
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
        """Extract the archive, downloading it first if absent.

        Extraction can overwrite local members. A download receives its final archive
        name only after successful extraction.
        """
        if self.is_fetched():
            with zipfile.ZipFile(self.archive_path) as archive:
                archive.extractall(self.extract_dir)
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
                    f"Download {ZENODO_BASE_URL}/{self.filename}?download=1 "
                    f"manually to {self.archive_path}."
                ) from error
        with zipfile.ZipFile(partial) as archive:
            archive.extractall(self.extract_dir)
        partial.rename(self.archive_path)
        print(f"  extracted into {self.extract_dir}", flush=True)
        return self.archive_path

    def package(self, members: list[Path]) -> Path:
        """Archive members relative to ``extract_dir`` for later extraction.

        Store GZIP, ZIP, and Parquet files without further compression.
        """
        missing = [p for p in members if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing archive inputs: {missing[:5]}")
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        partial = self.archive_path.with_suffix(".partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
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
        partial.replace(self.archive_path)
        return self.archive_path


REFERENCE_INPUTS = DataArtifact(
    filename=REFERENCE_ARCHIVE,
    extract_dir=paths.REFERENCE,
    description="the frozen RCSB search results and structures",
)
DESIGN_TABLES = DataArtifact(
    filename=DESIGN_ARCHIVE,
    extract_dir=paths.CAMPAIGNS,
    description="campaign structures, model scores, traces, and protocols",
)


def reference_members() -> list[Path]:
    """List frozen reference queries, search hits, metadata, and compressed CIFs."""
    from analysis import rcsb, reference

    members = [
        paths.reference_table(name)
        for name in (
            rcsb.SEARCH_HITS,
            rcsb.ENTRY_METADATA,
            reference.QUERY_RECORD,
            "natural_binder_query_all_hits.json",
        )
    ]
    members.extend(sorted(rcsb.CIF_CACHE.glob("*.cif.gz")))
    missing = [p for p in members if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing reference inputs: {missing}")
    return members


def design_members() -> list[Path]:
    """Preserve the published campaigns' model outputs, never derived measurements."""
    from analysis.targets import N_SEEDS, publication_campaigns

    members = []
    for target, arm in publication_campaigns():
        directory = paths.campaign_dir(target, arm)
        required = [
            directory / name
            for name in (
                "all_results.parquet",
                "step_trace.parquet",
                "protocol.json",
            )
        ]
        protocol_path = paths.campaign_protocol(target, arm)
        protocol = json.loads(protocol_path.read_text()) if protocol_path.is_file() else {}
        required.extend(
            paths.campaign_complexes(target, arm) / f"seed{seed:02d}.pdb"
            for seed in range(protocol.get("n_seeds", N_SEEDS))
        )
        missing = [p for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete campaign inputs: {missing[:5]}")
        members.extend(required)
    return members
