"""Download and cache ATLAS MD trajectory bundles.

The ATLAS HTTP API (https://www.dsimb.inserm.fr/ATLAS/api) serves per-chain
zip archives at `/ATLAS/{kind}/{pdb_chain}` for several payload sizes:

- `analysis`: protein-only trajectories at 1000 frames per replicate plus
  analysis TSVs (~15-30 MB per chain). Default for dev work.
- `protein`: protein-only trajectories at 10000 frames per replicate.
- `total`: complete system at 10000 frames per replicate (largest).

Archives are returned as zip despite the `.tar.gz` filenames sometimes used
in the literature; we store them as `{chain}.zip` in the cache.
"""

from __future__ import annotations

import csv
import logging
import tempfile
import time
import zipfile
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    import mdtraj as md

AtlasKind = Literal["analysis", "protein", "total"]
ATLAS_KINDS: tuple[AtlasKind, ...] = ("analysis", "protein", "total")
ATLAS_REPLICAS: tuple[int, int, int] = (1, 2, 3)

_API_BASE = "https://www.dsimb.inserm.fr/ATLAS/api"
_VAL_CSV = "atlas_val.csv"
_DOWNLOAD_TIMEOUT_S = 600
_CHUNK_SIZE = 1 << 16
# Two separate retry budgets that compound: urllib3 handles transient
# transport failures during request setup / initial read, while the outer
# loop in `_download_chain` covers mid-stream disconnects that fall outside
# urllib3's window once `stream=True` is iterating.
_SESSION_RETRIES = 3
_BUNDLE_RETRIES = 5
_RETRY_BACKOFF_S = 1.0

_LOG = logging.getLogger(__name__)


def default_cache_dir() -> Path:
    """Return the default ATLAS cache directory under the user home.

    `~/.cache/premval/atlas/` keeps trajectory data out of the repo tree
    while giving every dev a predictable shared location.
    """
    return Path.home() / ".cache" / "premval" / "atlas"


def load_val_chains() -> list[str]:
    """Return the 39 PDB chain identifiers in the AlphaFlow ATLAS val split.

    Reads the vendored `atlas_val.csv` shipped inside the package and
    returns the `name` column (e.g., `6cka_B`).
    """
    csv_text = resources.files(__package__).joinpath(_VAL_CSV).read_text(encoding="utf-8")
    reader = csv.DictReader(csv_text.splitlines())
    return [row["name"] for row in reader]


def _bundle_path(cache_dir: Path, kind: AtlasKind, chain: str) -> Path:
    return cache_dir / kind / f"{chain}.zip"


def _build_session() -> requests.Session:
    """Return a session that retries dropped connections and 5xx responses."""
    sess = requests.Session()
    retry = Retry(
        total=_SESSION_RETRIES,
        connect=_SESSION_RETRIES,
        read=_SESSION_RETRIES,
        backoff_factor=_RETRY_BACKOFF_S,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def _download_chain(chain: str, kind: AtlasKind, dest: Path, session: requests.Session) -> None:
    """Stream one chain bundle to `dest`, writing via a `.part` temp file.

    The ATLAS server periodically resets streaming connections mid-body,
    which falls outside the urllib3 Retry window (that only covers setup
    and initial read). We retry the entire request on transport errors,
    with exponential backoff bounded by `_BUNDLE_RETRIES`.
    """
    url = f"{_API_BASE}/ATLAS/{kind}/{chain}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(_BUNDLE_RETRIES):
        try:
            with session.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_S) as response:
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(dest)
            return
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
            last_error = exc
            _LOG.warning(
                "ATLAS download for %s failed (attempt %d/%d): %s",
                chain,
                attempt + 1,
                _BUNDLE_RETRIES,
                exc,
            )
            time.sleep(_RETRY_BACKOFF_S * (2**attempt))
    tmp.unlink(missing_ok=True)
    assert last_error is not None
    raise last_error


def fetch_val_split(
    cache_dir: Path | None = None,
    *,
    kind: AtlasKind = "analysis",
    chains: Iterable[str] | None = None,
    force: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Path]:
    """Download ATLAS bundles for the val split into the local cache.

    Resumable: cached bundles are kept and skipped, so a fetch that fails
    on chain N leaves chains 0..N-1 on disk and a re-invocation only
    downloads what is still missing.

    Args:
        cache_dir: Root cache directory. Defaults to `default_cache_dir()`.
            Bundles are written to `cache_dir/{kind}/{chain}.zip`.
        kind: Which ATLAS payload to fetch. Defaults to `"analysis"` which
            is the smallest (1000 frames/replicate, protein-only) and
            sufficient for evaluation development.
        chains: Iterable of chain identifiers to fetch. Defaults to the
            full 39-chain val split.
        force: If True, re-download even if a cached file exists.
        session: Optional pre-configured `requests.Session`. A fresh
            session is created and closed if not supplied.

    Returns:
        Mapping from chain identifier to the on-disk bundle path. Every
        returned path exists.

    Raises:
        requests.HTTPError: If the ATLAS API returns a non-2xx status.
        requests.ConnectionError: If the connection cannot be established
            or all `_BUNDLE_RETRIES` mid-stream retries are exhausted.
        requests.exceptions.ChunkedEncodingError: If streaming decoding
            fails on every retry.
    """
    root = cache_dir or default_cache_dir()
    target_dir = root / kind
    target_dir.mkdir(parents=True, exist_ok=True)

    targets = list(chains) if chains is not None else load_val_chains()
    owned_session = session is None
    sess = session or _build_session()
    try:
        results: dict[str, Path] = {}
        for index, chain in enumerate(targets, start=1):
            path = _bundle_path(root, kind, chain)
            if not force and path.exists():
                _LOG.info("[%d/%d] %s: cached", index, len(targets), chain)
            else:
                _LOG.info("[%d/%d] %s: downloading", index, len(targets), chain)
                _download_chain(chain, kind, path, sess)
            results[chain] = path
        return results
    finally:
        if owned_session:
            sess.close()


def load_chain_trajectory(
    chain: str,
    *,
    kind: AtlasKind = "analysis",
    cache_dir: Path | None = None,
) -> md.Trajectory:
    """Load a cached ATLAS bundle as one concatenated `mdtraj.Trajectory`.

    The bundle ships a topology PDB and three replica XTCs
    (`{chain}.pdb`, `{chain}_R{1,2,3}.xtc`). This loader extracts those
    files to a temp directory, loads each replica against the shared
    topology, and concatenates R1+R2+R3 in order (matching AlphaFlow's
    `prep_atlas.py` pattern). The topology PDB is used for atom/residue
    metadata only, never as a frame.

    Args:
        chain: PDB chain identifier such as `6cka_B`.
        kind: ATLAS payload tier; must match the cached bundle's tier.
        cache_dir: Root cache directory. Defaults to `default_cache_dir()`.

    Returns:
        A single trajectory with `n_frames` equal to the sum of frames
        across the three replicas (1000 each for `analysis`, 10000 each
        for `protein`/`total`).

    Raises:
        FileNotFoundError: If the chain bundle is not cached locally;
            call `fetch_val_split` first.
        KeyError: If the bundle is missing the topology PDB or any
            replica XTC.
    """
    import mdtraj as md

    root = cache_dir or default_cache_dir()
    bundle = _bundle_path(root, kind, chain)
    if not bundle.exists():
        raise FileNotFoundError(
            f"ATLAS bundle not cached at {bundle}; run fetch_val_split(chains=[{chain!r}])"
        )

    topology_name = f"{chain}.pdb"
    replica_names = [f"{chain}_R{r}.xtc" for r in ATLAS_REPLICAS]

    with zipfile.ZipFile(bundle) as zf, tempfile.TemporaryDirectory() as tmp:
        members = set(zf.namelist())
        for name in (topology_name, *replica_names):
            if name not in members:
                raise KeyError(f"{bundle}: missing expected member {name!r}")
        tmpdir = Path(tmp)
        zf.extract(topology_name, tmpdir)
        for name in replica_names:
            zf.extract(name, tmpdir)

        topology = md.load(str(tmpdir / topology_name)).topology
        replicas = [md.load(str(tmpdir / name), top=topology) for name in replica_names]

    return md.join(replicas, check_topology=True)
