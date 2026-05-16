from pathlib import Path

import mdtraj
import requests

ATLAS_BASE_URL = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/"

_session = requests.Session()


def default_cache_dir() -> Path:
    return Path("~/.cache/premval").expanduser()


def _download(chain: str, kind: str, xtc_path: Path, pdb_path: Path) -> None:
    raise NotImplementedError(
        f"ATLAS download not implemented; place {xtc_path} and {pdb_path} manually."
    )


def load_chain_trajectory(
    chain: str,
    kind: str = "analysis",
    cache_dir: Path | None = None,
) -> mdtraj.Trajectory:
    if cache_dir is None:
        cache_dir = default_cache_dir()

    base = cache_dir / "atlas" / kind
    xtc_path = base / f"{chain}.xtc"
    pdb_path = base / f"{chain}.pdb"

    if not (xtc_path.exists() and pdb_path.exists()):
        _download(chain, kind, xtc_path, pdb_path)

    return mdtraj.load(str(xtc_path), top=str(pdb_path))
