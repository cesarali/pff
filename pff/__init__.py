from pathlib import Path


def _load_key_file(path: Path) -> str | None:
    """Return the contents of a key file if it exists, otherwise ``None``."""

    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        # If the file is unreadable we surface the issue by returning ``None``
        # so callers can decide how to handle missing credentials.
        return None


base_dir = Path(__file__).resolve().parent
project_dir = (base_dir / "..").resolve()
data_dir = project_dir / "data"
test_resources_dir = project_dir / "tests" / "resources"
results_dir = project_dir / "results"
reports_dir = project_dir / "reports"
config_dir = project_dir / "config_files"

comet_keys_file = project_dir / "COMET_KEYS.txt"
hf_keys_file = project_dir / "KEYS.txt"

COMET_KEY = _load_key_file(comet_keys_file)
HUGGINGFACE_KEY = _load_key_file(hf_keys_file)

__all__ = [
    "COMET_KEY",
    "HUGGINGFACE_KEY",
    "base_dir",
    "comet_keys_file",
    "config_dir",
    "data_dir",
    "hf_keys_file",
    "project_dir",
    "reports_dir",
    "results_dir",
    "test_resources_dir",
]
