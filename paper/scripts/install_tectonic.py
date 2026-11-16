#!/usr/bin/env python3
"""Install pinned official Tectonic binary into ignored paper/tools/."""
import hashlib
import io
import platform
from pathlib import Path
import tarfile
import tempfile
from urllib.request import urlopen

VERSION = "0.17.0"
# Official release: https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0
ASSETS = {
    ("Linux", "x86_64"): (
        "x86_64-unknown-linux-musl",
        "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7",
    ),
}


def main():
    system = platform.system()
    machine = platform.machine()
    asset = ASSETS.get((system, machine))
    if asset is None:
        raise SystemExit(
            f"Automatic setup supports Linux x86_64; detected {system} {machine}. "
            "On macOS use `brew install tectonic`; on other platforms install "
            "Tectonic from https://tectonic-typesetting.github.io/book/latest/installation/ "
            "and run `make pdf`."
        )
    target, expected = asset
    url = (
        "https://github.com/tectonic-typesetting/tectonic/releases/download/"
        f"tectonic%40{VERSION}/tectonic-{VERSION}-{target}.tar.gz"
    )
    print(f"Downloading {url}")
    with urlopen(url, timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != expected:
        raise SystemExit("Download checksum mismatch; refusing to install.")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        member = bundle.getmember("tectonic")
        if not member.isfile():
            raise SystemExit("Release binary is not a regular file.")
        binary = bundle.extractfile(member).read()
    destination = Path(__file__).resolve().parents[1] / "tools" / "tectonic"
    destination.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
        temporary = Path(output.name)
        output.write(binary)
    try:
        temporary.chmod(0o755)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Installed {destination}")


if __name__ == "__main__":
    main()
