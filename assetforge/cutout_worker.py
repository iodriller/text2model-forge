"""Background-removal worker. Runs inside the TripoSR venv, which owns rembg/onnxruntime.

Reads "source|destination" path pairs from stdin, one per line, and writes RGBA cutouts.
Kept dependency-free of the assetforge package so any Python with rembg can execute it.
"""

import sys

from rembg import new_session, remove


def main() -> int:
    session = new_session("u2net")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        source, destination = line.split("|", 1)
        with open(source, "rb") as handle:
            data = handle.read()
        result = remove(data, session=session)
        with open(destination, "wb") as handle:
            handle.write(result)
        print(f"OK {destination}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
