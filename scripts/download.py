# Reproducibly download only the two raw files fixed by configs/replogle.yaml.
# Transfers resume from existing bytes and are accepted only after integrity checks.
from hashlib import file_digest
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

config = yaml.safe_load(Path("configs/replogle.yaml").read_text())
destination = Path("data/raw")
destination.mkdir(parents=True, exist_ok=True)
for context, source in config["data"]["files"].items():
    path = destination / source["filename"]
    offset = path.stat().st_size if path.exists() else 0
    if offset < source["bytes"]:
        request = Request(f"https://ndownloader.figshare.com/files/{source['file_id']}")
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        with urlopen(request) as response, path.open("ab" if offset else "wb") as output:
            assert response.status == (206 if offset else 200)
            while chunk := response.read(8 * 1024**2):
                output.write(chunk)
    assert path.stat().st_size == source["bytes"]
    if source["md5"]:
        with path.open("rb") as downloaded:
            assert file_digest(downloaded, "md5").hexdigest() == source["md5"]
    with path.open("rb") as downloaded:
        assert file_digest(downloaded, "sha256").hexdigest() == source["sha256"]
    print(context, path, path.stat().st_size)
