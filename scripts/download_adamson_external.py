# Resume and verify every official Adamson file pinned by the final confirmation protocol.
# Downloading and hashing do not decompress or inspect the expression matrix.
from hashlib import file_digest
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

config = yaml.safe_load(Path("configs/adamson_external_confirmation.yaml").read_text())
destination = Path(config["source"]["raw_directory"])
destination.mkdir(parents=True, exist_ok=True)
for source in config["source"]["files"].values():
    path = destination / source["filename"]
    offset = path.stat().st_size if path.exists() else 0
    if offset < source["bytes"]:
        request = Request(source["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
        with urlopen(request) as response, path.open("ab" if offset else "wb") as output:
            assert response.status == (206 if offset else 200)
            while chunk := response.read(8 * 1024**2):
                output.write(chunk)
    assert path.stat().st_size == source["bytes"]
    with path.open("rb") as downloaded:
        assert file_digest(downloaded, "sha256").hexdigest() == source["sha256"]
    print(path, path.stat().st_size, source["sha256"])
