# Resume and verify the two test-only Nadig/scPertEval files fixed in the external protocol.
# Downloading does not inspect expression outcomes; preparation is a separate audited step.
from hashlib import file_digest
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

config = yaml.safe_load(Path("configs/nadig_external_validation.yaml").read_text())
destination = Path(config["source"]["raw_directory"])
destination.mkdir(parents=True, exist_ok=True)
for context, source in config["source"]["files"].items():
    path = destination / source["filename"]
    offset = path.stat().st_size if path.exists() else 0
    if offset < source["bytes"]:
        request = Request(source["url"])
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        with urlopen(request) as response, path.open("ab" if offset else "wb") as output:
            assert response.status == (206 if offset else 200)
            while chunk := response.read(8 * 1024**2):
                output.write(chunk)
    assert path.stat().st_size == source["bytes"]
    with path.open("rb") as downloaded:
        assert file_digest(downloaded, "md5").hexdigest() == source["md5"]
    print(context, path, path.stat().st_size)
