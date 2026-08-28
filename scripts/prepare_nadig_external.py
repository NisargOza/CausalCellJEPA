# Freeze external source hashes, schemas, gene overlap, and eligible target metadata.
# This preparation intentionally never indexes either external expression matrix.
import json

from causalcelljepa.external import prepare_nadig_external

print(json.dumps(prepare_nadig_external(), indent=2, sort_keys=True))
