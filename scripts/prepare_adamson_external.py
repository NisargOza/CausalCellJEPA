# Freeze Adamson source hashes, cohort membership, and gene overlap before outcome parsing.
# The preparation function never decompresses the expression matrix.
import json

from causalcelljepa.external import prepare_adamson_external

print(json.dumps(prepare_adamson_external(), indent=2, sort_keys=True))
