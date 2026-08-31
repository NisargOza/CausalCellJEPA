# Build the outcome-free rank-64 STRING teacher and append it to frozen ESM+GO actions.
# The source file is checksum-pinned to the public TxPert repository commit in the config.
from causalcelljepa.actions import prepare_string_action

print(prepare_string_action())
