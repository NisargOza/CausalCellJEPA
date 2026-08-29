# Build the outcome-free ESM+GO cache with explicit teacher availability.
from causalcelljepa.actions import prepare_multiteacher_action

manifest = prepare_multiteacher_action(
    "configs/contextual_multiteacher_action.yaml", include_availability=True
)
print(manifest["artifact"])
print(manifest["report"])
