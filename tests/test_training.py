from copy import deepcopy

import numpy as np
import pytest
import torch

from causalcelljepa.model import CellEncoder, CellJEPA
from causalcelljepa.training import (
    _data_loader,
    batch_views,
    epoch_order,
    export_canonical_teacher,
    initial_train_state,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
    stage1_split,
)


def _tiny_training_objects():
    model = CellJEPA(
        CellEncoder(
            vocab_size=20,
            token_dim=12,
            latent_queries=2,
            blocks=1,
            heads=3,
            ffn_dim=24,
            dropout=0.0,
            cell_dim=8,
        ),
        predictor_hidden=16,
    )
    optimizer = torch.optim.AdamW([*model.online.parameters(), *model.predictor.parameters()])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, optimizer, scheduler


def test_stage1_split_order_and_per_cell_masks_are_replayable():
    cell_ids = [f"cell-{index}" for index in range(100)]
    train, validation = stage1_split(cell_ids, 0.2, 17)
    assert not np.intersect1d(train, validation).size
    assert sorted([*train, *validation]) == list(range(100))
    assert epoch_order(train, 17, 2) == epoch_order(train, 17, 2)
    assert epoch_order(train, 17, 2) != epoch_order(train, 17, 3)

    batch = {
        "gene_ids": torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
        "values": torch.ones(2, 4),
        "padding_mask": torch.zeros(2, 4, dtype=torch.bool),
        "cell_id": ["a", "b"],
    }
    first, teacher = batch_views(batch, (), 2, 17, 20, torch.device("cpu"))
    second, _ = batch_views(batch, (), 2, 17, 20, torch.device("cpu"))
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert torch.equal(teacher["gene_ids"], batch["gene_ids"])

    torch.manual_seed(123)
    before = torch.get_rng_state()
    next(iter(_data_loader(list(range(4)), range(4), 2, 0, 99)))
    assert torch.equal(torch.get_rng_state(), before)


def test_checkpoint_round_trip_provenance_and_teacher_export_guard(tmp_path):
    model, optimizer, scheduler = _tiny_training_objects()
    state = initial_train_state()
    state["global_step"] = 3
    provenance = {"frozen": "hash"}
    path = save_checkpoint(
        tmp_path / "checkpoint.pt", model, optimizer, scheduler, state, {"x": 1}, provenance
    )
    expected = deepcopy(model.state_dict())
    with torch.no_grad():
        next(model.online.parameters()).add_(5)
    loaded_state, configuration = load_checkpoint(
        path, model, optimizer, scheduler, provenance, "cpu"
    )
    assert loaded_state["global_step"] == 3 and configuration == {"x": 1}
    assert all(torch.equal(value, expected[key]) for key, value in model.state_dict().items())
    with pytest.raises(ValueError, match="provenance"):
        load_checkpoint(path, model, optimizer, scheduler, {"wrong": "hash"}, "cpu")
    with pytest.raises(ValueError, match="forbidden"):
        export_canonical_teacher(tmp_path / "teacher.pt", model, state, {"stage1": {}}, provenance)


def test_learning_rate_schedule_warms_then_cosines_to_floor():
    schedule = learning_rate_schedule(100, 0.1, 0.05)
    assert schedule(0) == 0.1
    assert schedule(9) == 1.0
    assert schedule(10) == 1.0
    assert schedule(100) == 0.05
