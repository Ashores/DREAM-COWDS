"""Two-GPU synchronization check using the production CQSM update functions.

This is a mechanism test on a small SAM-shaped network, not a performance run.
"""
import json
import os
import torch
import torch.distributed as dist

from continual_lora.methods import AdapterController, inject_continual_lora
from continual_lora.trainer import _all_reduce_gradients
from test_continual_lora import SemanticFocusToy

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
torch.manual_seed(414)
model = SemanticFocusToy()
inject_continual_lora(model, "cqsm_dual_lora", rank=16, num_tasks=2)
model.cuda(rank)
controller = AdapterController(model, "cqsm_dual_lora")
selector_ids = {id(p) for p in controller.selector_parameters()}
optimizer = torch.optim.AdamW([p for p in model.parameters()
                              if p.requires_grad and id(p) not in selector_ids],
                             lr=.02, weight_decay=.1)
gradient_max_differences = []
parameter_max_differences = []
history = None
global_changes = []
global_parameter = next(p for name, p in model.named_parameters() if name.endswith("global_B"))
for stage in range(2):
    controller.begin_task(stage)
    controller.reset_focus_active_optimizer_state(optimizer)
    for parameter in controller.focus_parameters(trainable_only=False):
        dist.broadcast(parameter.data, src=0)
    torch.manual_seed(700 + 10 * stage + rank)
    x = torch.randn(2, 3, model.width, device=rank) + rank
    model.eval()
    controller.anchor_router().inference_oracle_stage = stage
    controller.start_energy_covariance_collection()
    controller.begin_anchor_batch()
    with torch.no_grad():
        model(x)
    controller.finalize_energy_object_subspaces(stage)
    controller.anchor_router().inference_oracle_stage = None
    model.train()
    global_before = global_parameter.detach().clone()
    for _ in range(2):
        controller.begin_anchor_batch()
        optimizer.zero_grad(set_to_none=True)
        (model(x).square().mean() * (rank + 1)).backward()
        _all_reduce_gradients(model, world)
        flat = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.requires_grad])
        peers = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(peers, flat)
        difference = float((peers[0] - peers[1]).abs().max())
        gradient_max_differences.append(difference)
        assert difference == 0.0
        optimizer.step()
        controller.restore_focus_protected_parameters()
        flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        peers = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(peers, flat)
        difference = float((peers[0] - peers[1]).abs().max())
        parameter_max_differences.append(difference)
        assert difference == 0.0
    global_changes.append(float((global_parameter - global_before).abs().max()))
    assert global_changes[-1] > 0
    if history is not None:
        for module, (a, b, cb) in zip(controller.adapters(), history):
            assert torch.equal(module.lora_A[0], a)
            assert torch.equal(module.lora_B[0], b)
            if cb is not None:
                assert torch.equal(module.core_B[0], cb)
    history = [(m.lora_A[0].clone(), m.lora_B[0].clone(),
                m.core_B[0].clone() if m.core_B is not None else None)
               for m in controller.adapters()]
    controller.end_task()
if rank == 0:
    print(json.dumps({"test": "CQSM_two_GPU_manual_all_reduce", "world_size": world,
                      "gradient_max_differences": gradient_max_differences,
                      "parameter_max_differences": parameter_max_differences,
                      "global_parameter_change_per_task": global_changes,
                      "historical_private_and_bank_readout_bitwise_preserved": True}), flush=True)
dist.destroy_process_group()
