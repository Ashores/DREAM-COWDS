"""Two-process CPU test of restored gradients and task transitions; no GPUs used."""
import json
import os
import torch
import torch.distributed as dist

from continual_lora.methods import AdapterController, inject_continual_lora
from continual_lora.trainer import _all_reduce_gradients, _focus_backward
from test_continual_lora import SemanticFocusToy


def main():
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(414)
    model = SemanticFocusToy()
    inject_continual_lora(model, 'cqsm_dual_lora', num_tasks=2, cqsm_restore_legacy=True)
    ctl = AdapterController(model, 'cqsm_dual_lora')
    selector_ids = {id(p) for p in ctl.selector_parameters()}
    opt = torch.optim.AdamW([p for p in model.parameters()
                            if p.requires_grad and id(p) not in selector_ids], lr=.02, weight_decay=.1)
    old = None
    errors = []
    for stage in range(2):
        ctl.begin_task(stage)
        ctl.reset_focus_active_optimizer_state(opt)
        for p in ctl.focus_parameters(trainable_only=False):
            dist.broadcast(p.data, src=0)
        torch.manual_seed(700 + 10*stage + rank)
        x = torch.randn(2, 3, model.width) + rank
        model.eval()
        ctl.anchor_router().inference_oracle_stage = stage
        ctl.start_energy_covariance_collection()
        ctl.begin_anchor_batch()
        with torch.no_grad(): model(x)
        ctl.finalize_energy_object_subspaces(stage)
        ctl.anchor_router().inference_oracle_stage = None
        model.train()
        for _ in range(4):
            opt.zero_grad(set_to_none=True)
            ctl.begin_anchor_batch()
            output = model(x)
            losses = dict(semantic=output.square().mean(), box=-output.mean(),
                          mask=.3*output.abs().mean())
            total = sum(losses.values()) + ctl.regularization_loss() + .1*output.square().mean()
            _focus_backward(total, losses, ctl, True, strict_finite=True)
            _all_reduce_gradients(model, world)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            flat = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
            peers = [torch.empty_like(flat) for _ in range(world)]
            dist.all_gather(peers, flat)
            errors.append(float((peers[0]-peers[1]).abs().max()))
            opt.step()
            ctl.restore_focus_protected_parameters()
        if old is not None:
            for m, (a, b, cb) in zip(ctl.adapters(), old):
                assert torch.equal(m.lora_A[0], a) and torch.equal(m.lora_B[0], b)
                if cb is not None: assert torch.equal(m.core_B[0], cb)
        ctl.end_task()
        flat = torch.cat([v.detach().float().reshape(-1) for v in model.state_dict().values()])
        peers = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(peers, flat)
        errors.append(float((peers[0]-peers[1]).abs().max()))
        old = [(m.lora_A[0].clone(), m.lora_B[0].clone(),
                m.core_B[0].clone() if m.core_B is not None else None) for m in ctl.adapters()]
    assert max(errors) == 0
    assert bool((ctl.energy_object_core().general_scale[1] < 1).all())
    if rank == 0:
        print(json.dumps(dict(status='passed', backend='gloo', world_size=world,
            max_gradient_and_parameter_difference=max(errors),
            historical_slices_preserved=True, calibration_active=True)), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__': main()
