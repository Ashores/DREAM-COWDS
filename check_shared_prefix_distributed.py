"""Exercise shared retention, native gradient averaging and protected task slices."""
import json
import argparse
import torch
import torch.distributed as dist
from continual_lora.trainer import _all_reduce_gradients, _focus_backward, _project_functional_gradients
from continual_lora.global_coordination import clip_adapter_gradients
from test_shared_prefix import setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', choices=['visual_prefix', 'visual_residual'], default='visual_prefix')
    parser.add_argument('--global-coordination', action='store_true')
    parser.add_argument('--global-gating', action='store_true')
    parser.add_argument('--online-deconflict', action='store_true')
    args = parser.parse_args()
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.global_coordination:
        # A rank-local conflict is deliberately different from the batch-level
        # conflict; this checks averaging before projection, plus preservation
        # of a fourth objective in the ordinary backward pass.
        from test_global_coordination import ScalarController
        from torch import nn
        analytic = nn.Module(); analytic.global_B = nn.Parameter(torch.tensor([.2, .3]))
        coefficients = torch.tensor([[[2., 1.], [-1., 0.], [0., 1.]],
                                     [[0., 1.], [1., -2.], [1., 0.]]])
        loss = {name: (analytic.global_B*row).sum()
                for name, row in zip(('semantic', 'box', 'mask'), coefficients[rank])}
        _focus_backward(sum(loss.values())+2*analytic.global_B.sum(), loss,
                        ScalarController(analytic), True, strict_finite=True,
                        global_conflict=True, global_world_size=world)
        _all_reduce_gradients(analytic, world)
        reference, _, _ = _project_functional_gradients([(row,) for row in coefficients.mean(0)])
        torch.testing.assert_close(analytic.global_B.grad, reference[0]+2, rtol=1e-6, atol=1e-6)
    model, ctl, keeper, x, injection = setup(checkpointed=True, layout=args.layout, global_gating=args.global_gating)
    if args.online_deconflict:
        from continual_lora.online_deconflict import attach_online_deconflict,online_deconflict_loss
        attach_online_deconflict(model,injection)
        ctl.online_deconflict=True
    selector_ids = {id(p) for p in ctl.selector_parameters()}
    from continual_lora.global_coordination import global_parameters
    shared_parameters=global_parameters(model)
    shared_ids = {id(p) for p in shared_parameters}
    private = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad
        and id(p) not in (selector_ids | shared_ids)], lr=.003, weight_decay=.01)
    shared = torch.optim.AdamW(shared_parameters, lr=.003, weight_decay=.01)
    fixed_a = [m.global_A.clone() for _,m in keeper.shared]
    errors, ratios = [], []
    old = None
    for stage in range(2):
        keeper.begin_task(stage)
        if stage: ctl.begin_task(stage)
        ctl.reset_focus_active_optimizer_state(private)
        for p in ctl.focus_parameters(trainable_only=False): dist.broadcast(p.data, src=0)
        torch.manual_seed(117+stage*10+rank)
        current = x + rank + stage*.1
        if stage:
            model.eval(); ctl.anchor_router().inference_oracle_stage=stage
            ctl.start_energy_covariance_collection(); ctl.begin_anchor_batch()
            with torch.no_grad(): model(current)
            ctl.finalize_energy_object_subspaces(stage)
            ctl.anchor_router().inference_oracle_stage=None
        model.train()
        for step in range(4):
            private.zero_grad(set_to_none=True); shared.zero_grad(set_to_none=True)
            keeper.begin_batch(); ctl.begin_anchor_batch(); output=model(current)
            feature_loss, gradients=keeper.loss_and_gradients()
            losses=dict(semantic=output.square().mean(),box=-output.mean(),mask=.3*output.abs().mean())
            auxiliary=online_deconflict_loss(model)[0] if args.online_deconflict else output.new_zeros(())
            stats = _focus_backward(sum(losses.values())+ctl.regularization_loss()+auxiliary,losses,ctl,True,
                strict_finite=True, global_conflict=args.global_coordination, global_world_size=world)
            if args.global_coordination:
                assert stats['global_functional_projection_applied'] == 1
            _all_reduce_gradients(model,world)
            report=keeper.apply(gradients,world_size=world)
            if stage and step:
                assert feature_loss>0
                ratios.append(report['shared_retention_gradient_ratio'])
            flat=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            peers=[torch.empty_like(flat) for _ in range(world)]; dist.all_gather(peers,flat)
            errors.append(float((peers[0]-peers[1]).abs().max()))
            clip_adapter_gradients(model, ctl, args.global_coordination)
            private.step();shared.step();ctl.restore_focus_protected_parameters()
        if old is not None:
            for m,(a,b,cb) in zip(ctl.adapters(),old):
                assert torch.equal(m.lora_A[0],a) and torch.equal(m.lora_B[0],b)
                if cb is not None: assert torch.equal(m.core_B[0],cb)
        ctl.end_task()
        old=[(m.lora_A[0].clone(),m.lora_B[0].clone(),m.core_B[0].clone() if m.core_B is not None else None)
             for m in ctl.adapters()]
        for (_,m),a in zip(keeper.shared,fixed_a):
            assert torch.equal(m.global_A,a) and torch.count_nonzero(m.global_B[model.width:2*model.width])==0
        flat=torch.cat([v.detach().float().flatten() for v in model.state_dict().values()])
        peers=[torch.empty_like(flat) for _ in range(world)];dist.all_gather(peers,flat)
        errors.append(float((peers[0]-peers[1]).abs().max()))
    assert max(errors)==0 and all(abs(r-.2)<1e-5 for r in ratios)
    if rank==0:
        print(json.dumps(dict(status='passed',layout=args.layout,world_size=world,max_gradient_and_parameter_difference=max(errors),
            global_coordination=args.global_coordination,global_gating=args.global_gating,
            online_deconflicting=args.online_deconflict,
            shared_retention_gradient_ratios=ratios,historical_private_slices_preserved=True,
            frozen_shared_down_factors_preserved=True,key_projection_unmodified=True)))
    keeper.close();dist.destroy_process_group()


if __name__=='__main__':main()
