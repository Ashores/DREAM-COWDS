import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from continual_lora.methods import CQSMDualLoRALinear,shared_only_adapter_forward
from continual_lora.online_deconflict import (attach_online_deconflict,online_deconflict_loss,
                                              set_deconflict_strength,polar,spd_sqrt)
from continual_lora.global_coordination import global_parameters
from continual_lora.trainer import adapter_state_dict,load_adapter_checkpoint
from test_shared_prefix import setup


def make(checkpointed=False,nonzero=True):
    model,ctl,keeper,x,report=setup(checkpointed,layout='visual_residual')
    report=attach_online_deconflict(model,report)
    ctl.online_deconflict=True
    if nonzero:
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m,CQSMDualLoRALinear):
                    m.lora_B[0].normal_(std=.03)
                    if m.shared_enabled:
                        m.global_B.normal_(std=.03)
                        m.global_B.mul_(m.global_output_mask)
    return model,ctl,keeper,x,report


@pytest.mark.parametrize('scale',[0.,.3,1.])
def test_batched_polar_is_finite_at_zero_or_repeated_singular_values(scale):
    torch.manual_seed(72)
    b=(torch.linalg.qr(torch.randn(3,24,4)).Q*scale).requires_grad_()
    q,h=polar(b)
    root=spd_sqrt(h)
    assert torch.isfinite(q).all() and torch.isfinite(root).all()
    (q.square().sum()+h.square().sum()+root.square().sum()).backward()
    assert torch.isfinite(b.grad).all()
    if scale:
        torch.testing.assert_close(q@h,b,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('checkpointed',[False,True])
def test_zero_start_preserves_native_forward_and_lora_gradients(checkpointed):
    model,ctl,keeper,x,_=make(checkpointed,nonzero=False)
    model.train()
    ctl.begin_anchor_batch();set_deconflict_strength(model,0.)
    torch.manual_seed(88);reference=model(x)
    ctl.begin_anchor_batch();set_deconflict_strength(model,1.)
    torch.manual_seed(88);actual=model(x)
    assert torch.equal(actual,reference)
    loss,report=online_deconflict_loss(model)
    assert torch.isfinite(loss)
    (actual.square().mean()+loss).backward()
    shared=[m for m in model.modules() if isinstance(m,CQSMDualLoRALinear) and m.shared_enabled]
    assert all(m.global_B.grad is not None and m.global_B.grad.abs().sum()>0 for m in shared)
    assert all(m.lora_B.grad is not None and m.lora_B.grad.abs().sum()>0 for m in shared)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    keeper.close()


@pytest.mark.parametrize('checkpointed',[False,True])
def test_live_rewriting_and_structural_gradients_exclude_private(checkpointed):
    model,ctl,keeper,x,_=make(checkpointed)
    model.train();ctl.begin_anchor_batch()
    output=model(x)
    loss,report=online_deconflict_loss(model)
    private=[p for n,p in model.named_parameters() if n.endswith(('.lora_A','.lora_B','.core_B')) and p.requires_grad]
    gradients=torch.autograd.grad(loss,private,allow_unused=True,retain_graph=True)
    assert all(g is None for g in gradients)
    (output.square().mean()+loss).backward()
    bridge=[p for n,p in model.named_parameters() if '.sdo_bridge.' in n]
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in bridge)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    global_ids={id(p) for p in global_parameters(model)}
    assert all(id(p) in global_ids for p in bridge)
    keeper.close()


def test_cache_refreshes_with_live_factors_and_preserves_k_and_input_space():
    model,ctl,keeper,x,_=make()
    model.eval();ctl.anchor_router().inference_oracle_stage=0
    ctl.begin_anchor_batch();model(x)
    module=keeper.shared[0][1]
    bridge=module.sdo_bridge
    first=bridge.prepare(module)
    assert bridge.prepare(module) is first
    with torch.no_grad():
        module.global_B.add_(.01*module.global_output_mask)
        module.lora_B[0].mul_(1.2)
        for part in bridge.parts.values():part.network[-1].weight.normal_(std=.1)
    second=bridge.prepare(module)
    assert second is not first
    correction=bridge(x,module)
    width=module.in_features
    assert torch.count_nonzero(correction[...,width:2*width])==0
    assert correction.abs().sum()>0
    input_null=torch.eye(width)-module.global_A.T@module.global_A
    for name,part in bridge.parts.items():
        db=part.correction(second[name],bridge.strength)[0]
        leak=db@module.global_A@input_null
        torch.testing.assert_close(leak,torch.zeros_like(leak),atol=2e-6,rtol=0)
    keeper.close()


def test_shared_only_retention_bypasses_private_conditioned_operator():
    model,ctl,keeper,x,_=make(checkpointed=True)
    module=keeper.shared[0][1]
    with torch.no_grad():
        for part in module.sdo_bridge.parts.values():part.network[-1].weight.normal_(std=.2)
    with shared_only_adapter_forward():reference=module(x)
    set_deconflict_strength(model,0.)
    with shared_only_adapter_forward():actual=module(x)
    assert torch.equal(reference,actual)
    # Test the real two-task retention graph and native checkpoint replay.
    ctl.begin_task(1);keeper.begin_task(1)
    with torch.no_grad():
        for _,m in keeper.shared:
            m.global_B.add_(.02*m.global_output_mask)
            m.lora_B[1].normal_(std=.03)
    model.train();set_deconflict_strength(model,1.)
    keeper.begin_batch();ctl.begin_anchor_batch();value=model(x)
    retained,gradients=keeper.loss_and_gradients()
    assert retained>0
    (value.square().mean()+online_deconflict_loss(model)[0]).backward()
    report=keeper.apply(gradients)
    assert report['shared_retention_applied']==1
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    keeper.close()


def test_online_operator_does_not_reset_rng_or_change_original_initialization():
    torch.manual_seed(52)
    model,ctl,keeper,x,report=setup(layout='visual_residual')
    original={n:p.clone() for n,p in model.named_parameters()}
    state=torch.get_rng_state().clone()
    attach_online_deconflict(model,report)
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(original[n],p) for n,p in model.named_parameters() if n in original)
    keeper.close()


def test_parameter_accounting_and_schema6_roundtrip(tmp_path,monkeypatch):
    from continual_lora.cqsm_accounting import cqsm_parameter_accounting
    from continual_lora import trainer
    from test_shared_prefix import PrefixToy
    monkeypatch.setattr(trainer,'enable_lora_training_compatibility',lambda:None)
    model,ctl,keeper,x,report=make()
    with torch.no_grad():
        for _,module in keeper.shared:
            for part in module.sdo_bridge.parts.values():
                part.network[-1].weight.normal_(std=.1)
    model.eval();ctl.anchor_router().inference_oracle_stage=0
    ctl.begin_anchor_batch();expected=model(x)
    accounting=cqsm_parameter_accounting(model,ctl,0)
    extra=sum(p.numel() for n,p in model.named_parameters() if '.sdo_bridge.' in n)
    assert accounting['counts']['deconflict_allocated']==extra
    assert accounting['counts']['deconflict_current_optimizable']==extra
    config=dict(method='cqsm_dual_lora',rank=16,alpha=8.,stream=[{},{}],
        focus_group_ranks=report.group_ranks,cqsm_schema_version=6,cqsm_restore_legacy=True,
        cqsm_shared=True,cqsm_private=True,cqsm_bank_mode='all',cqsm_basis='covariance',
        cqsm_shared_layout='visual_residual',global_rank=10,cqsm_global_gating=False,
        energy_calibration_lambda=3.,focus_orthogonality_weight=.001,
        global_deconflict=True,sdo_hidden=32,sdo_max_correction=.25,
        sdo_conflict_weight=.05,sdo_preservation_weight=1.)
    payload=dict(config=config,completed_task=0,adapter=adapter_state_dict(model),controller=ctl.state_dict())
    path=tmp_path/'online.pt';torch.save(payload,path)
    torch.manual_seed(19);restored=PrefixToy()
    fresh,_,_=load_adapter_checkpoint(restored,path)
    restored.eval();fresh.anchor_router().inference_oracle_stage=0
    fresh.begin_anchor_batch()
    torch.testing.assert_close(restored(x),expected)
    assert fresh.online_deconflict
    damaged=copy.deepcopy(payload)
    key=next(k for k in damaged['adapter'] if '.sdo_bridge.' in k)
    del damaged['adapter'][key]
    torch.save(damaged,tmp_path/'incomplete.pt')
    with pytest.raises(RuntimeError,match='Incomplete CQSM'):
        load_adapter_checkpoint(PrefixToy(),tmp_path/'incomplete.pt')
    keeper.close()
