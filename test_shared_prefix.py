import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from test_continual_lora import SemanticFocusToy
from continual_lora.methods import AdapterController, CQSMDualLoRALinear, inject_continual_lora
from continual_lora.shared_prefix import SharedPrefixRetention, SharedOnlyPrefixRetention
from continual_lora.trainer import adapter_state_dict, load_adapter_checkpoint, _focus_backward


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(width, 3*width)
        self.attn.proj = nn.Linear(width, width)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(width, width)
        self.mlp.fc2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        q, k, v = self.attn.qkv(x).chunk(3, dim=-1)
        x = x + self.dropout(self.attn.proj((q+k+v).tanh()))
        return x + self.dropout(self.mlp.fc2(self.mlp.fc1(x).relu()))


class PrefixToy(SemanticFocusToy):
    def __init__(self, checkpointed=False):
        super().__init__()
        self.backbone.vision_backbone.trunk.blocks = nn.ModuleList([Block(self.width) for _ in range(8)])
        self.checkpointed = checkpointed

    def forward(self, x):
        for block in self.backbone.vision_backbone.trunk.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.checkpointed else block(x)
        x = self.transformer.decoder.layers[0].linear1(x)
        return self.dot_prod_scoring.hs_proj(x)


def setup(checkpointed=False, layout='visual_prefix', shared=True, retention_ratio=.2,
          global_gating=False):
    torch.manual_seed(19)
    model = PrefixToy(checkpointed)
    report = inject_continual_lora(model, 'cqsm_dual_lora', rank=16, num_tasks=2,
                                  cqsm_restore_legacy=True, cqsm_shared_layout=layout, cqsm_shared=shared,
                                  cqsm_global_rank=10, cqsm_global_gating=global_gating)
    ctl = AdapterController(model, 'cqsm_dual_lora')
    keeper_class = SharedOnlyPrefixRetention if layout == 'visual_residual' else SharedPrefixRetention
    keeper = keeper_class(model, gradient_ratio=retention_ratio)
    x = torch.randn(2, 3, model.width)
    ctl.begin_task(0)
    model.eval(); ctl.anchor_router().inference_oracle_stage=0
    ctl.start_energy_covariance_collection(); ctl.begin_anchor_batch()
    with torch.no_grad():model(x)
    ctl.finalize_energy_object_subspaces(0)
    ctl.anchor_router().inference_oracle_stage=None; model.train()
    return model, ctl, keeper, x, report


def test_prefix_assignment_preserves_selector_inputs_and_qv_mask():
    model, ctl, keeper, x, report = setup()
    model.eval()
    assert keeper.block_ids == [4, 5]
    assert report.module_groups['global_shared'] == [
        'backbone.vision_backbone.trunk.blocks.4.attn.qkv',
        'backbone.vision_backbone.trunk.blocks.5.attn.qkv']
    for n,m in model.named_modules():
        if isinstance(m,CQSMDualLoRALinear) and '.blocks.' in n:
            block=int(n.split('.blocks.')[1].split('.')[0])
            assert m.private_enabled == (block >= 6)
    ctl.begin_anchor_batch(); model(x)
    original = ctl.anchor_router()._visual_descriptor.clone()
    for _,m in keeper.shared:
        with torch.no_grad():m.global_B.add_(.1)
        delta=m(x)-m.base(x)
        assert torch.count_nonzero(delta[...,model.width:2*model.width]) == 0
    ctl.begin_anchor_batch(); model(x)
    assert torch.equal(original,ctl.anchor_router()._visual_descriptor)
    keeper.close()


@pytest.mark.parametrize('checkpointed', [False, True])
def test_true_teacher_rng_and_retention_gradients(checkpointed):
    model,ctl,keeper,x,_=setup(checkpointed)
    keeper.begin_task(1);keeper.begin_batch();ctl.begin_anchor_batch();model(x)
    value,grads=keeper.loss_and_gradients()
    assert value == 0  # Equal snapshots and matched dropout, despite train mode.
    teacher=[p.clone() for p in keeper.previous]
    with torch.no_grad():
        for _,m in keeper.shared:m.global_B.add_(.03*m.global_output_mask)
    torch.manual_seed(117)
    keeper.begin_batch();ctl.begin_anchor_batch();y=model(x)
    value,grads=keeper.loss_and_gradients()
    assert value>0 and any(g.abs().sum()>0 for g in grads)
    assert all(torch.equal(a,b) for a,b in zip(teacher,keeper.previous))
    assert all(p.grad is None for p in model.parameters())  # grad queries do not write private gradients.
    native=y.square().mean()
    native.backward()
    private_before={id(p):p.grad.clone() for n,p in model.named_parameters()
                    if p.grad is not None and not n.endswith('global_B')}
    report=keeper.apply(grads)
    assert report['shared_retention_gradient_ratio']==pytest.approx(.2,rel=1e-5)
    assert all(torch.equal(private_before[id(p)],p.grad) for p in model.parameters() if id(p) in private_before)
    assert not keeper.active and not keeper.students and not keeper.teachers
    keeper.close()


def test_prefix_checkpoint_loading_inference_and_next_teacher(tmp_path,monkeypatch):
    from continual_lora import trainer
    monkeypatch.setattr(trainer,'enable_lora_training_compatibility',lambda:None)
    model,ctl,keeper,x,report=setup()
    with torch.no_grad():
        for _,m in keeper.shared:m.global_B.add_(.02*m.global_output_mask)
    ctl.end_task();model.eval();ctl.begin_anchor_batch();expected=model(x)
    payload=dict(config=dict(method='cqsm_dual_lora',rank=16,alpha=8,cqsm_schema_version=3,
                 cqsm_restore_legacy=True,cqsm_shared_layout='visual_prefix',global_rank=10,
                 cqsm_shared=True,cqsm_private=True,cqsm_bank_mode='all',cqsm_basis='covariance',
                 energy_calibration_lambda=3.,focus_orthogonality_weight=.001,
                 focus_group_ranks=report.group_ranks,stream=[{},{}]),
                 adapter=adapter_state_dict(model),controller=ctl.state_dict(),completed_task=0)
    path=tmp_path/'prefix.pt';torch.save(payload,path)
    torch.manual_seed(19)
    other=PrefixToy();oc,_,_=load_adapter_checkpoint(other,path)
    other.eval();oc.begin_anchor_batch()
    assert torch.equal(expected,other(x))
    for m in other.modules():
        if isinstance(m,CQSMDualLoRALinear) and m.shared_enabled:
            assert m.global_rank==10 and not m.private_enabled and not m.global_A.requires_grad
    next_keeper=SharedPrefixRetention(other);next_keeper.begin_task(1)
    assert len(next_keeper.previous)==2
    keeper.close();next_keeper.close()


@pytest.mark.parametrize('layout', ['visual_prefix', 'visual_residual'])
def test_private_conflict_projection_leaves_global_native_gradient_intact(layout):
    model,ctl,keeper,x,_=setup(layout=layout)
    ctl.begin_anchor_batch();y=model(x)
    functional=dict(semantic=y.square().mean(),box=y.mean(),mask=y.abs().mean())
    total=sum(functional.values())
    expected=torch.autograd.grad(total,keeper.params,retain_graph=True)
    _focus_backward(total,functional,ctl,True,strict_finite=True)
    assert all(torch.allclose(a,p.grad) for a,p in zip(expected,keeper.params))
    keeper.close()


def test_zero_retention_ratio_is_a_structural_control_without_teacher_forward():
    model,ctl,keeper,x,_=setup()
    keeper.close()
    control=SharedPrefixRetention(model,gradient_ratio=0.)
    control.begin_task(1);control.begin_batch();ctl.begin_anchor_batch();model(x)
    assert not control.active and not control.teachers and not control.students
    assert control.loss_and_gradients()==(None,None)
    assert control.apply(None)['shared_retention_applied']==0
    control.close()


@pytest.mark.parametrize('global_rank',[2,10])
def test_legacy_parallel_checkpoints_restore_the_recorded_shared_rank(tmp_path,monkeypatch,global_rank):
    from continual_lora import trainer
    monkeypatch.setattr(trainer,'enable_lora_training_compatibility',lambda:None)
    torch.manual_seed(13)
    model=SemanticFocusToy()
    report=inject_continual_lora(model,'cqsm_dual_lora',rank=16,num_tasks=2,cqsm_restore_legacy=True,
        cqsm_shared_layout='late_parallel',cqsm_global_rank=global_rank)
    ctl=AdapterController(model,'cqsm_dual_lora');ctl.begin_task(0)
    x=torch.randn(2,3,model.width)
    model.eval();ctl.anchor_router().inference_oracle_stage=0
    ctl.start_energy_covariance_collection();ctl.begin_anchor_batch()
    with torch.no_grad():model(x)
    ctl.finalize_energy_object_subspaces(0);ctl.anchor_router().inference_oracle_stage=None
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m,CQSMDualLoRALinear) and m.shared_enabled:m.global_B.normal_(std=.02)
    ctl.end_task();ctl.begin_anchor_batch();expected=model(x)
    # Old releases did not contain the layout field; loader must default to the legacy layout.
    cfg=dict(method='cqsm_dual_lora',rank=16,alpha=8,cqsm_schema_version=2,cqsm_restore_legacy=True,
             global_rank=global_rank,cqsm_shared=True,cqsm_private=True,cqsm_bank_mode='all',
             cqsm_basis='covariance',energy_calibration_lambda=3.,focus_orthogonality_weight=0.,
             focus_group_ranks=report.group_ranks,stream=[{},{}])
    path=tmp_path/f'legacy_{global_rank}.pt'
    torch.save(dict(config=cfg,adapter=adapter_state_dict(model),controller=ctl.state_dict(),completed_task=0),path)
    torch.manual_seed(13);other=SemanticFocusToy();oc,_,_=load_adapter_checkpoint(other,path)
    other.eval();oc.begin_anchor_batch()
    assert torch.equal(expected,other(x))
    assert all(m.global_rank==global_rank for m in other.modules()
               if isinstance(m,CQSMDualLoRALinear) and m.shared_enabled)
