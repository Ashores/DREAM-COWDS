import copy

import pytest
import torch
from torch import nn

from continual_lora.methods import (AdapterController, CapacityPartitionFocusLoRALinear,
                                    CompositionalPromptMLPTaskRouter,
                                    ConsensusComplementLoRALinear,
                                    FocusLoRALinear,
                                    ImportancePoolFocusLoRALinear,
                                    IndependentCapacityTaskLoRALinear,
                                    EnergyDecomposedObjectCore,
                                    EnergyObjectSubspaceLoRALinear,
                                    SemanticProtectedObjectCore,
                                    ObjectCoupledTaskLoRALinear,
                                    MultiAnchorTaskDecoupledLoRALinear,
                                    MultiAnchorTaskRouter,
                                    MLPTaskDecoupledLoRALinear,
                                    ObjectConsistentSkillRouter,
                                    PromptMLPTaskRouter,
                                    SAMCLSharedA,
                                    SAMCLSLoRALinear,
                                    SequentialLoRALinear,
                                    TaskObjectCore,
                                    inject_continual_lora)
from continual_lora.trainer import (_focus_mask_box_consistency,
                                    enable_lora_training_compatibility,
                                    _consensus_complement_backward,
                                    _object_bidirectional_box_mask_consistency,
                                    _object_consensus_backward,
                                    _project_functional_gradients,
                                    _rank_normalized_utility,
                                    _shared_core_residual_backward,
                                    _weighted_functional_losses,
                                    UnifiedObjectBridge)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(12, 20)

    def forward(self, x):
        return self.layer(x)


@pytest.mark.parametrize("method", ["sequential_lora", "lora_ewc", "inflora", "bilora", "rank1_expert", "rank1_expert_pm", "lwf_lora", "er_lora", "derpp_lora", "cumulative_lora"])
def test_methods_forward_backward_and_fixed_budget(method):
    torch.manual_seed(1)
    model = Toy()
    frozen = copy.deepcopy(model.layer.weight)
    report = inject_continual_lora(model, method, rank=10, alpha=5, num_tasks=5,
                                   target_patterns=(r"^layer$",))
    controller = AdapterController(model, method)
    count = sum(p.numel() for p in model.parameters())
    for task in range(5):
        controller.begin_task(task)
        x = torch.randn(3, 4, 12)
        loss = model(x).square().mean() + controller.regularization_loss()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.data.add_(p.grad, alpha=-1e-3); p.grad = None
        controller.end_task()
        assert sum(p.numel() for p in model.parameters()) == count
    assert model(torch.randn(2, 12)).shape == (2, 20)
    assert torch.equal(model.layer.base.weight, frozen)
    assert report.trainable_parameters > 0


def test_ewc_penalty_changes_after_update():
    model=Toy(); inject_continual_lora(model,"lora_ewc",rank=4,alpha=4,target_patterns=(r"^layer$",))
    ctl=AdapterController(model,"lora_ewc",ewc_lambda=1.0)
    fisher={n:torch.ones_like(p) for n,p in model.named_parameters() if p.requires_grad}
    ctl.consolidate_ewc(fisher)
    assert ctl.regularization_loss().item() == 0
    with torch.no_grad():
        next(p for p in model.parameters() if p.requires_grad).add_(0.1)
    assert ctl.regularization_loss().item() > 0


def test_parameter_matched_rank1_equals_standard_lora():
    a=Toy(); ra=inject_continual_lora(a,"sequential_lora",rank=10,alpha=5,
                                      target_patterns=(r"^layer$",))
    b=Toy(); rb=inject_continual_lora(b,"rank1_expert_pm",rank=10,alpha=5,
                                      target_patterns=(r"^layer$",))
    assert ra.trainable_parameters == rb.trainable_parameters


class _Attention(nn.Module):
    def __init__(self, width):
        super().__init__(); self.qkv=nn.Linear(width,width); self.proj=nn.Linear(width,width)


class _Mlp(nn.Module):
    def __init__(self, width):
        super().__init__(); self.fc1=nn.Linear(width,width); self.fc2=nn.Linear(width,width)


class _Block(nn.Module):
    def __init__(self, width):
        super().__init__(); self.attn=_Attention(width); self.mlp=_Mlp(width)


class _FFN(nn.Module):
    def __init__(self, width):
        super().__init__(); self.linear1=nn.Linear(width,width); self.linear2=nn.Linear(width,width)


class _MLP(nn.Module):
    def __init__(self, width, count):
        super().__init__(); self.layers=nn.ModuleList([nn.Linear(width,width) for _ in range(count)])


class _Language(nn.Module):
    def forward(self, x):
        memory=x.mean(dim=0,keepdim=True)
        return torch.zeros(1,1,dtype=torch.bool),memory,memory


class FocusToy(nn.Module):
    """Small module tree with the same functional names as SAM 3."""
    def __init__(self, width=12):
        super().__init__(); self.width=width
        self.backbone=nn.Module(); self.backbone.vision_backbone=nn.Module()
        self.backbone.language_backbone=_Language()
        self.backbone.vision_backbone.trunk=nn.Module()
        self.backbone.vision_backbone.trunk.blocks=nn.ModuleList([_Block(width) for _ in range(8)])
        self.transformer=nn.Module(); self.transformer.encoder=nn.Module()
        self.transformer.encoder.text_pooling_proj=nn.Linear(width,width)
        self.transformer.encoder.layers=nn.ModuleList([_FFN(width)])
        self.transformer.decoder=nn.Module(); self.transformer.decoder.layers=nn.ModuleList([_FFN(width)])
        self.transformer.decoder.presence_token_head=_MLP(width,3)
        self.transformer.decoder.bbox_embed=_MLP(width,3)
        self.transformer.decoder.ref_point_head=_MLP(width,2)
        self.segmentation_head=nn.Module(); self.segmentation_head.mask_predictor=nn.Module()
        self.segmentation_head.mask_predictor.mask_embed=_MLP(width,3)

    def forward(self, x):
        values=[]
        for block in self.backbone.vision_backbone.trunk.blocks:
            values.extend((block.attn.qkv(x),block.attn.proj(x),
                           block.mlp.fc1(x),block.mlp.fc2(x)))
        self.backbone.language_backbone(x)
        for layer in (*self.transformer.encoder.layers,*self.transformer.decoder.layers):
            values.extend((layer.linear1(x),layer.linear2(x)))
        values.extend(layer(x) for layer in self.transformer.decoder.presence_token_head.layers[:2])
        values.extend(layer(x) for layer in self.transformer.decoder.bbox_embed.layers[:2])
        values.extend(layer(x) for layer in self.transformer.decoder.ref_point_head.layers[:2])
        values.extend(layer(x) for layer in self.segmentation_head.mask_predictor.mask_embed.layers)
        return torch.stack(values).mean(0)


class SemanticFocusToy(FocusToy):
    def __init__(self, width=12):
        super().__init__(width)
        self.dot_prod_scoring=nn.Module()
        self.dot_prod_scoring.hs_proj=nn.Linear(width,width)
        self.dot_prod_scoring.prompt_proj=nn.Linear(width,width)

    def forward(self,x):
        base=super().forward(x)
        query=self.dot_prod_scoring.hs_proj(x)
        text=self.dot_prod_scoring.prompt_proj(x)
        return torch.stack((base,query,text)).mean(0)


def test_focus_is_function_not_task_partitioned_and_fixed_budget():
    torch.manual_seed(3)
    baseline=FocusToy()
    baseline_report=inject_continual_lora(baseline,"sequential_lora",rank=10,alpha=5)
    model=FocusToy()
    report=inject_continual_lora(model,"focus_lora",rank=10,alpha=5,num_tasks=5)
    assert set(report.module_groups) == {"visual","semantic","box","mask"}
    assert report.persistent_adapter_parameters <= baseline_report.persistent_adapter_parameters
    assert all("task" not in key.lower() and "dataset" not in key.lower()
               for key in model.state_dict())
    count=sum(p.numel() for p in model.parameters())
    controller=AdapterController(model,"focus_lora")
    for stage in range(3):
        controller.begin_task(stage)
        loss=model(torch.randn(2,12)).square().mean()+controller.regularization_loss()
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    parameter.add_(parameter.grad,alpha=-1e-2); parameter.grad=None
        diagnostics=controller.end_task()
        assert diagnostics["layers"] == len(report.modules)
        assert 0 <= diagnostics["mean_relative_compression_error"] <= 1
        assert sum(p.numel() for p in model.parameters()) == count
    adapters=[m for m in model.modules() if isinstance(m,FocusLoRALinear)]
    assert all(int(module.consolidations)==3 for module in adapters)
    assert all(torch.count_nonzero(module.plastic_B)==0 for module in adapters)


def test_functional_loss_reconstruction_and_pcgrad():
    core=torch.tensor(59.0,requires_grad=True)
    loss_map={
        "loss_ce":torch.tensor(1.0,requires_grad=True),
        "presence_loss":torch.tensor(0.5,requires_grad=True),
        "loss_bbox":torch.tensor(1.0,requires_grad=True),
        "loss_giou":torch.tensor(2.0,requires_grad=True),
        "loss_mask":torch.tensor(0.1,requires_grad=True),
        "core_loss":core,
    }
    groups=_weighted_functional_losses(loss_map,core)
    assert groups["semantic"].item()==30
    assert groups["box"].item()==9
    assert groups["mask"].item()==20
    assert groups["reconstruction_gap"].item()==0
    gradients=[(torch.tensor([1.0,0.0]),),(torch.tensor([-1.0,1.0]),),
               (torch.tensor([0.0,1.0]),)]
    projected,_,cosines=_project_functional_gradients(gradients)
    assert cosines["semantic_box"] < 0
    assert torch.isfinite(projected[0]).all()


def test_mask_box_consistency_uses_only_prediction_geometry():
    class Output: pass
    output=Output()
    prediction={
        "pred_boxes":torch.tensor([[[0.5,0.5,0.5,0.5]]],requires_grad=True),
        "pred_masks":torch.zeros(1,1,8,8,requires_grad=True),
        "pred_logits":torch.ones(1,1,1,requires_grad=True),
    }
    output.output=[[prediction]]
    value=_focus_mask_box_consistency(output)
    assert value.ndim==0 and torch.isfinite(value) and value.item()>0
    value.backward()
    assert prediction["pred_boxes"].grad is not None
    assert prediction["pred_masks"].grad is not None


def test_focus_keeps_fp32_master_factors_with_bfloat16_model():
    model=FocusToy().to(torch.bfloat16)
    inject_continual_lora(model,"focus_lora",rank=10,alpha=5)
    adapters=[module for module in model.modules() if isinstance(module,FocusLoRALinear)]
    assert adapters and all(module.stable_A.dtype==torch.float32 for module in adapters)
    output=model(torch.randn(2,12,dtype=torch.bfloat16))
    assert output.dtype==torch.bfloat16
    output.float().square().mean().backward()
    assert all(module.plastic_A.grad is not None and module.plastic_A.grad.dtype==torch.float32
               for module in adapters)


def test_focus_v2_preserves_allocated_atoms_without_compression_or_task_keys():
    torch.manual_seed(7)
    baseline=FocusToy()
    baseline_report=inject_continual_lora(baseline,"sequential_lora",rank=10,alpha=5)
    model=FocusToy()
    report=inject_continual_lora(model,"focus_lora_v2",rank=10,alpha=5,num_tasks=5,
                                 focus_admission_ratio=0.25)
    assert report.persistent_adapter_parameters <= baseline_report.persistent_adapter_parameters
    assert all("task" not in key.lower() and "dataset" not in key.lower()
               for key in model.state_dict())
    adapters=[m for m in model.modules() if isinstance(m,ImportancePoolFocusLoRALinear)]
    assert adapters and all(module.lora_A.dtype==torch.float32 for module in adapters)
    parameter_count=sum(p.numel() for p in model.parameters())
    controller=AdapterController(model,"focus_lora_v2")
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=1e-2,weight_decay=0.1)

    for stage in range(3):
        controller.begin_task(stage)
        controller.reset_focus_active_optimizer_state(optimizer)
        protected_before=[(module.lora_A.detach().clone(),module.lora_B.detach().clone(),
                           module._protected_mask().clone()) for module in adapters]
        optimizer.zero_grad(set_to_none=True)
        loss=model(torch.randn(2,12)).square().mean()+controller.regularization_loss()
        loss.backward()
        controller.accumulate_focus_importance()
        optimizer.step()
        controller.restore_focus_protected_parameters()
        for module,(old_a,old_b,protected) in zip(adapters,protected_before):
            assert torch.equal(module.lora_A[protected],old_a[protected])
            assert torch.equal(module.lora_B[:,protected],old_b[:,protected])
        diagnostics=controller.end_task()
        assert diagnostics["mean_relative_compression_error"] == 0
        assert diagnostics["mean_allocated_rank"] > 0
        assert sum(p.numel() for p in model.parameters()) == parameter_count

    assert all(int(module.consolidations)==3 for module in adapters)
    assert all(torch.count_nonzero(module.allocated_mask)>0 for module in adapters)
    assert all(torch.count_nonzero(module.lora_B[:,module.allocated_mask])>0
               for module in adapters)


def test_focus_v2_recycles_only_low_importance_atoms_when_full():
    base=nn.Linear(6,6)
    module=ImportancePoolFocusLoRALinear(base,rank=4,alpha=4,num_tasks=8,
                                         function_group="visual",admission_ratio=0.5)
    module.begin_task(0); module.end_task()
    module.begin_task(1); module.end_task()
    assert module.allocated_mask.all()
    module.importance.copy_(torch.tensor([0.8,0.1,0.9,0.2]))
    module.begin_task(2)
    assert torch.equal(torch.where(module.active_mask)[0],torch.tensor([1,3]))


@pytest.mark.parametrize(("rank","expected"),[
    (16,[3,3,4,3,3]),
    (15,[3,3,3,3,3]),
    (8,[2,1,2,1,2]),
])
def test_focus_v3_balanced_capacity_is_disjoint_and_exhaustive(rank,expected):
    module=CapacityPartitionFocusLoRALinear(nn.Linear(20,20),rank=rank,alpha=8,
                                            num_tasks=5,function_group="visual")
    seen=torch.zeros(rank,dtype=torch.bool)
    for stage,count in enumerate(expected):
        module.begin_task(stage)
        assert int(module.active_mask.sum())==count
        assert not bool((seen & module.active_mask).any())
        seen |= module.active_mask
        report=module.end_task()
        assert report["recycled_rank"]==0
        assert report["relative_compression_error"]==0
    assert seen.all() and module.allocated_mask.all()


def test_focus_v3_protects_old_capacity_and_forward_uses_no_task_identity():
    torch.manual_seed(11)
    module=CapacityPartitionFocusLoRALinear(nn.Linear(12,20),rank=10,alpha=5,
                                            num_tasks=5,function_group="semantic",
                                            top_k=4)
    optimizer=torch.optim.AdamW(module.adapter_parameters(),lr=1e-2,weight_decay=0.1)
    module.begin_task(0)
    first_slots=module.active_mask.clone()
    loss=module(torch.randn(4,12)).square().mean()
    loss.backward(); optimizer.step(); module.restore_protected_parameters(); module.end_task()
    old_a=module.lora_A.detach().clone(); old_b=module.lora_B.detach().clone()

    module.begin_task(1); module.reset_active_optimizer_state(optimizer)
    optimizer.zero_grad(set_to_none=True)
    loss=module(torch.randn(4,12)).square().mean()+module.regularization_loss()
    loss.backward(); optimizer.step(); module.restore_protected_parameters()
    assert torch.equal(module.lora_A[first_slots],old_a[first_slots])
    assert torch.equal(module.lora_B[:,first_slots],old_b[:,first_slots])
    x=torch.randn(3,12)
    before=module(x)
    module.task_id=999
    after=module(x)
    assert torch.equal(before,after)
    assert all("task" not in key.lower() and "dataset" not in key.lower()
               for key in module.state_dict())


def test_focus_v3_real_function_groups_match_standard_lora_budget():
    baseline=FocusToy()
    baseline_report=inject_continual_lora(baseline,"sequential_lora",rank=16,alpha=8)
    model=FocusToy()
    report=inject_continual_lora(model,"focus_lora_v3",rank=16,alpha=8,num_tasks=5)
    assert set(report.module_groups)=={"visual","semantic","box","mask"}
    assert report.persistent_adapter_parameters<=baseline_report.persistent_adapter_parameters
    adapters=[m for m in model.modules() if isinstance(m,CapacityPartitionFocusLoRALinear)]
    assert adapters and all(module.rank>=5 for module in adapters)


def test_focus_v3_capacity_routing_modes_are_finite_and_task_id_free():
    torch.manual_seed(19)
    module=CapacityPartitionFocusLoRALinear(nn.Linear(12,20),rank=10,alpha=5,
                                            num_tasks=5,function_group="visual",top_k=4)
    for stage in range(5):
        module.begin_task(stage)
        with torch.no_grad():
            chosen=module.lora_B[:,module.active_mask]
            module.lora_B[:,module.active_mask]=torch.randn_like(chosen)*0.05
        module.end_task()
    x=torch.randn(2,3,12)
    outputs={}
    for mode,kwargs in (
        ("atom_topk",{}),
        ("block_energy",{"blocks":1}),
        ("block_energy",{"blocks":2}),
        ("oracle_block",{"oracle_stage":1}),
    ):
        module.set_routing(mode,**kwargs)
        outputs[(mode,tuple(kwargs.items()))]=module(x)
        assert torch.isfinite(outputs[(mode,tuple(kwargs.items()))]).all()
    module.set_routing("block_energy",blocks=1)
    before=module(x)
    module.task_id=12345
    assert torch.equal(before,module(x))
    assert not torch.equal(outputs[("oracle_block",(("oracle_stage",1),))],
                           outputs[("atom_topk",())])


def test_focus_v3_oracle_routing_requires_valid_stage():
    module=CapacityPartitionFocusLoRALinear(nn.Linear(8,8),rank=5,alpha=5,
                                            num_tasks=5,function_group="box")
    with pytest.raises(ValueError): module.set_routing("oracle_block")
    with pytest.raises(ValueError): module.set_routing("oracle_block",oracle_stage=5)


def test_task_decoupled_anchor_router_accumulates_and_routes_without_task_id():
    router=MultiAnchorTaskRouter(feature_dim=4,num_tasks=2)
    router.train(); router.begin_task(0)
    router.prepare(torch.tensor([[1.,0.,0.,0.],[.9,.1,0.,0.]]),training=True)
    router.begin_task(1)
    router.prepare(torch.tensor([[0.,1.,0.,0.],[.1,.9,0.,0.]]),training=True)
    router.eval(); router.prepare(torch.tensor([[1.,0.,0.,0.]]),training=False)
    route=router.route_for(torch.randn(1,3,5))
    assert route.shape==(1,1,2)
    assert route[0,0,0]>route[0,0,1]
    assert torch.allclose(route.sum(-1),torch.ones(1,1))


def test_task_decoupled_uses_one_shared_route_and_fixed_lora_budget():
    torch.manual_seed(23)
    baseline=FocusToy()
    baseline_report=inject_continual_lora(baseline,"sequential_lora",rank=16,alpha=8)
    model=FocusToy()
    report=inject_continual_lora(model,"task_decoupled_lora",rank=16,alpha=8,
                                 num_tasks=5)
    controller=AdapterController(model,"task_decoupled_lora")
    adapters=[m for m in model.modules()
              if isinstance(m,MultiAnchorTaskDecoupledLoRALinear)]
    routers=[m for m in model.modules() if isinstance(m,MultiAnchorTaskRouter)]
    assert len(routers)==1 and adapters
    assert all(m._anchor_router is routers[0] for m in adapters)
    assert report.persistent_adapter_parameters<=baseline_report.persistent_adapter_parameters
    parameter_count=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-2)
    for stage in range(2):
        controller.begin_task(stage)
        optimizer.zero_grad(set_to_none=True)
        model.train(); model(torch.randn(2,12)).square().mean().backward(); optimizer.step()
        controller.restore_focus_protected_parameters(); controller.end_task()
    assert sum(p.numel() for p in model.parameters())==parameter_count
    assert routers[0].anchor_count.tolist()==[1.,1.,0.,0.,0.]
    model.eval()
    x=torch.randn(1,12)
    output=model(x)
    assert torch.isfinite(output).all()
    before=output.detach().clone(); adapters[0].task_id=999
    assert torch.equal(before,model(x))


def test_prompt_mlp_router_learns_with_embedding_rehearsal_and_one_route():
    torch.manual_seed(29)
    router=PromptMLPTaskRouter(visual_dim=8,num_tasks=2,embeddings_per_task=16)
    for stage,center in enumerate((
        torch.tensor([1.,0,0,0,0,0,0,0.]),
        torch.tensor([0.,1,0,0,0,0,0,0.]),
    )):
        router.train(); router.begin_task(stage)
        for _ in range(4):
            visual=center.unsqueeze(0).repeat(4,1)+torch.randn(4,8)*0.01
            router.begin_batch(); router.prepare_visual(visual,training=True)
        router.fit_selector(epochs=80,lr=5e-2,batch_size=8)
    router.eval(); router.begin_batch()
    router.prepare_visual(torch.tensor([[1.,0,0,0,0,0,0,0.]]),training=False)
    assert router._route.argmax(-1).item()==0
    route_before=router._route.clone()
    router.prepare_text(torch.randn(3,5),training=False)
    assert torch.equal(route_before,router._route)
    router.begin_batch()
    router.prepare_visual(torch.tensor([[0.,1,0,0,0,0,0,0.]]),training=False)
    assert router._route.argmax(-1).item()==1
    assert router.embedding_count.tolist()==[16,16]


def test_prompt_mlp_router_oracle_is_evaluation_only():
    router=PromptMLPTaskRouter(visual_dim=8,num_tasks=2,embeddings_per_task=4)
    router.eval(); router.inference_oracle_stage=1
    router.prepare_visual(torch.randn(3,8),training=False)
    assert torch.equal(router._route,torch.tensor([[0.,1.]]).repeat(3,1))
    router.begin_batch(); router.train(); router.begin_task(0)
    router.prepare_visual(torch.randn(3,8),training=True)
    assert torch.equal(router._route,torch.tensor([[1.,0.]]).repeat(3,1))


def test_task_decoupled_mlp_fits_budget_and_uses_one_mid_visual_route():
    baseline=FocusToy()
    baseline_report=inject_continual_lora(baseline,"sequential_lora",rank=16,alpha=8)
    model=FocusToy()
    report=inject_continual_lora(model,"task_decoupled_mlp",rank=16,alpha=8,num_tasks=5)
    assert report.persistent_adapter_parameters<=baseline_report.persistent_adapter_parameters
    assert "selector" in report.module_groups
    adapters=[m for m in model.modules() if isinstance(m,MLPTaskDecoupledLoRALinear)]
    routers=[m for m in model.modules() if isinstance(m,PromptMLPTaskRouter)]
    assert adapters and len(routers)==1
    assert routers[0].routing_block==4
    assert all(int(name.split(".blocks.")[1].split(".")[0])>=4
               for name in report.module_groups["visual"])
    controller=AdapterController(model,"task_decoupled_mlp")
    controller.begin_task(0); controller.begin_anchor_batch(); model.train()
    output=model(torch.randn(2,12))
    loss=output.square().mean()+controller.selector_loss()
    loss.backward()
    assert all(p.grad is None for p in routers[0].selector.parameters())
    assert routers[0].embedding_count.tolist()==[2,0,0,0,0]
    route=routers[0]._route.clone()
    assert all(torch.equal(route,adapter._anchor_router._route) for adapter in adapters)


def test_samcl_slora_shares_a_across_layers_and_protects_old_modules():
    torch.manual_seed(19)
    model=FocusToy()
    report=inject_continual_lora(model,"samcl_slora",rank=3,alpha=3,num_tasks=2)
    adapters=[m for m in model.modules() if isinstance(m,SAMCLSLoRALinear)]
    banks=[m for m in model.modules() if isinstance(m,SAMCLSharedA)]
    routers=[m for m in model.modules() if isinstance(m,PromptMLPTaskRouter)]
    assert adapters and len(banks)==1 and len(routers)==1
    assert all(adapter._shared_a is banks[0] for adapter in adapters)
    assert report.group_ranks=={"visual":3}
    assert routers[0].routing_block==4

    controller=AdapterController(model,"samcl_slora")
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad and
         p not in set(routers[0].selector.parameters())],lr=1e-2,weight_decay=.1)
    controller.begin_task(0); controller.begin_anchor_batch(); model.train()
    model(torch.randn(2,12)).square().mean().backward()
    optimizer.step(); controller.restore_focus_protected_parameters(); optimizer.zero_grad()
    controller.end_task()
    old_a=banks[0].lora_A[0].detach().clone()
    old_b=[adapter.lora_B[0].detach().clone() for adapter in adapters]

    controller.begin_task(1); controller.begin_anchor_batch()
    model(torch.randn(2,12)).square().mean().backward()
    optimizer.step(); controller.restore_focus_protected_parameters()
    assert torch.equal(banks[0].lora_A[0],old_a)
    assert all(torch.equal(adapter.lora_B[0],value)
               for adapter,value in zip(adapters,old_b))
    assert routers[0].embedding_count.tolist()==[2,2]


def test_independent_capacity_gives_every_task_a_full_function_specific_module():
    torch.manual_seed(23)
    model=FocusToy()
    report=inject_continual_lora(
        model,"independent_capacity_lora",rank=16,alpha=8,num_tasks=2)
    adapters=[m for m in model.modules()
              if isinstance(m,IndependentCapacityTaskLoRALinear)]
    routers=[m for m in model.modules()
             if isinstance(m,CompositionalPromptMLPTaskRouter)]
    assert adapters and len(routers)==1
    assert report.group_ranks=={"visual":5,"semantic":4,"box":3,"mask":3}
    for adapter in adapters:
        assert adapter.lora_A.shape[:2]==(2,report.group_ranks[adapter.function_group])
        assert adapter.lora_B.shape[0]==2
        assert adapter._anchor_router is routers[0]

    controller=AdapterController(model,"independent_capacity_lora")
    selector_ids={id(p) for p in routers[0].selector.parameters()}
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad and id(p) not in selector_ids],
        lr=1e-2,weight_decay=.1)
    controller.begin_task(0); controller.begin_anchor_batch(); model.train()
    model(torch.randn(2,12)).square().mean().backward()
    optimizer.step(); controller.restore_focus_protected_parameters(); optimizer.zero_grad()
    controller.end_task()
    old=[(adapter.lora_A[0].detach().clone(),adapter.lora_B[0].detach().clone())
         for adapter in adapters]

    controller.begin_task(1); controller.begin_anchor_batch()
    model(torch.randn(2,12)).square().mean().backward()
    optimizer.step(); controller.restore_focus_protected_parameters()
    assert all(torch.equal(adapter.lora_A[0],a) and torch.equal(adapter.lora_B[0],b)
               for adapter,(a,b) in zip(adapters,old))
    assert all(adapter.allocated_mask.tolist()==[True,True] for adapter in adapters)
    assert routers[0].embedding_count.tolist()==[2,2]


def test_object_coupled_capacity_shares_one_query_core_and_protects_history():
    torch.manual_seed(37)
    model=FocusToy()
    report=inject_continual_lora(
        model,"object_coupled_capacity_lora",rank=16,alpha=8,num_tasks=2)
    adapters=[m for m in model.modules()
              if isinstance(m,ObjectCoupledTaskLoRALinear)]
    cores=[m for m in model.modules() if isinstance(m,TaskObjectCore)]
    routers=[m for m in model.modules()
             if isinstance(m,CompositionalPromptMLPTaskRouter)]
    coupled=[m for m in adapters if m.uses_object_core]
    assert adapters and coupled and len(cores)==1 and len(routers)==1
    assert all(m._object_core is cores[0] for m in adapters)
    assert report.group_ranks=={"visual":5,"semantic":4,"box":3,"mask":3}
    assert any("decoder.layers.0.linear1" in name
               for name in report.module_groups["object_core"])
    assert any("bbox_embed.layers.0" in name
               for name in report.module_groups["object_core"])
    assert any("mask_embed.layers.0" in name
               for name in report.module_groups["object_core"])
    assert all(m.rank+cores[0].rank==report.group_ranks[m.function_group]
               for m in coupled)

    controller=AdapterController(model,"object_coupled_capacity_lora")
    selector_ids={id(p) for p in routers[0].selector.parameters()}
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad and id(p) not in selector_ids],
        lr=1e-2,weight_decay=.1)
    controller.begin_task(0); controller.begin_anchor_batch(); model.train()
    model(torch.randn(2,12)).square().mean().backward()
    assert any(m.core_B.grad is not None and torch.count_nonzero(m.core_B.grad)>0
               for m in coupled)
    optimizer.step(); controller.restore_focus_protected_parameters(); optimizer.zero_grad()
    controller.end_task()
    old_core=cores[0].lora_A[0].detach().clone()
    old_b=[m.core_B[0].detach().clone() for m in coupled]

    controller.begin_task(1); controller.begin_anchor_batch()
    model(torch.randn(2,12)).square().mean().backward()
    optimizer.step(); controller.restore_focus_protected_parameters()
    assert torch.equal(cores[0].lora_A[0],old_core)
    assert all(torch.equal(m.core_B[0],value) for m,value in zip(coupled,old_b))
    assert routers[0].embedding_count.tolist()==[2,2]


def test_energy_object_subspace_is_data_driven_fixed_and_task_free_routed():
    torch.manual_seed(41)
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_object_subspace_lora",rank=16,alpha=8,num_tasks=2)
    core=next(m for m in model.modules() if isinstance(m,EnergyDecomposedObjectCore))
    adapters=[m for m in model.modules()
              if isinstance(m,EnergyObjectSubspaceLoRALinear)]
    router=next(m for m in model.modules() if type(m) is PromptMLPTaskRouter)
    controller=AdapterController(model,"energy_object_subspace_lora")
    assert report.method=="energy_object_subspace_lora"
    assert core.general_rank==core.isolated_rank==1
    assert not core.lora_A.requires_grad
    assert not hasattr(router,"composition_logits")

    controller.begin_task(0)
    controller.start_energy_covariance_collection()
    model.train(); controller.begin_anchor_batch()
    model(torch.randn(3,12))
    diagnostics=controller.finalize_energy_object_subspaces(0)
    assert diagnostics["collection_consumer_calls"] > 0
    assert core.subspace_ready.tolist()==[True,False]
    gram=core.lora_A[0].float() @ core.lora_A[0].float().T
    assert torch.allclose(gram,torch.eye(2),atol=1e-4)

    controller.begin_anchor_batch()
    model(torch.randn(3,12)).square().mean().backward()
    coupled=[m for m in adapters if m.uses_object_core]
    assert any(m.core_B.grad is not None and torch.count_nonzero(m.core_B.grad)>0
               for m in coupled)
    controller.end_task()
    assert float(core.historical_covariance.trace()) > 0
    assert torch.isfinite(core.general_scale[0]).all()

    historical_A=core.lora_A[0].clone()
    controller.begin_task(1)
    controller.start_energy_covariance_collection()
    controller.begin_anchor_batch(); model(torch.randn(3,12)+2.0)
    controller.finalize_energy_object_subspaces(1)
    assert torch.equal(core.lora_A[0],historical_A)
    assert core.subspace_ready.tolist()==[True,True]


def test_energy_object_compositional_uses_independent_router():
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_object_compositional_lora",rank=16,alpha=8,num_tasks=2)
    router=next(m for m in model.modules()
                if isinstance(m,CompositionalPromptMLPTaskRouter))
    assert report.method=="energy_object_compositional_lora"
    assert hasattr(router,"composition_logits")
    assert any(isinstance(m,EnergyDecomposedObjectCore) for m in model.modules())


def test_energy_visual_mlp_adapts_late_vit_mlp():
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_visual_mlp_lora",rank=16,alpha=8,num_tasks=2)
    visual=report.module_groups["visual"]
    core=next(m for m in model.modules() if isinstance(m,EnergyDecomposedObjectCore))
    assert report.method=="energy_visual_mlp_lora"
    assert core.general_rank==2 and core.isolated_rank==0
    assert any(name.endswith("mlp.fc1") for name in visual)
    assert any(name.endswith("mlp.fc2") for name in visual)
    assert any(name.endswith("attn.qkv") for name in visual)
    mlp_blocks={int(name.split(".blocks.")[1].split(".")[0])
                for name in visual if ".mlp." in name}
    attn_blocks={int(name.split(".blocks.")[1].split(".")[0])
                 for name in visual if ".attn." in name}
    assert mlp_blocks==attn_blocks


def test_energy_visual_mlp_can_reload_split_core_from_config_rank():
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_visual_mlp_lora",rank=16,alpha=8,num_tasks=2,
        general_object_rank=1)
    core=next(m for m in model.modules() if isinstance(m,EnergyDecomposedObjectCore))
    assert report.method=="energy_visual_mlp_lora"
    assert core.general_rank==1 and core.isolated_rank==1


def test_energy_visual_mlp_rank8_shrinks_object_core():
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_visual_mlp_lora",rank=8,alpha=8,num_tasks=2)
    core=next(m for m in model.modules() if isinstance(m,EnergyDecomposedObjectCore))
    assert report.method=="energy_visual_mlp_lora"
    assert core.rank==1 and core.general_rank==1 and core.isolated_rank==0
    assert report.group_ranks["semantic"] > core.rank
    assert report.group_ranks["box"] > core.rank
    assert report.group_ranks["mask"] > core.rank


def test_independent_visual_mlp_adapts_late_vit_mlp():
    model=FocusToy()
    report=inject_continual_lora(
        model,"independent_visual_mlp_lora",rank=16,alpha=8,num_tasks=2)
    visual=report.module_groups["visual"]
    assert report.method=="independent_visual_mlp_lora"
    assert any(isinstance(m,CompositionalPromptMLPTaskRouter) for m in model.modules())
    assert not any(isinstance(m,EnergyDecomposedObjectCore) for m in model.modules())
    assert any(name.endswith("mlp.fc1") for name in visual)
    assert any(name.endswith("mlp.fc2") for name in visual)
    mlp_blocks={int(name.split(".blocks.")[1].split(".")[0])
                for name in visual if ".mlp." in name}
    attn_blocks={int(name.split(".blocks.")[1].split(".")[0])
                 for name in visual if ".attn." in name}
    assert mlp_blocks==attn_blocks


def test_energy_visual_mlp_general_only_has_no_isolated_rank():
    torch.manual_seed(7)
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_visual_mlp_general_only",rank=16,alpha=8,num_tasks=2)
    core=next(m for m in model.modules() if isinstance(m,EnergyDecomposedObjectCore))
    controller=AdapterController(model,"energy_visual_mlp_general_only")
    assert report.method=="energy_visual_mlp_general_only"
    assert core.general_rank==2 and core.isolated_rank==0
    controller.begin_task(0)
    controller.start_energy_covariance_collection()
    model.train(); controller.begin_anchor_batch()
    model(torch.randn(3,12))
    diagnostics=controller.finalize_energy_object_subspaces(0)
    assert diagnostics["isolated_energy_ratio"]==0.0
    assert core.subspace_ready[0]
    gram=core.lora_A[0].float() @ core.lora_A[0].float().T
    assert torch.allclose(gram,torch.eye(2),atol=1e-4)
    controller.begin_anchor_batch()
    loss=model(torch.randn(3,12)).square().mean()
    loss.backward()
    controller.end_task()
    assert float(core.historical_covariance.trace())>0


def test_energy_visual_mlp_independent_core_separates_query_facing_bases():
    torch.manual_seed(17)
    model=SemanticFocusToy()
    report=inject_continual_lora(
        model,"energy_visual_mlp_independent_core",rank=16,alpha=8,num_tasks=2)
    controller=AdapterController(model,"energy_visual_mlp_independent_core")
    cores=controller.energy_object_cores()
    coupled=[module for module in controller.adapters()
             if isinstance(module,EnergyObjectSubspaceLoRALinear)
             and module.uses_object_core]
    assert report.method=="energy_visual_mlp_independent_core"
    assert len(cores)==4
    assert len({id(module._object_core) for module in coupled})==4
    assert all(core.general_rank==2 and core.isolated_rank==0 for core in cores)
    assert sum(int(module._is_object_core_source) for module in coupled)==4

    controller.begin_task(0)
    controller.start_energy_covariance_collection()
    model.train(); controller.begin_anchor_batch()
    model(torch.randn(3,12))
    diagnostics=controller.finalize_energy_object_subspaces(0)
    assert diagnostics["object_core_count"]==4
    assert len(diagnostics["per_core"])==4
    assert all(bool(core.subspace_ready[0]) for core in cores)


def test_energy_visual_mlp_isolated_only_has_no_general_rank():
    torch.manual_seed(8)
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_visual_mlp_isolated_only",rank=16,alpha=8,num_tasks=2)
    core=next(m for m in model.modules() if isinstance(m,EnergyDecomposedObjectCore))
    controller=AdapterController(model,"energy_visual_mlp_isolated_only")
    assert report.method=="energy_visual_mlp_isolated_only"
    assert core.general_rank==0 and core.isolated_rank==2
    controller.begin_task(0)
    controller.start_energy_covariance_collection()
    model.train(); controller.begin_anchor_batch()
    model(torch.randn(3,12))
    diagnostics=controller.finalize_energy_object_subspaces(0)
    assert diagnostics["general_energy"]==0.0
    assert core.subspace_ready[0]
    controller.begin_anchor_batch()
    loss=model(torch.randn(3,12)).square().mean()
    loss.backward()
    controller.end_task()
    assert core.general_scale.shape[-1]==0


def test_energy_visual_mlp_encoder_skips_decoder_and_object_core():
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_visual_mlp_encoder",rank=16,alpha=8,num_tasks=2)
    visual=report.module_groups["visual"]
    assert report.method=="energy_visual_mlp_encoder"
    assert any(name.endswith("mlp.fc1") for name in visual)
    assert any(name.endswith("attn.qkv") for name in visual)
    assert not report.module_groups["semantic"]
    assert not report.module_groups["box"]
    assert not report.module_groups["mask"]
    assert not report.module_groups.get("object_core")
    assert not any(isinstance(m,EnergyDecomposedObjectCore) for m in model.modules())
    assert isinstance(model.transformer.decoder.bbox_embed.layers[0], nn.Linear)


def test_energy_visual_mlp_mlp_only_skips_attn():
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_visual_mlp_mlp_only",rank=16,alpha=8,num_tasks=2)
    visual=report.module_groups["visual"]
    core=next(m for m in model.modules() if isinstance(m,EnergyDecomposedObjectCore))
    assert report.method=="energy_visual_mlp_mlp_only"
    assert all(".mlp." in name for name in visual)
    assert not any(".attn." in name for name in visual)
    assert min(int(name.split(".blocks.")[1].split(".")[0]) for name in visual) >= 4
    assert core.general_rank==2 and core.isolated_rank==0
    assert isinstance(model.backbone.vision_backbone.trunk.blocks[4].attn.qkv, nn.Linear)


def test_energy_visual_mlp_early_mlp_uses_pre_cut_blocks():
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_visual_mlp_early_mlp",rank=16,alpha=8,num_tasks=2)
    visual=report.module_groups["visual"]
    assert report.method=="energy_visual_mlp_early_mlp"
    assert all(".mlp." in name for name in visual)
    assert max(int(name.split(".blocks.")[1].split(".")[0]) for name in visual) < 4
    assert isinstance(model.backbone.vision_backbone.trunk.blocks[4].attn.qkv, nn.Linear)
    assert isinstance(model.backbone.vision_backbone.trunk.blocks[4].mlp.fc1, nn.Linear)


def test_energy_visual_mlp_random_core_ignores_covariance():
    torch.manual_seed(5)
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_visual_mlp_random_core",rank=16,alpha=8,num_tasks=2)
    core=next(m for m in model.modules() if isinstance(m,EnergyDecomposedObjectCore))
    controller=AdapterController(model,"energy_visual_mlp_random_core")
    assert report.method=="energy_visual_mlp_random_core"
    assert core.random_directions
    assert core.general_rank==2 and core.isolated_rank==0
    controller.begin_task(0)
    controller.start_energy_covariance_collection()
    model.train(); controller.begin_anchor_batch()
    model(torch.randn(3,12))
    diagnostics=controller.finalize_energy_object_subspaces(0)
    assert diagnostics["random_directions"]==1
    gram=core.lora_A[0].float() @ core.lora_A[0].float().T
    assert torch.allclose(gram,torch.eye(2),atol=1e-4)


def test_energy_visual_mlp_decoder_skips_vit_and_keeps_object_core():
    model=FocusToy()
    report=inject_continual_lora(
        model,"energy_visual_mlp_decoder",rank=16,alpha=8,num_tasks=2)
    core=next(m for m in model.modules() if isinstance(m,EnergyDecomposedObjectCore))
    controller=AdapterController(model,"energy_visual_mlp_decoder")
    assert report.method=="energy_visual_mlp_decoder"
    assert not report.module_groups["visual"]
    assert report.module_groups["semantic"]
    assert report.module_groups["box"]
    assert report.module_groups["mask"]
    assert report.module_groups["object_core"]
    assert core.general_rank==2 and core.isolated_rank==0
    assert controller.anchor_router() is not None
    assert isinstance(
        model.backbone.vision_backbone.trunk.blocks[4].attn.qkv, nn.Linear)


def test_visual_mlp_eval_routes_fused_addmm_through_adapter_forward():
    """SAM 3 fused addmm_act reads linear.bias; LoRA wrappers have none."""
    import sam3.model.vitdet as vitdet
    from sam3.perflib import fused

    adapter = SequentialLoRALinear(nn.Linear(8, 16), rank=2, alpha=4)
    x = torch.randn(2, 8)
    with torch.no_grad():
        with pytest.raises(AttributeError, match="bias"):
            fused.addmm_act(nn.GELU, adapter, x)
        enable_lora_training_compatibility()
        y = vitdet.addmm_act(nn.GELU, adapter, x)
    assert y.shape == (2, 16)
    assert torch.isfinite(y).all()


def test_semantic_protected_subspace_collects_all_heads_and_frozen_text():
    torch.manual_seed(43)
    model=SemanticFocusToy()
    report=inject_continual_lora(
        model,"semantic_protected_subspace_lora",rank=16,alpha=8,num_tasks=2)
    core=next(m for m in model.modules()
              if isinstance(m,SemanticProtectedObjectCore))
    controller=AdapterController(model,"semantic_protected_subspace_lora")
    assert report.group_ranks=={"visual":4,"semantic":6,"box":5,"mask":5}
    assert core.rank==4 and core.general_rank==2 and core.isolated_rank==2
    assert len(report.module_groups["visual"])==6
    assert report.module_groups["semantic_anchor"]

    controller.begin_task(0)
    controller.start_energy_covariance_collection()
    model.train(); controller.begin_anchor_batch()
    model(torch.randn(4,12))
    diagnostics=controller.finalize_energy_object_subspaces(0)
    assert diagnostics["semantic_text_calls"] > 0
    assert diagnostics["collection_consumer_calls"] > 0
    assert all(abs(diagnostics[f"{group}_covariance_trace"]-12)<1e-3
               for group in ("semantic","box","mask"))
    gram=core.lora_A[0].float() @ core.lora_A[0].float().T
    assert torch.allclose(gram,torch.eye(4),atol=1e-4)
    controller.end_task()
    assert float(core.historical_covariance.trace()) > 0
    assert float(core.historical_specific_covariance.trace()) > 0


def test_compositional_anchor_starts_independent_and_only_updates_current_row():
    router=CompositionalPromptMLPTaskRouter(
        visual_dim=8,num_tasks=3,embeddings_per_task=4)
    router.train(); router.begin_task(1)
    router.prepare_visual(torch.randn(1,8),training=True)
    independent=router.route_for(torch.randn(1,2,5)).reshape(-1,3)[0]
    assert torch.equal(independent,torch.tensor([0.,1.,0.]))

    with torch.no_grad():
        router.composition_logits[1,0]=torch.atanh(torch.tensor(.5))
    inherited=router.route_for(torch.randn(1,2,5)).reshape(-1,3)[0]
    assert torch.allclose(inherited,torch.tensor([.5,1.,0.]),atol=1e-6)
    inherited[0].backward()
    gradient=router.composition_logits.grad
    assert gradient[1,0].abs()>0
    assert torch.count_nonzero(gradient)==1

    router.begin_task(2)
    protected=router.composition_logits.detach().clone()
    with torch.no_grad():
        router.composition_logits[1,0].add_(.25)
        router.composition_logits[2,0].add_(.25)
    router.restore_protected_composition()
    assert torch.equal(router.composition_logits[1],protected[1])
    assert router.composition_logits[2,0]!=protected[2,0]


def test_compositional_anchor_method_keeps_reference_lora_budget():
    baseline=FocusToy()
    baseline_report=inject_continual_lora(baseline,"sequential_lora",rank=16,alpha=8)
    model=FocusToy()
    report=inject_continual_lora(
        model,"compositional_anchor_lora",rank=16,alpha=8,num_tasks=5)
    routers=[m for m in model.modules()
             if isinstance(m,CompositionalPromptMLPTaskRouter)]
    assert len(routers)==1
    assert report.persistent_adapter_parameters<=baseline_report.persistent_adapter_parameters
    assert report.method=="compositional_anchor_lora"


def test_object_consistent_router_composes_parent_and_learns_utility_not_stage():
    torch.manual_seed(31)
    router=ObjectConsistentSkillRouter(visual_dim=8,num_tasks=3,embeddings_per_task=16)
    center0=torch.tensor([[1.,0,0,0,0,0,0,0.]]).repeat(8,1)
    center1=torch.tensor([[0.,1,0,0,0,0,0,0.]]).repeat(8,1)
    router.begin_task(0)
    router.store_utility(center0,torch.zeros(8,1))
    router.begin_task(1)
    # Stage-one samples counterfactually prefer old skill zero. The selector
    # target therefore intentionally differs from the stream-stage index.
    router.store_utility(center1,torch.tensor([[0.,1.]]).repeat(8,1))
    report=router.fit_selector(epochs=60,lr=5e-2,batch_size=8)
    assert report["accuracy"]>.99
    router.eval(); router.force_stage(None); router.begin_batch()
    router.prepare_visual(center1[:1],training=False)
    assert router._route.argmax(-1).item()==0

    router.set_parent(1,0,score=.1,base_score=.5)
    router.force_stage(1); router.prepare_visual(center1[:1],training=False)
    assert router._route.tolist()==[[1.,1.,0.]]


def test_object_selector_keeps_dense_owner_examples_for_stable_routing():
    router=ObjectConsistentSkillRouter(visual_dim=8,num_tasks=3,embeddings_per_task=16)
    router.begin_task(1); router.train(); router.begin_batch()
    features=torch.randn(4,8)
    router.prepare_visual(features,training=True)
    assert router.embedding_count[1].item()==4
    stored=router.utility_buffer[1,:4].float()
    assert torch.isfinite(stored[:,1]).all()
    assert (~torch.isfinite(stored[:,0])).all()


def test_object_utility_ranking_and_bidirectional_geometry_are_differentiable():
    components=torch.tensor([[1.,4.],[2.,1.],[3.,2.]])
    utility=_rank_normalized_utility(components)
    assert utility.argmin().item()==1
    assert torch.allclose(utility,torch.tensor([.5,.25,.75]))

    class Output: pass
    output=Output()
    prediction={
        "pred_boxes":torch.tensor([[[.5,.5,.5,.5]]],requires_grad=True),
        "pred_masks":torch.zeros(1,1,8,8,requires_grad=True),
        "pred_logits":torch.ones(1,1,1,requires_grad=True),
    }
    output.output=[[prediction]]
    value=_object_bidirectional_box_mask_consistency(output)
    assert value.ndim==0 and torch.isfinite(value) and value.item()>0
    value.backward()
    assert prediction["pred_boxes"].grad is not None
    assert prediction["pred_masks"].grad is not None


def test_object_consensus_keeps_only_agreeing_shared_coordinates():
    parameter=nn.Parameter(torch.tensor([1.,1.]))

    class Controller:
        def focus_parameters(self, group, trainable_only=True):
            return [parameter] if group=="visual" else []

    box=(parameter*torch.tensor([1.,1.])).sum()
    mask=(parameter*torch.tensor([1.,-1.])).sum()
    report=_object_consensus_backward(box+mask,{"box":box,"mask":mask},Controller())
    assert torch.allclose(parameter.grad,torch.tensor([1.,0.]))
    assert report["box_mask_gradient_agreement"]==pytest.approx(.5)


def test_consensus_complement_routes_common_and_private_gradients():
    parameter=nn.Parameter(torch.ones(3))

    class Controller:
        def consensus_complement_parameters(self):
            return [(parameter,
                     torch.tensor([True,False,False]),
                     torch.tensor([False,True,False]),
                     torch.tensor([False,False,True]))]

    semantic=(parameter*torch.tensor([1.,1.,1.])).sum()
    box=(parameter*torch.tensor([1.,2.,3.])).sum()
    mask=(parameter*torch.tensor([1.,-2.,-3.])).sum()
    report=_consensus_complement_backward(
        semantic+box+mask,{"semantic":semantic,"box":box,"mask":mask},Controller())
    assert torch.allclose(parameter.grad,torch.tensor([2.,2.,-3.]))
    assert report["box_mask_gradient_agreement"]==pytest.approx(1/3)
    assert report["box_complement_energy_fraction"]>0
    assert report["mask_complement_energy_fraction"]>0


def test_shared_core_uses_projected_full_rank_and_private_head_residuals():
    shared=nn.Parameter(torch.ones(2))
    box_head=nn.Parameter(torch.ones(1))
    mask_head=nn.Parameter(torch.ones(1))

    class Controller:
        def focus_parameters(self, group, trainable_only=True):
            return {"visual":[shared],"semantic":[],"box":[box_head],
                    "mask":[mask_head]}[group]

    semantic=(shared*torch.tensor([.5,.5])).sum()+3*box_head.sum()+4*mask_head.sum()
    box=(shared*torch.tensor([1.,0.])).sum()+2*box_head.sum()+5*mask_head.sum()
    mask=(shared*torch.tensor([-1.,1.])).sum()+7*box_head.sum()+6*mask_head.sum()
    report=_shared_core_residual_backward(
        semantic+box+mask,{"semantic":semantic,"box":box,"mask":mask},Controller())
    # Conflicting shared directions are projected rather than deleting a fixed
    # subset of rank coordinates; private heads keep only their matching loss.
    assert report["shared_conflict_projected"]==1
    assert torch.isfinite(shared.grad).all() and bool((shared.grad!=0).all())
    assert torch.allclose(box_head.grad,torch.tensor([2.]))
    assert torch.allclose(mask_head.grad,torch.tensor([6.]))


def test_object_consistent_method_keeps_reference_lora_budget():
    baseline=FocusToy()
    baseline_report=inject_continual_lora(baseline,"sequential_lora",rank=16,alpha=8)
    model=FocusToy()
    report=inject_continual_lora(model,"object_consistent_lora",rank=16,alpha=8,
                                 num_tasks=5)
    routers=[m for m in model.modules() if isinstance(m,ObjectConsistentSkillRouter)]
    assert len(routers)==1
    assert report.persistent_adapter_parameters<=baseline_report.persistent_adapter_parameters


def test_shared_core_residual_method_keeps_full_rank_fixed_budget():
    baseline=FocusToy()
    baseline_report=inject_continual_lora(baseline,"sequential_lora",rank=16,alpha=8)
    model=FocusToy()
    report=inject_continual_lora(model,"shared_core_residual_lora",rank=16,alpha=8,
                                 num_tasks=5)
    adapters=[m for m in model.modules() if isinstance(m,MLPTaskDecoupledLoRALinear)]
    routers=[m for m in model.modules() if isinstance(m,ObjectConsistentSkillRouter)]
    assert adapters and len(routers)==1
    assert not any(isinstance(m,ConsensusComplementLoRALinear) for m in adapters)
    assert report.persistent_adapter_parameters<=baseline_report.persistent_adapter_parameters


def test_unified_object_bridge_aligns_three_views_without_parameters():
    class Segmentation(nn.Module):
        def __init__(self):
            super().__init__(); self.instance_seg_head=nn.Conv2d(4,4,1)
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.segmentation_head=Segmentation()
    class Output: pass

    model=Model(); before=sum(p.numel() for p in model.parameters())
    bridge=UnifiedObjectBridge(model,max_instances=4,temperature=.1)
    assert sum(p.numel() for p in model.parameters())==before
    queries=torch.randn(1,3,4,requires_grad=True)
    pixels=torch.randn(1,4,8,8,requires_grad=True)
    prediction={
        "queries":queries,
        "pred_logits":torch.tensor([[[3.],[2.],[-2.]]]),
        "pred_boxes":torch.tensor([[[.5,.5,.5,.5],[.25,.25,.25,.25],[.8,.8,.1,.1]]]),
        "pred_masks":torch.randn(1,3,8,8),
        "indices":(torch.tensor([0,0]),torch.tensor([0,1]),None),
    }
    output=Output(); output.output=[[prediction]]
    bridge.begin_batch(); bridge._observe(None,None,pixels)
    loss,report=bridge.loss(output)
    assert torch.isfinite(loss) and report["instances"]==2
    loss.backward()
    assert queries.grad is not None and pixels.grad is not None


def test_unified_object_query_teacher_does_not_distort_pixel_views():
    class Segmentation(nn.Module):
        def __init__(self):
            super().__init__(); self.instance_seg_head=nn.Conv2d(4,4,1)
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.segmentation_head=Segmentation()
    class Output: pass

    queries=torch.randn(1,2,4,requires_grad=True)
    pixels=torch.randn(1,4,8,8,requires_grad=True)
    prediction={"queries":queries,"pred_logits":torch.ones(1,2,1),
                "pred_boxes":torch.tensor([[[.5,.5,.5,.5],[.25,.25,.2,.2]]]),
                "pred_masks":torch.randn(1,2,8,8),
                "indices":(torch.tensor([0,0]),torch.tensor([0,1]),None)}
    output=Output();output.output=[[prediction]]
    bridge=UnifiedObjectBridge(Model(),mode="query_teacher")
    bridge.begin_batch();bridge._observe(None,None,pixels)
    loss,_=bridge.loss(output);loss.backward()
    assert queries.grad is not None
    assert pixels.grad is None


def test_semantic_unified_object_uses_native_text_anchor_without_parameters():
    class Segmentation(nn.Module):
        def __init__(self):
            super().__init__(); self.instance_seg_head=nn.Conv2d(4,4,1)
    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.hs_proj=nn.Linear(4,4,bias=False)
            self.prompt_proj=nn.Linear(4,4,bias=False)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.segmentation_head=Segmentation(); self.dot_prod_scoring=Scorer()
    class Output: pass

    model=Model(); before=sum(p.numel() for p in model.parameters())
    bridge=UnifiedObjectBridge(model,mode="semantic_query_teacher")
    assert sum(p.numel() for p in model.parameters())==before
    queries=torch.randn(1,3,4,requires_grad=True)
    pixels=torch.randn(1,4,8,8,requires_grad=True)
    text=torch.randn(1,4,requires_grad=True)
    prediction={"queries":queries,"pred_logits":torch.ones(1,3,1),
                "pred_boxes":torch.tensor([[[.5,.5,.5,.5],[.25,.25,.2,.2],
                                             [.8,.8,.1,.1]]]),
                "pred_masks":torch.randn(1,3,8,8),
                "indices":(torch.tensor([0,0]),torch.tensor([0,1]),None)}
    output=Output();output.output=[[prediction]]
    bridge.begin_batch()
    model.dot_prod_scoring.prompt_proj(text)
    model.dot_prod_scoring.hs_proj(queries.unsqueeze(0))
    bridge._observe(None,None,pixels)
    loss,report=bridge.loss(output);loss.backward()
    assert torch.isfinite(loss) and report["semantic_alignment"]>0
    assert queries.grad is not None and pixels.grad is None
    # The language embedding is a teacher, not a trainable target of geometry.
    assert text.grad is None


def test_unified_object_method_uses_object_router_without_inference_growth():
    baseline=FocusToy()
    baseline_report=inject_continual_lora(baseline,"sequential_lora",rank=16,alpha=8)
    model=FocusToy()
    report=inject_continual_lora(model,"unified_object_lora",rank=16,alpha=8,
                                 num_tasks=5)
    assert len([m for m in model.modules() if isinstance(m,ObjectConsistentSkillRouter)])==1
    assert report.persistent_adapter_parameters<=baseline_report.persistent_adapter_parameters


def test_consensus_complement_method_has_role_masks_without_parameter_growth():
    baseline=FocusToy()
    baseline_report=inject_continual_lora(baseline,"sequential_lora",rank=16,alpha=8)
    model=FocusToy()
    report=inject_continual_lora(model,"consensus_complement_lora",rank=16,alpha=8,
                                 num_tasks=5)
    adapters=[m for m in model.modules() if isinstance(m,ConsensusComplementLoRALinear)]
    assert adapters
    controller=AdapterController(model,"consensus_complement_lora")
    controller.begin_task(0)
    entries=controller.consensus_complement_parameters()
    assert entries and all(len(entry)==4 for entry in entries)
    for _,object_role,box_role,mask_role in entries:
        assert not bool((object_role & box_role | object_role & mask_role |
                         box_role & mask_role).any())
    assert report.persistent_adapter_parameters<=baseline_report.persistent_adapter_parameters
