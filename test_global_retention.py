import torch
import pytest
from torch.utils.checkpoint import checkpoint
from test_cqsm import setup_model, initialize_task, step
from continual_lora.global_retention import GlobalRetention

def test_fixed_basis_and_local_retention_gradients():
    model,c,o,_=setup_model(restore_legacy=True)
    keeper=GlobalRetention(model)
    x=torch.randn(2,3,model.width)
    initialize_task(model,c,o,0,x)
    for m in keeper.modules:
        assert not m.global_A.requires_grad
        assert torch.allclose(m.global_A@m.global_A.T,torch.eye(m.global_rank),atol=1e-6)
    keeper.begin_task(0);keeper.begin_batch();c.begin_anchor_batch();model(x)
    assert keeper.loss()==0
    step(model,c,o,x)
    keeper.begin_task(1)
    with torch.no_grad():
        for m in keeper.modules:m.global_B.add_(0.1)
    before=[m.global_A.clone() for m in keeper.modules]
    o.zero_grad(set_to_none=True);keeper.begin_batch();c.begin_anchor_batch();model(x)
    loss=keeper.loss();assert loss>0;loss.backward()
    shared={id(m.global_B) for m in keeper.modules}
    assert any(m.global_B.grad is not None and m.global_B.grad.norm()>0 for m in keeper.modules)
    assert all(p.grad is None for p in model.parameters() if id(p) not in shared)
    o.step()
    assert all(torch.equal(a,m.global_A) for a,m in zip(before,keeper.modules))
    assert not keeper.active and not keeper.terms
    keeper.close()

def test_restore_trainability_and_state_reload():
    model,c,_,_=setup_model(restore_legacy=True)
    c.restore_focus_trainability()
    keeper=GlobalRetention(model)
    assert all(not m.global_A.requires_grad and m.global_B.requires_grad for m in keeper.modules)
    other,_,_,_=setup_model(restore_legacy=True)
    other.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.allclose(v,other.state_dict()[k],equal_nan=True) for k,v in model.state_dict().items())


@pytest.mark.parametrize('reentrant', [False, True])
def test_retention_outside_activation_checkpoint_matches_plain_gradients(reentrant):
    results = []
    for checkpointed in [False, True]:
        model,c,o,_=setup_model(restore_legacy=True)
        x=torch.randn(2,3,model.width,requires_grad=True)
        initialize_task(model,c,o,0,x)
        keeper=GlobalRetention(model)
        keeper.begin_task(1)
        with torch.no_grad():
            for m in keeper.modules:m.global_B.add_(0.1)
        keeper.begin_batch();c.begin_anchor_batch()
        def forward(value):
            c.begin_anchor_batch()
            return model(value)
        y=checkpoint(forward,x,use_reentrant=reentrant) if checkpointed else forward(x)
        retention=keeper.loss()
        assert retention>0
        (y.square().mean()+retention).backward()
        results.append([m.global_B.grad.clone() for m in keeper.modules])
        assert not keeper.active and not keeper.terms
        keeper.close()
    assert all(torch.allclose(a,b,atol=1e-6) for a,b in zip(*results))
