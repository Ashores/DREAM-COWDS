"""Train-time, asymmetric SDO-inspired Global/Private composition.

The fixed orthonormal Global A is the stable input basis throughout training.
Only Global output directions are rewritten. Detached, live factor signatures
condition a residual MLP; native task gradients train Global, current Private,
CQSM and the MLP, while geometric objectives train Global and the MLP only.
All geometric feature construction is outside autograd, including zero-rank
initialization. No images, features, or future-task statistics are stored here.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F


def spd_sqrt(matrix, iterations=18):
    """Batched SPD square root without singular-vector differentiation."""
    eye=torch.eye(matrix.shape[-1],device=matrix.device,dtype=matrix.dtype)
    scale=matrix.square().sum((-2,-1),keepdim=True).sqrt().clamp_min(1e-12)
    y,z=matrix/scale,eye.expand_as(matrix)
    for _ in range(iterations):
        update=.5*(3*eye-z@y)
        y,z=y@update,update@z
    return .5*(y+y.transpose(-2,-1))*scale.sqrt()


def polar(b, iterations=18):
    gram=b.transpose(-2,-1)@b
    eye=torch.eye(gram.shape[-1],device=b.device,dtype=b.dtype)
    ridge=gram.diagonal(dim1=-2,dim2=-1).mean(-1,keepdim=True).detach().clamp_min(1e-12)*1e-6
    gram=gram+ridge.unsqueeze(-1)*eye
    scale=gram.square().sum((-2,-1),keepdim=True).sqrt().clamp_min(1e-12)
    y,z=gram/scale,eye.expand_as(gram)
    for _ in range(iterations):
        update=.5*(3*eye-z@y)
        y,z=y@update,update@z
    h=.5*(y+y.transpose(-2,-1))*scale.sqrt()
    inverse=.5*(z+z.transpose(-2,-1))/scale.sqrt()
    return b@inverse,h


def normalized_pair_energy(qg,hg,qp,hp,root_hp):
    product=spd_sqrt(hg)@qg.transpose(-2,-1)@qp@root_hp
    energy=product.square().sum((-2,-1))
    magnitude=(hg.diagonal(dim1=-2,dim2=-1).sum(-1)*
               hp.diagonal(dim1=-2,dim2=-1).sum(-1)).clamp_min(1e-12)
    return energy/magnitude


class LiveOutputRewriter(nn.Module):
    def __init__(self,global_rank,private_rank,hidden=32,max_correction=.25):
        super().__init__()
        self.global_rank=int(global_rank)
        self.private_rank=int(private_rank)
        self.max_correction=float(max_correction)
        context=global_rank*private_rank+global_rank+private_rank+2
        width=global_rank+private_rank+context
        self.network=nn.Sequential(nn.Linear(width,hidden),nn.SiLU(),nn.Linear(hidden,global_rank))
        self.gate=nn.Linear(context,1)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def correction(self,features,strength):
        data,context,rms,valid=features['features'],features['context'],features['rms'],features['valid']
        gate=self.gate(context).sigmoid().unsqueeze(-1)
        return self.network(data).tanh()*(rms*self.max_correction*strength)*gate*valid[:,None,None]


class OnlineDeconflictor(nn.Module):
    def __init__(self,global_rank,private_rank,*,hidden=32,max_correction=.25,
                 conflict_weight=.05,preservation_weight=1.):
        super().__init__()
        self.parts=nn.ModuleDict({key:LiveOutputRewriter(global_rank,private_rank,hidden,max_correction)
                                  for key in ['q','v']})
        self.register_buffer('strength',torch.ones(()))
        self.conflict_weight=float(conflict_weight)
        self.preservation_weight=float(preservation_weight)
        self._cache=None
        self._cache_key=None

    @torch.no_grad()
    def prepare(self,module):
        key=(module.global_A._version,module.global_B._version,module.lora_A._version,
             module.lora_B._version,module.allocated_mask._version,str(module.global_B.device))
        if self._cache_key==key and self._cache is not None:
            return self._cache
        allocated=torch.where(module.allocated_mask)[0]
        if not len(allocated):
            raise RuntimeError('begin_task must precede online deconflicting')
        if not torch.equal(allocated,torch.arange(len(allocated),device=allocated.device)):
            raise RuntimeError('Online deconflicting expects a contiguous private-task prefix')
        device=module.global_B.device
        with torch.autocast(device_type=device.type,enabled=False):
            private_a=module.lora_A[allocated].detach().float()
            a_gram=private_a@private_a.transpose(-2,-1)
            a_root=spd_sqrt(a_gram)
            width=module.in_features
            result={}
            for name,sl in [('q',slice(0,width)),('v',slice(2*width,3*width))]:
                g=module.global_B[sl].detach().float()
                p=(module.lora_B[allocated,sl].detach().float()*module.scaling)@a_root
                qg,hg=polar(g)
                qp,hp=polar(p)
                root_hp=spd_sqrt(hp)
                overlap=qg.T@qp
                rho=overlap.square().sum((-2,-1))/min(g.shape[-1],p.shape[-1])
                ng=g.norm().clamp_min(1e-8)
                np=p.square().sum((-2,-1)).sqrt().clamp_min(1e-8)
                rms=g.square().mean().sqrt().clamp_min(1e-6)
                p_rms=p.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
                cg=g.square().sum(0).sqrt()/ng
                cp=p.square().sum(-2).sqrt()/np[:,None]
                context=torch.cat((overlap.flatten(1),cg.expand(len(allocated),-1),cp,
                                   rho[:,None],torch.log(np/ng)[:,None]),dim=-1)
                signatures=torch.cat(((g/rms).expand(len(allocated),-1,-1),p/p_rms,
                                      context[:,None,:].expand(-1,g.shape[0],-1)),dim=-1)
                valid=(g.square().sum()>1e-12)&(p.square().sum((-2,-1))>1e-12)
                reference=normalized_pair_energy(qg,hg,qp,hp,root_hp).clamp_min(1e-7)
                result[name]={'features':signatures,'context':context,'rms':rms,'valid':valid.float(),
                    'qg':qg,'hg':hg,'qp':qp,'hp':hp,'root_hp':root_hp,'reference':reference,
                    'original_g':g,'slice':sl,'allocated':allocated}
        self._cache,self._cache_key=result,key
        return result

    def forward(self,x,module):
        # No dynamic captures or differentiable statistics enter checkpoint hooks.
        if float(self.strength)==0.:
            return x.new_zeros(*x.shape[:-1],module.out_features)
        cache=self.prepare(module)
        route=module._anchor_router.route_for(x,group=module.function_group)
        selected=torch.unique(route.reshape(-1,route.shape[-1]).argmax(-1)).tolist()
        pieces={}
        for name,part in self.parts.items():
            features=cache[name]
            with torch.autocast(device_type=x.device.type,enabled=False):
                db=part.correction(features,self.strength)
            update=x.new_zeros(*x.shape[:-1],module.in_features)
            for task in selected:
                if task>=len(db):
                    raise RuntimeError('Selected a private module absent from the signature cache')
                delta=F.linear(F.linear(x,module.global_A.to(x.dtype)),db[task].to(x.dtype))
                update=update+delta*route[...,task].unsqueeze(-1)
            pieces[name]=update
        return torch.cat((pieces['q'],torch.zeros_like(pieces['q']),pieces['v']),dim=-1)

    def losses(self,module):
        cache=self.prepare(module)
        losses=[]
        for name,part in self.parts.items():
            features=cache[name]
            db=part.correction(features,self.strength)
            g=module.global_B[features['slice']].float()
            transformed=g[None]+db
            q,h=polar(transformed)
            value=normalized_pair_energy(q,h,features['qp'],features['hp'],features['root_hp'])
            valid=features['valid']
            conflict=(value/features['reference']*valid).sum()/valid.sum().clamp_min(1)
            q0,h0=features['qg'],features['hg']
            cross=q0.T@q
            subspace=((q.transpose(-2,-1)@q).square().sum((-2,-1))+q0.square().sum()
                       -2*cross.square().sum((-2,-1))).clamp_min(0)/max(g.shape[-1],1)
            magnitude=(h-h0).square().sum((-2,-1))/h0.square().sum().clamp_min(1e-10)
            residual=db.square().sum((-2,-1))/features['original_g'].square().sum().clamp_min(1e-10)
            preserve=((subspace+magnitude+residual)*valid).sum()/valid.sum().clamp_min(1)
            ratio=transformed.square().sum((-2,-1)).sqrt()/features['original_g'].norm().clamp_min(1e-8)
            overlap=(q.transpose(-2,-1)@features['qp']).square().sum((-2,-1))/min(g.shape[-1],features['qp'].shape[-1])
            report={'relative_correction':(residual.sqrt()*valid).sum()/valid.sum().clamp_min(1),
                    'relative_global_norm':(ratio*valid).sum()/valid.sum().clamp_min(1),
                    'overlap':(overlap*valid).sum()/valid.sum().clamp_min(1),
                    'valid_pairs':valid.sum()}
            losses.append((conflict,preserve,report))
        return (torch.stack([v[0] for v in losses]).mean(),torch.stack([v[1] for v in losses]).mean(),
                {k:torch.stack([v[2][k] for v in losses]).mean() for k in losses[0][2]})


def attach_online_deconflict(model,report,*,hidden=32,max_correction=.25,
                            conflict_weight=.05,preservation_weight=1.):
    from .methods import CQSMDualLoRALinear
    names=[]
    for name,module in list(model.named_modules()):
        if not isinstance(module,CQSMDualLoRALinear) or not module.shared_enabled:
            continue
        if not module.private_enabled or module.out_features!=3*module.in_features or module.global_gate_logit is not None:
            raise ValueError('Online deconflicting requires ungated, co-located Q/V Global and Private')
        if getattr(module,'sdo_bridge',None) is not None:
            raise RuntimeError('Online deconflictor already attached')
        seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
        # Preserve the random stream used for LoRA initialization/data sampling.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            bridge=OnlineDeconflictor(module.global_rank,module.rank,hidden=hidden,
                max_correction=max_correction,conflict_weight=conflict_weight,
                preservation_weight=preservation_weight)
        module.add_module('sdo_bridge',bridge.to(module.global_B.device))
        names.append(name)
    if not names:
        raise ValueError('No eligible Global/Private mappings')
    groups={**(report.module_groups or {}),'deconflict_operator':names}
    extra=sum(p.numel() for n,p in model.named_parameters() if '.sdo_bridge.' in n)
    return replace(report,module_groups=groups,trainable_parameters=report.trainable_parameters+extra,
                   total_parameters=report.total_parameters+extra,
                   persistent_adapter_parameters=report.persistent_adapter_parameters+extra)


@torch.no_grad()
def set_deconflict_strength(model,value):
    for module in model.modules():
        if isinstance(module,OnlineDeconflictor):
            module.strength.fill_(float(value))


def online_deconflict_loss(model):
    entries=[]
    for module in model.modules():
        bridge=getattr(module,'sdo_bridge',None)
        if not isinstance(bridge,OnlineDeconflictor):
            continue
        if float(bridge.strength)==0.:
            continue
        with torch.autocast(device_type=module.global_B.device.type,enabled=False):
            conflict,preserve,report=bridge.losses(module)
            weighted=bridge.strength*(bridge.conflict_weight*conflict+bridge.preservation_weight*preserve)
        entries.append((weighted,conflict,preserve,report))
    if not entries:
        return next(model.parameters()).new_zeros(()),{'sdo_enabled':1,'sdo_strength':0.,'sdo_weighted_loss':0.}
    loss=torch.stack([v[0] for v in entries]).mean()
    report={'sdo_enabled':1,'sdo_strength':float(next(m for m in model.modules() if isinstance(m,OnlineDeconflictor)).strength),
            'sdo_conflict':float(torch.stack([v[1].detach() for v in entries]).mean()),
            'sdo_preservation':float(torch.stack([v[2].detach() for v in entries]).mean()),
            'sdo_weighted_loss':float(loss.detach())}
    report.update({'sdo_'+k:float(torch.stack([v[3][k].detach() for v in entries]).mean()) for k in entries[0][3]})
    return loss,report
