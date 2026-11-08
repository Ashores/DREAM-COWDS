"""Count effective CQSM slices separately from preallocated tensor storage."""
from .methods import CQSMDualLoRALinear


def cqsm_parameter_accounting(model, controller, task_id):
    """Report the task's optimization policy, not instantaneous nonzero gradients.

    Preallocated future/historical task slices can carry ``requires_grad=True``
    on their containing Tensor but are not independently updatable. Router
    training is a separate end-of-task optimization, not part of the joint loss.
    """
    groups = {name: 0 for name in (
        "private_allocated", "private_retained", "private_current_optimizable",
        "bank_readouts_allocated", "bank_readouts_retained", "bank_readouts_current_optimizable",
        "global_allocated", "global_current_optimizable")}
    cores = {}
    deconflict_allocated = 0
    deconflict_optimizable = 0
    for module in model.modules():
        if not isinstance(module, CQSMDualLoRALinear):
            continue
        allocated = int(module.allocated_mask.sum().item())
        per_task = module.lora_A[0].numel() + module.lora_B[0].numel()
        groups["private_allocated"] += module.num_tasks * per_task
        groups["private_retained"] += allocated * per_task
        if module.private_enabled:
            groups["private_current_optimizable"] += per_task
        if module.shared_enabled:
            bridge = getattr(module, "sdo_bridge", None)
            if bridge is not None:
                deconflict_allocated += sum(p.numel() for p in bridge.parameters())
                deconflict_optimizable += sum(p.numel() for p in bridge.parameters() if p.requires_grad)
            total = module.global_A.numel() + module.global_B.numel()
            groups["global_allocated"] += total
            groups["global_current_optimizable"] += (module.global_A.numel() if module.global_A.requires_grad else 0)
            groups["global_current_optimizable"] += (int(module.global_output_mask.sum()) * module.global_rank
                                                     if module.global_B.requires_grad else 0)
            if module.global_gate_logit is not None:
                groups["global_allocated"] += module.global_gate_logit.numel()
                if module.global_gate_logit.requires_grad:
                    groups["global_current_optimizable"] += module.global_gate_logit.numel()
        if module.uses_object_core:
            core = module._object_core
            cores[id(core)] = core
            readout = module.core_B[0].numel()
            groups["bank_readouts_allocated"] += module.num_tasks * readout
            groups["bank_readouts_retained"] += int(core.subspace_ready.sum().item()) * readout
            if module.bank_mode != "off":
                groups["bank_readouts_current_optimizable"] += readout
    router = controller.anchor_router()
    selector = sum(p.numel() for p in router.selector.parameters())
    basis_allocated = sum(core.lora_A.numel() for core in cores.values())
    basis_retained = sum(int(core.subspace_ready.sum().item()) * core.lora_A[0].numel()
                         for core in cores.values())
    core_buffers = []
    for index, core in enumerate(cores.values()):
        for name, value in core.named_buffers():
            core_buffers.append({"core": index, "name": name, "elements": value.numel(),
                                 "bytes": value.numel() * value.element_size(),
                                 "persistent": name not in core._non_persistent_buffers_set})
    descriptor_count = int(router.embedding_count.sum().item())
    descriptor_width = int(router.embedding_buffer.shape[-1])
    raw_requires_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    joint = sum(groups[name] for name in (
        "private_current_optimizable", "bank_readouts_current_optimizable",
        "global_current_optimizable")) + deconflict_optimizable
    if deconflict_allocated:
        groups.update(deconflict_allocated=deconflict_allocated,
                      deconflict_current_optimizable=deconflict_optimizable)
    return {
        "task_id": int(task_id), "definition": "slice-level optimization policy during this task",
        "counts": groups,
        "joint_loss_effective_trainable": joint,
        "selector_separate_optimizer_trainable": selector,
        "combined_optimizer_effective_trainable": joint + selector,
        "tensor_requires_grad_count_not_effective": raw_requires_grad,
        "adapter_preallocated_parameters": sum(groups[name] for name in (
            "private_allocated", "bank_readouts_allocated", "global_allocated")) + selector + basis_allocated + deconflict_allocated,
        "adapter_seen_task_parameters": sum(groups[name] for name in (
            "private_retained", "bank_readouts_retained", "global_allocated")) + selector + basis_retained + deconflict_allocated,
        "fixed_query_basis_allocated_elements": basis_allocated,
        "fixed_query_basis_retained_elements": basis_retained,
        "covariance_and_control_buffers": core_buffers,
        "descriptor_buffer_allocated_bytes": router.embedding_buffer.numel() * router.embedding_buffer.element_size(),
        "descriptor_buffer_populated_bytes": descriptor_count * descriptor_width * router.embedding_buffer.element_size(),
        "descriptor_count": descriptor_count, "descriptor_width": descriptor_width,
        "notes": ["Frozen bases count as stored parameters, never optimizer degrees of freedom.",
                  "Statistical/control buffers and retained image descriptors are not parameters.",
                  "Optimizer state, temporary activations and frozen SAM3 weights excluded from adapter counts.",
                  "Private-off and bank-off ablations retain allocated storage but remove their optimization and forward contributions."]}
