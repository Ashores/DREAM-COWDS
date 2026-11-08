"""Cache-identity checks independent of SAM 3, datasets, or GPU execution."""
import pytest
from evaluate_continual_lora import validate_cached_metrics


def record(adapter=None):
    return {"annotation_sha256":"annotations-A", "base_checkpoint_sha256":"backbone-A",
            "adapter_checkpoint_sha256":adapter,
            "adapter_checkpoint":None if adapter is None else "adapter.pt",
            "threshold":0.3, "images":115}


def verify(metrics, adapter=None):
    validate_cached_metrics(metrics,annotation_sha256="annotations-A",
                            base_checkpoint_sha256="backbone-A",
                            adapter_checkpoint_sha256=adapter,
                            expected_images=115,location="synthetic-cache")


@pytest.mark.parametrize("adapter", [None,"adapter-A"])
def test_matching_frozen_and_adapted_records_are_accepted(adapter):
    verify(record(adapter),adapter)


@pytest.mark.parametrize("adapter", [None,"adapter-A"])
def test_changed_backbone_at_the_same_path_is_rejected(adapter):
    metrics=record(adapter)
    metrics["base_checkpoint_sha256"]="backbone-B"
    with pytest.raises(RuntimeError,match="base_checkpoint_sha256"):
        verify(metrics,adapter)


def test_legacy_record_without_backbone_hash_is_rejected():
    metrics=record()
    del metrics["base_checkpoint_sha256"]
    with pytest.raises(RuntimeError,match="base_checkpoint_sha256"):
        verify(metrics)


@pytest.mark.parametrize("key,value", [
    ("images",1), ("annotation_sha256","another-split"),
    ("threshold",0.5), ("adapter_checkpoint_sha256","adapter-B")])
def test_partial_or_incompatible_frozen_references_are_rejected(key,value):
    metrics=record()
    metrics[key]=value
    with pytest.raises(RuntimeError,match=key): verify(metrics)


def test_adapted_result_cannot_be_used_as_frozen_reference():
    metrics=record()
    metrics["adapter_checkpoint"]="unexpected-adapter.pt"
    with pytest.raises(RuntimeError,match="adapter_checkpoint"): verify(metrics)


def test_missing_adapter_identity_is_not_assumed_to_mean_frozen():
    metrics=record()
    del metrics["adapter_checkpoint_sha256"]
    with pytest.raises(RuntimeError,match="adapter_checkpoint_sha256"): verify(metrics)
