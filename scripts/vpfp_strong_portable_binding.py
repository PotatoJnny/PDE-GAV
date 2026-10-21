"""Explicit optional codec/proof binding; no model loading or selection here."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

SCHEMA = "vpfp_explicit_portable_codec_binding_v1"
CONVERSION_SCHEMA = "vpfp_explicit_portable_codec_conversion_probe_v1"
CODEC_SCHEMA = "vpfp_portable_initial_conditioned_frozen_buffers_v1"
SAMPLER_SCHEMA = "vpfp_explicit_portable_initial_conditioned_sampler_v1"


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "explicit lowercase SHA256 required")
    return value


def json_sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _checked_file(path, expected):
    path = Path(path)
    _require(path.is_absolute() and path.is_file() and not path.is_symlink(), "regular absolute portable asset required")
    return path


def portable_config(config):
    """Absent means legacy; null, partial, or malformed means failure."""
    if "portable_codec" not in config:
        return None
    item = config["portable_codec"]
    _require(isinstance(item, dict) and set(item) == {"schema", "path", "sha256", "conversion_result", "conversion_result_sha256"}
        and item["schema"] == SCHEMA, "invalid explicit portable codec config fields/schema")
    for key in ("path", "conversion_result"):
        _require(isinstance(item[key], str) and Path(item[key]).is_absolute(), "portable source paths must be absolute")
    _sha(item["sha256"])
    _sha(item["conversion_result_sha256"])
    return item


def portable_roles(config):
    item = portable_config(config)
    if item is None:
        return {}
    return {"portable_codec": (item["path"], item["sha256"], "portable-codec.pt"),
        "portable_codec_proof": (item["conversion_result"], item["conversion_result_sha256"], "portable-conversion-result.json")}


def validate_portable_proof(config, *, frozen_assets=None):
    """Verify the actual exporter schema; replay markers are not physics gates."""
    item = portable_config(config)
    if item is None:
        return None
    roles = portable_roles(config)
    actual = {}
    for role, (source, digest, _) in roles.items():
        source_path = _checked_file(source, digest)
        actual[role] = source_path
        if frozen_assets is not None:
            entry = frozen_assets[role]
            _require(entry["sha256"] == digest and Path(entry["source_path"]) == source_path.resolve(),
                "frozen portable source identity differs from config")
            actual[role] = _checked_file(entry["path"], digest)
    raw = actual["portable_codec_proof"].read_bytes()
    proof = json.loads(raw)
    _require(proof.get("schema") == CONVERSION_SCHEMA and proof.get("stage") == "complete", "wrong portable conversion proof schema/stage")
    for key in ("execution_passed", "inputs_final_verified", "outputs_final_verified", "source_final_verified"):
        _require(proof.get(key) is True, f"portable conversion incomplete: {key}")
    for key in ("old_checkpoint_restored", "guidance_ready", "guidance_enabled", "scientific_admission", "base_weights_modified"):
        _require(proof.get(key) is False, f"conversion proof contains unexpected qualification: {key}")
    _require(not proof.get("failures") and proof.get("new_samples") == 0 and proof.get("pde_solves") == 0,
        "conversion must be an unfailed load-only operation")
    _require(type(proof.get("tests", {}).get("returncode")) is int and proof["tests"]["returncode"] == 0,
        "portable conversion requires successful CPU tests")
    _require(type(proof.get("replay_markers", {}).get("completed")) is int and proof["replay_markers"]["completed"] == 128,
        "portable conversion requires completed128 replay")
    candidate, validation = proof["candidate"], proof["load_validation"]
    _require(candidate["path"] == item["path"] and candidate["sha256"] == item["sha256"], "conversion selected a different codec file")
    _require(proof["output_sha256"].get(item["path"]) == item["sha256"], "codec absent from verified conversion outputs")
    identity, compact, sampler = candidate["identity"], validation["compact_normalizer_identity"], validation["sampler_identity"]
    for key in ("execution_passed", "eigh_eigvalsh_disabled", "all_buffer_bytes_unchanged", "all_original_model_bytes_unchanged"):
        _require(validation.get(key) is True, f"portable load proof incomplete: {key}")
    _require(validation.get("loaded_buffer_count") == 15 and validation.get("samples_generated") == 0
        and validation.get("prior_runtime_loaded") is False, "unexpected portable validation scope")
    _require(identity.get("schema") == SAMPLER_SCHEMA and identity.get("source_checkpoint_sha256") == config["base_checkpoint_sha256"]
        and proof["input_sha256"].get(config["base_checkpoint"]) == config["base_checkpoint_sha256"], "portable proof belongs to another base checkpoint")
    for key in ("model_state_sha256", "original_normalizer_sha256", "candidate_normalizer_sha256", "buffer_state_sha256"):
        _sha(identity[key])
    _require(compact.get("schema") == CODEC_SCHEMA and compact.get("sha256") == _sha(candidate["content_sha256"])
        and compact.get("identity") == identity and len(compact.get("buffer_manifest", {})) == 15,
        "portable compact content binding differs")
    digest = json_sha(compact)
    _require(validation.get("compact_normalizer_sha256") == digest, "portable compact normalization SHA differs")
    _require(sampler.get("checkpoint_schema") == SAMPLER_SCHEMA
        and sampler.get("checkpoint_sha256") == config["base_checkpoint_sha256"]
        and sampler.get("portable_bundle_sha256") == item["sha256"]
        and sampler.get("portable_content_sha256") == candidate["content_sha256"]
        and sampler.get("portable_codec") == identity, "portable sampler proof binding differs")
    # Conversion revision need not equal the round revision. Its immutable
    # receipt SHA authenticates that earlier operation; no "guidance ready"
    # boolean or output-error threshold is inferred from it.
    return {"schema": SCHEMA, "source_checkpoint_sha256": config["base_checkpoint_sha256"],
        "codec_file_sha256": item["sha256"], "conversion_result_sha256": item["conversion_result_sha256"],
        "codec_content_sha256": candidate["content_sha256"], "normalizer_sha256": digest,
        "codec_identity": deepcopy(identity), "replay_markers": deepcopy(proof["replay_markers"]),
        "selection": "explicit_round_config_not_loader_failure_fallback", "scientific_admission": False}


def validate_loaded_portable(sampler, binding):
    compact = sampler.normalizer.to_payload()
    identity = sampler.identity
    _require(identity.get("checkpoint_schema") == SAMPLER_SCHEMA
        and identity.get("checkpoint_sha256") == binding["source_checkpoint_sha256"]
        and identity.get("portable_bundle_sha256") == binding["codec_file_sha256"]
        and identity.get("portable_content_sha256") == binding["codec_content_sha256"]
        and identity.get("portable_codec") == binding["codec_identity"]
        and json_sha(compact) == binding["normalizer_sha256"], "actual loaded portable sampler differs from conversion proof")
