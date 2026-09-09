#!/usr/bin/env python3
"""Create a self-contained U1 package using the pinned engine converters."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "third_party" / "llama.cpp"
sys.path.insert(0, str(LLAMA / "gguf-py"))

import gguf
import numpy as np
from safetensors import safe_open
import torch


def is_generation(name):
    return (name.startswith("language_model.") and "_mot_gen" in name) or name.startswith((
        "fm_modules.vision_model_mot_gen.embeddings.",
        "fm_modules.timestep_embedder.",
        "fm_modules.noise_scale_embedder.",
        "fm_modules.fm_head.",
    ))


def validate_config(source):
    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["NEOChatModel"]:
        raise ValueError("Only dense SenseNova U1.5 checkpoints are supported")
    llm = config["llm_config"]
    # The pinned sd.cpp generation loader uses these defaults when the
    # understanding tensors are absent. Reject other variants explicitly.
    expected = {
        "hidden_size": 4096, "intermediate_size": 12288, "head_dim": 128,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "num_hidden_layers": 42, "rms_norm_eps": 1e-6,
        "rope_theta": 5000000, "rope_theta_hw": 10000,
    }
    for key, value in expected.items():
        if llm.get(key) != value:
            raise ValueError(f"Unsupported U1.5 configuration: {key}={llm.get(key)!r}")
    if llm.get("num_experts", 0) or llm.get("attention_bias", False) or llm.get("rope_scaling"):
        raise ValueError("MoE, attention bias, and scaled RoPE are not supported")


def source_shards(source):
    index = source / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    shards = []
    for name in names:
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"Missing or external checkpoint shard: {name}")
        shards.append(path)
    return shards


def write_generation(shards, output):
    types = {
        "BF16": (np.dtype("uint16"), gguf.GGMLQuantizationType.BF16),
        "F16": (np.dtype("float16"), gguf.GGMLQuantizationType.F16),
        "F32": (np.dtype("float32"), gguf.GGMLQuantizationType.F32),
    }
    writer = gguf.GGUFWriter(output, "sensenova_u1")
    selected = []
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            names = sorted(name for name in tensors.keys() if is_generation(name))
            for name in names:
                tensor = tensors.get_slice(name)
                dtype, ggml_type = types[tensor.get_dtype()]
                shape = tensor.get_shape()
                writer.add_tensor_info(name, shape, dtype, math.prod(shape) * dtype.itemsize,
                                       raw_dtype=ggml_type)
            selected.append((shard, names))
    names = {name for _, shard_names in selected for name in shard_names}
    required = {
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
        "fm_modules.fm_head.conv1.weight",
    }
    if not required <= names:
        raise ValueError(f"Missing generation tensors: {sorted(required - names)}")
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        # Stream one tensor at a time, preserving its source dtype and bytes.
        for shard, shard_names in selected:
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                for name in shard_names:
                    tensor = tensors.get_tensor(name)
                    data = tensor.view(torch.uint16) if tensor.dtype == torch.bfloat16 else tensor
                    writer.write_tensor_data(data.numpy())
                    del tensor, data
    finally:
        writer.close()
    return len(names)


def convert(source, output, outtype):
    source = source.resolve(strict=True)
    output = output.absolute()
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new directory")
    validate_config(source)
    shards = source_shards(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        package = Path(temporary) / "package"
        package.mkdir()
        subprocess.run([
            sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(source),
            "--outtype", outtype, "--outfile", str(package / "understanding.gguf"),
        ], check=True)
        count = write_generation(shards, package / "generation.gguf")
        manifest = {
            "format": "umm", "version": 1, "architecture": "sensenova_u1",
            "understanding": "understanding.gguf", "generation": "generation.gguf",
        }
        (package / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
        package.rename(output)
    print(f"Created {output} ({count} generation tensors; tokenizer embedded in understanding.gguf)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Official dense U1.5 checkpoint directory")
    parser.add_argument("--output", required=True, type=Path, help="New model package directory")
    parser.add_argument("--outtype", choices=("bf16", "f16", "f32", "q8_0"), default="bf16",
                        help="Understanding weights format (default: bf16); generation preserves source precision")
    args = parser.parse_args()
    try:
        convert(args.checkpoint, args.output, args.outtype)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"umm conversion: {error}\n")


if __name__ == "__main__":
    main()
