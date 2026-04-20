#!/usr/bin/env python3
# Quantization requires a separate venv — llmcompressor conflicts with the main project:
#   llmcompressor>=0.9.0 requires accelerate<=1.12.0 (project needs >=1.13.0)
#
# Setup:
#   python -m venv /tmp/quant-env
#   /tmp/quant-env/bin/pip install llmcompressor torch transformers datasets accelerate huggingface_hub
#   /tmp/quant-env/bin/python scripts/quantize_awq.py [--push-only]
#
# Push only (uses main venv, no llmcompressor needed):
#   uv run scripts/quantize_awq.py --push-only --hf-token $HF_TOKEN
import argparse
import logging
import os
from pathlib import Path

# Must be set before torch initializes the CUDA allocator.
# expandable_segments reduces fragmentation that causes OOM on tight GPUs (<12GB).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = "models/qwen3-4b-instruct-2507-awq"
DEFAULT_HUB_REPO = "quannguyen204/Qwen3-4B-Instruct-2507-AWQ-W4A16"

NUM_CALIBRATION_SAMPLES = 512
# 256 keeps activation memory ~130MB per forward pass on a 4B fp16 model (~8GB weights).
MAX_SEQ_LENGTH = 256


def build_calibration_dataset(tokenizer: AutoTokenizer):
    """Build a tokenized Vietnamese calibration dataset for llm-compressor oneshot.

    Calibration distribution should match real inference inputs.
    Using Vietnamese instruction data preserves activation ranges for Vietnamese tokens.
    """
    candidates = [
        ("bkai-foundation-models/vi-alpaca", "train"),
        ("5CD-AI/Vietnamese-alpaca-gpt4-data-viet", "train"),
    ]

    raw_ds = None
    for dataset_id, split in candidates:
        try:
            logger.info(f"Loading calibration dataset: {dataset_id}")
            # Load 2x samples to have enough after filtering short ones
            raw_ds = load_dataset(
                dataset_id, split=f"{split}[:{NUM_CALIBRATION_SAMPLES * 2}]"
            )
            break
        except Exception as e:
            logger.warning(f"Could not load {dataset_id}: {e}")

    if raw_ds is None:
        raise RuntimeError(
            "Could not load any Vietnamese calibration dataset. "
            "Check your internet connection or HuggingFace access."
        )

    raw_ds = raw_ds.shuffle(seed=42)

    def preprocess(row):
        instruction = (row.get("instruction") or "").strip()
        input_text = (row.get("input") or "").strip()
        output_text = (row.get("output") or "").strip()

        user_turn = f"{instruction}\n\n{input_text}" if input_text else instruction
        messages = [
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": output_text},
        ]
        # apply_chat_template includes Qwen3 special tokens (<|im_start|>, <|im_end|>)
        # which must be present in calibration data so AWQ sees real serving token distributions.
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    ds = raw_ds.map(preprocess, remove_columns=raw_ds.column_names)

    def tokenize(sample):
        # add_special_tokens=False: chat_template already added bos/special tokens
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["input_ids"]) >= 32)

    if len(ds) > NUM_CALIBRATION_SAMPLES:
        ds = ds.select(range(NUM_CALIBRATION_SAMPLES))

    logger.info(f"Calibration dataset ready: {len(ds)} Vietnamese samples")
    return ds


def push_folder_to_hub(local_dir: str, hub_repo: str, hf_token: str | None) -> None:
    """Upload a local saved model folder to HuggingFace Hub.

    Uses upload_folder — no model loading required, works from the main project venv.
    """
    if hf_token:
        login(token=hf_token)

    api = HfApi()
    api.create_repo(repo_id=hub_repo, repo_type="model", exist_ok=True)

    logger.info(f"Uploading {local_dir} → {hub_repo}")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=hub_repo,
        repo_type="model",
    )
    logger.info(f"Uploaded to https://huggingface.co/{hub_repo}")


def quantize(
    model_id: str,
    output_dir: str,
    hf_token: str | None,
    push_to_hub: bool,
    hub_repo: str,
) -> None:
    # Lazy import: llmcompressor conflicts with main project deps (accelerate version),
    # so this script must run in a separate venv when quantizing.
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError(
            f"llmcompressor not found ({e}). "
            "Run quantization in a separate venv — see the comment at the top of this script."
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for AWQ quantization")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # bfloat16: same memory as fp16 but much larger dynamic range (same exponent as fp32).
        # Qwen3 is trained in bfloat16 — loading in fp16 causes overflow/NaN during
        # AWQ scale grid search because fp16 max is ~65504 vs bfloat16 ~3.4e38.
        dtype=torch.bfloat16,
        device_map="auto",
    )

    calib_ds = build_calibration_dataset(tokenizer)
    torch.cuda.empty_cache()

    # AWQ recipe:
    #   scheme="W4A16_ASYM" : 4-bit weights, 16-bit activations, asymmetric (zero_point),
    #                          group_size=128 — best accuracy/throughput balance for vLLM
    #   offload_device=cpu  : offload cached activation tensors to CPU during scale search;
    #                          critical for GPUs <12GB where fp16 weights already use ~8GB
    recipe = AWQModifier(
        ignore=["lm_head"],
        scheme="W4A16_ASYM",
        targets="Linear",
        offload_device=torch.device("cpu"),
    )

    logger.info("Starting AWQ quantization via llm-compressor...")
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    logger.info(f"Saving quantized model to {output_path}")
    model.save_pretrained(str(output_path), save_compressed=True)
    tokenizer.save_pretrained(str(output_path))
    logger.info("Local save complete")

    if push_to_hub:
        push_folder_to_hub(str(output_path), hub_repo, hf_token)

    logger.info("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3-4B-Instruct-2507 to AWQ (W4A16) via llm-compressor"
    )
    parser.add_argument(
        "--model-id", default=MODEL_ID, help="HF model ID or local path"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Local output directory"
    )
    parser.add_argument("--hf-token", default=None, help="HuggingFace write token")
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push to HF Hub after quantization"
    )
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="Skip quantization — upload an already-saved local model to HF Hub",
    )
    parser.add_argument(
        "--hub-repo", default=DEFAULT_HUB_REPO, help="HF repo ID to push to"
    )
    args = parser.parse_args()

    if args.push_only:
        push_folder_to_hub(args.output_dir, args.hub_repo, args.hf_token)
    else:
        quantize(
            model_id=args.model_id,
            output_dir=args.output_dir,
            hf_token=args.hf_token,
            push_to_hub=args.push_to_hub,
            hub_repo=args.hub_repo,
        )


if __name__ == "__main__":
    main()
