#!/usr/bin/env python3
# uv add autoawq autoawq-kernels transformers datasets huggingface_hub accelerate
import argparse
import logging
import os
from pathlib import Path

# Must be set before torch initializes the CUDA allocator.
# expandable_segments reduces fragmentation that causes OOM on tight GPUs (<12GB).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from awq import AutoAWQForCausalLM
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = "models/qwen3-4b-instruct-2507-awq"
DEFAULT_HUB_REPO = "quannguyen204/Qwen3-4B-Instruct-2507-AWQ-4bit"

# AWQ config tuned for vLLM:
#   w_bit=4        : 4-bit weights — standard for AWQ, ~3.5x memory reduction
#   q_group_size=128: channel grouping — balances accuracy vs kernel speed;
#                     smaller (64) improves accuracy at throughput cost
#   zero_point=True: asymmetric quant — better accuracy for instruction models
#   version="GEMM" : GEMM kernel — higher throughput for batched prefill in vLLM;
#                    use "GEMV" only for single-stream, decode-heavy workloads
AWQ_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

CALIB_N_SAMPLES = 512
# 256 tokens keeps peak activation memory ~130MB per forward pass on 4B model,
# well within an 11GB GPU that already holds the fp16 weights (~8GB).
CALIB_SEQ_LEN = 256


def _format_chat(tokenizer: AutoTokenizer, instruction: str, response: str) -> str:
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_calibration_data(tokenizer: AutoTokenizer) -> list[str]:
    candidates = [
        ("bkai-foundation-models/vi-alpaca", "train", "instruction", "output", "input"),
        (
            "5CD-AI/Vietnamese-alpaca-gpt4-data-viet",
            "train",
            "instruction",
            "output",
            "input",
        ),
    ]

    ds = None
    for dataset_id, split, instr_col, output_col, input_col in candidates:
        try:
            logger.info(f"Loading calibration dataset: {dataset_id}")
            ds = load_dataset(dataset_id, split=split)
            break
        except Exception as e:
            logger.warning(f"Could not load {dataset_id}: {e}")

    if ds is None:
        raise RuntimeError(
            "Could not load any Vietnamese calibration dataset. "
            "Check your internet connection or HuggingFace access."
        )

    samples: list[str] = []
    for row in ds:
        if len(samples) >= CALIB_N_SAMPLES:
            break

        instruction = row.get(instr_col, "").strip()
        input_text = row.get(input_col, "").strip()
        output_text = row.get(output_col, "").strip()

        if not instruction or not output_text:
            continue

        user_turn = f"{instruction}\n\n{input_text}" if input_text else instruction

        try:
            text = _format_chat(tokenizer, user_turn, output_text)
        except Exception:
            text = f"{user_turn}\n{output_text}"

        tokens = tokenizer(text, truncation=True, max_length=CALIB_SEQ_LEN)
        if len(tokens["input_ids"]) < 32:
            continue

        samples.append(text)

    logger.info(f"Built {len(samples)} Vietnamese calibration samples")
    return samples


def quantize(
    model_id: str,
    output_dir: str,
    hf_token: str | None,
    push_to_hub: bool,
    hub_repo: str,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for AWQ quantization")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load model in fp16 on GPU for quantization
    logger.info(f"Loading model: {model_id}")
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        safetensors=True,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Calibration data
    calib_data = build_calibration_data(tokenizer)

    # Free any cached allocations before the memory-heavy quantization loop
    torch.cuda.empty_cache()

    # Quantize
    logger.info(f"Starting AWQ quantization — config: {AWQ_CONFIG}")
    model.quantize(tokenizer, quant_config=AWQ_CONFIG, calib_data=calib_data)

    # Save locally
    logger.info(f"Saving quantized model to {output_path}")
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info("Local save complete")

    # Push to Hub
    if push_to_hub:
        if hf_token:
            login(token=hf_token)

        logger.info(f"Pushing to HuggingFace Hub: {hub_repo}")
        model.push_to_hub(hub_repo, use_temp_dir=True)
        tokenizer.push_to_hub(hub_repo)
        logger.info(f"Uploaded to https://huggingface.co/{hub_repo}")

    logger.info("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3-4B-Instruct-2507 to AWQ for vLLM serving"
    )
    parser.add_argument(
        "--model-id", default=MODEL_ID, help="HF model ID or local path"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Local output directory"
    )
    parser.add_argument("--hf-token", default=None, help="HuggingFace write token")
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push quantized model to HF Hub"
    )
    parser.add_argument(
        "--hub-repo", default=DEFAULT_HUB_REPO, help="HF repo ID to push to"
    )
    args = parser.parse_args()

    quantize(
        model_id=args.model_id,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )


if __name__ == "__main__":
    main()
