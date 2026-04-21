"""Quantize Qwen3.5-4B (multimodal, MoE) using llm-compressor.

Default preset: vast.ai RTX 3090 (24 GB)
  --scheme w4a16     : GPTQ W4A16 with calibration data (primary)
  --scheme fp8-dynamic: FP8 dynamic, data-free (safe fallback for MoE)

Usage:
  # W4A16 (recommended, may auto-fallback to FP8 if MoE not yet supported)
  uv run --group quantize python scripts/quantize_llmcompressor.py \\
      --scheme w4a16 --auto-fallback \\
      --push-to-hub --hub-id <your-namespace>/Qwen3.5-4B-W4A16-G128

  # FP8 always-works path
  uv run --group quantize python scripts/quantize_llmcompressor.py --scheme fp8-dynamic
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B", help="HuggingFace model ID to quantize")
    p.add_argument("--scheme", choices=["w4a16", "fp8-dynamic"], default="w4a16", help="Quantization scheme")
    p.add_argument("--output-dir", default=None, help="Output dir (default: <model-id-basename>-<SCHEME>-G128)")
    p.add_argument("--num-calibration-samples", type=int, default=256, help="Calibration samples (W4A16 only)")
    p.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length for calibration")
    p.add_argument("--calibration-dataset", default="khoalnd/EconVNNews", help="HF dataset for text calibration")
    p.add_argument("--multimodal-calib-ratio", type=float, default=0.3, help="Fraction of calibration samples that include an image (0 = text-only)")
    p.add_argument("--push-to-hub", action="store_true", help="Push quantized model to HuggingFace Hub")
    p.add_argument("--hub-id", default=None, help="HF Hub repo ID for push (e.g. your-namespace/Qwen3.5-4B-W4A16)")
    p.add_argument("--auto-fallback", action="store_true", help="On W4A16 failure, automatically retry with FP8-dynamic")
    p.add_argument("--sanity-check", action="store_true", default=True, help="Run a quick text + image generation after quantization")
    p.add_argument("--no-sanity-check", dest="sanity_check", action="store_false")
    return p.parse_args()


# ─── Calibration data ─────────────────────────────────────────────────────────

def _build_calibration_dataset(processor, args):
    """Build a mixed text+multimodal calibration dataset."""
    from datasets import load_dataset
    import random

    print(f"[calib] loading {args.calibration_dataset} for text samples …")
    raw = load_dataset(args.calibration_dataset, split="train", streaming=True)
    text_samples = []
    for item in raw:
        text = item.get("text") or item.get("content") or ""
        if len(text) > 60:
            text_samples.append(text[:1500])
        if len(text_samples) >= args.num_calibration_samples * 2:
            break

    n_multimodal = int(args.num_calibration_samples * args.multimodal_calib_ratio)
    n_text = args.num_calibration_samples - n_multimodal
    selected_texts = random.sample(text_samples, min(n_text, len(text_samples)))

    result = []

    # Text-only samples
    for txt in selected_texts:
        msgs = [{"role": "user", "content": [{"type": "text", "text": txt[:1000]}]}]
        enc = processor.apply_chat_template(
            msgs,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=False,
        )
        result.append(enc)

    # Multimodal samples (use sample images from Docmatix or a fallback solid-color PNG)
    if n_multimodal > 0:
        print(f"[calib] building {n_multimodal} multimodal calibration samples …")
        try:
            from datasets import load_dataset as _ld
            docmatix = _ld("HuggingFaceM4/Docmatix", split="train", streaming=True)
            mm_images = []
            for item in docmatix:
                imgs = item.get("images") or []
                if imgs:
                    mm_images.append(imgs[0])
                if len(mm_images) >= n_multimodal:
                    break
        except Exception as e:
            print(f"[calib] Docmatix unavailable ({e}), using placeholder images")
            mm_images = []

        prompts_vi = [
            "Mô tả biểu đồ này.",
            "Trích xuất thông tin từ bảng.",
            "Phân tích số liệu trong ảnh.",
            "Đọc và tóm tắt nội dung ảnh.",
        ]
        for i in range(n_multimodal):
            img = mm_images[i] if i < len(mm_images) else _make_placeholder_image()
            prompt = prompts_vi[i % len(prompts_vi)]
            content = [{"type": "image", "image": img}, {"type": "text", "text": prompt}]
            msgs = [{"role": "user", "content": content}]
            try:
                enc = processor.apply_chat_template(
                    msgs,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=False,
                )
                result.append(enc)
            except Exception as e:
                print(f"[calib] skipping multimodal sample {i}: {e}")

    print(f"[calib] total samples: {len(result)} ({n_text} text, {n_multimodal} multimodal target)")
    return result


def _make_placeholder_image():
    """Return a tiny solid-color PIL Image as calibration placeholder."""
    from PIL import Image
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


# ─── Recipes ─────────────────────────────────────────────────────────────────

def _w4a16_recipe(model):
    from llmcompressor.modifiers.quantization import GPTQModifier

    # Detect decoder layer class name at runtime for sequential_targets
    layer_class = _detect_decoder_layer(model)
    print(f"[recipe] sequential_targets={layer_class!r}")

    return [
        GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            sequential_targets=[layer_class] if layer_class else [],
            ignore=[
                "re:.*lm_head",
                "re:.*vision_tower.*",
                "re:.*visual.*",
                "re:.*multi_modal_projector.*",
                "re:.*mm_projector.*",
                "re:.*embed_tokens",
                "re:.*gate$",
                "re:.*router.*",
            ],
            dampening_frac=0.1,
            group_size=128,
        )
    ]


def _fp8_recipe():
    from llmcompressor.modifiers.quantization import QuantizationModifier

    return QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=[
            "re:.*lm_head",
            "re:.*vision_tower.*",
            "re:.*multi_modal_projector.*",
            "re:.*gate$",
        ],
    )


def _detect_decoder_layer(model) -> str | None:
    """Return the transformer block class name to use as sequential_target."""
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if "DecoderLayer" in cls or "TransformerBlock" in cls:
            return cls
    return None


# ─── Quantize ─────────────────────────────────────────────────────────────────

def _run_w4a16(model, processor, dataset, args, output_dir: str):
    from llmcompressor import oneshot

    recipe = _w4a16_recipe(model)
    print(f"[quant] running W4A16 GPTQ oneshot (samples={len(dataset)}, seq_len={args.max_seq_length}) …")
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(dataset),
    )
    _save(model, processor, output_dir)


def _run_fp8(model, processor, output_dir: str):
    from llmcompressor import oneshot

    recipe = _fp8_recipe()
    print("[quant] running FP8-dynamic (data-free) …")
    oneshot(model=model, dataset=None, recipe=recipe, num_calibration_samples=0)
    _save(model, processor, output_dir)


def _save(model, processor, output_dir: str):
    print(f"[save] writing to {output_dir} …")
    model.save_pretrained(output_dir, save_compressed=True)
    processor.save_pretrained(output_dir)
    print(f"[save] done → {output_dir}")


# ─── Sanity check ─────────────────────────────────────────────────────────────

def _sanity_check(output_dir: str):
    """Reload quantized model and run one text + one image generation."""
    print("[sanity] reloading quantized model …")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch

    proc = AutoProcessor.from_pretrained(output_dir, trust_remote_code=True)
    mdl = AutoModelForImageTextToText.from_pretrained(
        output_dir, dtype="auto", device_map="auto", trust_remote_code=True
    )
    mdl.eval()

    # Text-only
    msgs_text = [{"role": "user", "content": [{"type": "text", "text": "GDP là gì?"}]}]
    enc = proc.apply_chat_template(msgs_text, tokenize=True, return_tensors="pt", add_generation_prompt=True, return_dict=True)
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl.generate(**enc, max_new_tokens=32)
    text_out = proc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[sanity] text response: {text_out!r}")
    assert len(text_out) > 0, "text generation returned empty output"

    # Image + text
    try:
        from PIL import Image
        img = Image.new("RGB", (128, 128), color=(100, 149, 237))
        msgs_img = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Ảnh này màu gì?"}]}]
        enc_img = proc.apply_chat_template(msgs_img, tokenize=True, return_tensors="pt", add_generation_prompt=True, return_dict=True)
        enc_img = {k: v.to(mdl.device) for k, v in enc_img.items()}
        with torch.no_grad():
            out_img = mdl.generate(**enc_img, max_new_tokens=32)
        img_out = proc.decode(out_img[0][enc_img["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"[sanity] image response: {img_out!r}")
        assert len(img_out) > 0, "image generation returned empty output"
    except Exception as e:
        print(f"[sanity] image check skipped: {e}")

    print("[sanity] ✓ quantized model looks healthy")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    model_id = args.model_id
    scheme = args.scheme
    basename = Path(model_id).name
    suffix = "W4A16-G128" if scheme == "w4a16" else "FP8-Dynamic"
    output_dir = args.output_dir or f"{basename}-{suffix}"

    print(f"[init] model={model_id}  scheme={scheme}  output={output_dir}")
    print(f"[init] preset: vast.ai RTX 3090 (device_map=auto, dtype=bfloat16)")

    # Load model + processor
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("[load] loading processor …")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    print("[load] loading model (this may take a few minutes) …")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("[load] model loaded")

    if scheme == "w4a16":
        dataset = _build_calibration_dataset(processor, args)
        try:
            _run_w4a16(model, processor, dataset, args, output_dir)
        except Exception as exc:
            print(f"[quant] W4A16 failed: {exc}")
            if args.auto_fallback:
                print("[quant] auto-fallback → FP8-dynamic")
                fp8_dir = output_dir.replace("W4A16-G128", "FP8-Dynamic")
                output_dir = fp8_dir
                _run_fp8(model, processor, output_dir)
            else:
                print("[quant] rerun with --auto-fallback or use --scheme fp8-dynamic")
                sys.exit(1)
    else:
        _run_fp8(model, processor, output_dir)

    if args.sanity_check:
        _sanity_check(output_dir)

    if args.push_to_hub:
        hub_id = args.hub_id or f"{os.environ.get('HF_USERNAME', 'user')}/{Path(output_dir).name}"
        print(f"[hub] pushing to {hub_id} …")
        model_reload = AutoModelForImageTextToText.from_pretrained(
            output_dir, dtype="auto", device_map="cpu", trust_remote_code=True
        )
        processor_reload = AutoProcessor.from_pretrained(output_dir, trust_remote_code=True)
        model_reload.push_to_hub(hub_id)
        processor_reload.push_to_hub(hub_id)
        print(f"[hub] ✓ pushed → https://huggingface.co/{hub_id}")

    print(f"\n✓ Quantization complete: {output_dir}")
    print("Next: vllm serve", output_dir, "--dtype bfloat16 --limit-mm-per-prompt '{\"image\":4}' --max-model-len 32768 --port 8004")


if __name__ == "__main__":
    main()
