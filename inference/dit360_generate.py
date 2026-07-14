"""
Generate a 360 panorama with DiT360 (Insta360-Research/DiT360-Panorama-Image-Generation).

This is a *local* diffusers pipeline (downloads weights to disk, runs on your GPU) -
unlike inference/hf_inference_edit.py, it does not use HF Inference Providers/credits.

Note: per the model card, DiT360 is text-to-image (+ inpainting/outpainting), not a
general image-editing model - it generates a new panorama from a prompt, it does not
take an existing photo and re-light it the way FLUX.2 image-to-image does.

Usage:
    python inference/dit360_generate.py \
        --prompt "Singapore street at night, neon signs, wet reflective streets, ..." \
        --output inference/outputs/dit360_singapore_night.png
"""

import argparse
import inspect

import torch
from diffusers import DiffusionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Text prompt describing the panorama")
    parser.add_argument("--output", default="inference/outputs/dit360_output.png")
    parser.add_argument("--model", default="Insta360-Research/DiT360-Panorama-Image-Generation")
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Offload the pipeline to CPU between steps to reduce peak VRAM (slower)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="DiT360 likely ships a custom pipeline class; enabled by default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    # Only pass kwargs the pipeline's __call__ actually accepts - the exact
    # signature of this custom pipeline isn't documented, so build the call
    # defensively instead of guessing parameter names.
    accepted = set(inspect.signature(pipe.__call__).parameters)
    candidate_kwargs = {
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
    }
    call_kwargs = {k: v for k, v in candidate_kwargs.items() if v is not None and k in accepted}

    skipped = {k for k, v in candidate_kwargs.items() if v is not None and k not in accepted}
    if skipped:
        print(f"Note: pipeline does not accept these params, ignoring: {sorted(skipped)}")

    if args.seed is not None and "generator" in accepted:
        call_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(args.seed)

    image = pipe(args.prompt, **call_kwargs).images[0]
    image.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
