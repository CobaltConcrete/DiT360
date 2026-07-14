"""
Edit an image with FLUX.2 via Hugging Face Inference Providers (hosted API, fal).

No local GPU/weights needed - this calls a hosted provider through your Hugging
Face account and uses your monthly inference credits / pay-as-you-go billing.
Make sure you're logged in first:

    hf auth login

Usage:
    python inference/hf_inference_edit.py \
        --image inference/inputs/Street_View_360_1.jpg \
        --prompt "Change this scene to nighttime..." \
        --output inference/outputs/night.png
"""

import argparse
import json

from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated.types import ImageToImageTargetSize
from huggingface_hub.inference._providers.fal_ai import FalAIImageToImageTask, FalAIQueueTask


def _patch_debug_response():
    """Print the raw provider response before FalAIImageToImageTask tries to parse
    it into ["images"][0]["url"], so we can see its actual shape if that fails."""
    real_get_response = FalAIQueueTask.get_response

    def debug_get_response(self, response, request_params=None):
        result = real_get_response(self, response, request_params)
        print("\n=== RAW FAL RESPONSE ===")
        print(json.dumps(result, indent=2, default=str)[:3000])
        print("=== END RAW RESPONSE ===\n")
        return result

    FalAIQueueTask.get_response = debug_get_response


def _patch_flux2_image_urls_field():
    """huggingface_hub's generic fal image-to-image payload sends a single
    "image_url" string, but fal's FLUX.2 endpoint requires "image_urls" (a list) -
    it supports multi-reference editing. Rename the field after the library
    builds its payload."""
    real_prepare_payload = FalAIImageToImageTask._prepare_payload_as_dict

    def patched_prepare_payload(self, inputs, parameters, provider_mapping_info):
        payload = real_prepare_payload(self, inputs, parameters, provider_mapping_info)
        if "image_url" in payload:
            payload["image_urls"] = [payload.pop("image_url")]
        return payload

    FalAIImageToImageTask._prepare_payload_as_dict = patched_prepare_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--prompt", required=True, help="Edit instruction / text prompt")
    parser.add_argument("--output", default="inference/outputs/output.png", help="Path to save the result")
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-dev", help="Model repo id")
    parser.add_argument("--provider", default="fal-ai", help="Inference provider")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--negative_prompt", default=None, help="What to avoid in the output")
    parser.add_argument("--width", type=int, default=None, help="Output width in pixels")
    parser.add_argument("--height", type=int, default=None, help="Output height in pixels")
    parser.add_argument("--token", default=None, help="HF access token (defaults to cached `hf auth login` token)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _patch_debug_response()
    _patch_flux2_image_urls_field()
    client = InferenceClient(provider=args.provider, api_key=args.token)

    with open(args.image, "rb") as f:
        input_image = f.read()

    extra = {}
    if args.num_inference_steps is not None:
        extra["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        extra["guidance_scale"] = args.guidance_scale
    if args.negative_prompt is not None:
        extra["negative_prompt"] = args.negative_prompt
    if args.width is not None and args.height is not None:
        extra["target_size"] = ImageToImageTargetSize(width=args.width, height=args.height)

    image = client.image_to_image(
        input_image,
        prompt=args.prompt,
        model=args.model,
        **extra,
    )

    image.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
