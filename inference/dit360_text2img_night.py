"""
Generate a brand-new night-time panorama from text only, using DiT360's
text-to-image pipeline (same code path as the repo's inference.py) - no input
photo, for comparison against dit360_edit_no_mask.py's edited-from-a-photo
result.

Reuses the same cached FLUX.1-dev weights downloaded by
dit360_edit_no_mask.py. Uses CPU offload since the 16GB GPU can't hold the
full fp16 pipeline at once.
"""
import torch

from src.pipeline import DiT360Pipeline

device = torch.device("cuda:0")

prompt = (
    "This is a panorama. A quiet residential street in Singapore at night, "
    "low-rise apartment buildings and terraced houses with red roofs, "
    "streetlights and windows glowing warm yellow, dark navy sky, wet "
    "reflective road."
)

pipe = DiT360Pipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
)
pipe.load_lora_weights("Insta360-Research/DiT360-Panorama-Image-Generation")
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = pipe(
    prompt,
    width=2048,
    height=1024,
    num_inference_steps=28,
    guidance_scale=2.8,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]

image.save("inference/outputs/dit360_text2img_night.png")
print("Saved inference/outputs/dit360_text2img_night.png")
