"""
Edit inference/inputs/Street_View_360_1.jpg into a night scene using DiT360's
RF-Inversion editing pipeline, WITHOUT a spatial mask.

Day->night is a global lighting/color change, not a localized edit, so no
region needs to be hard-locked to the source the way editing.py's mask does.
Structure is instead preserved by the RF-Inversion latents themselves, tuned
via `eta` (faithfulness) and the `start_timestep`/`stop_timestep` blending
window (editability).

First run downloads black-forest-labs/FLUX.1-dev in diffusers format
(~34GB), cached under ~/.cache/huggingface afterward. Uses CPU offload since
the 16GB GPU can't hold the full fp16 pipeline at once.
"""
import torch
from PIL import Image

from pa_src.pipeline import RFPanoInversionParallelFluxPipeline

device = torch.device("cuda:0")
dtype = torch.float16

height, width = 1024, 2048
timestep = 50            # denoising / inversion steps
seed = 0
guidance_scale = 2.8
eta = 0.85                # faithfulness vs editability: lower = more freedom to relight
start_timestep = 0.0       # (fraction of steps) eta-blending window start
stop_timestep = 0.6        # window end - turn off early so late steps can relight freely

source_prompt = ""  # RF-Inversion recommends an empty prompt for inversion itself
prompt = (
    "This is a panorama image. A quiet residential street in Singapore during "
    "a sunny day, low-rise apartment buildings and terraced houses with red "
    "roofs, green trees, parked cars, and a clear blue sky with clouds."
)
new_prompt = (
    "This is a panorama image. The same quiet residential street in Singapore "
    "at night, streetlights and windows glowing warm yellow, dark navy night "
    "sky, building lights illuminated."
)

pipe = RFPanoInversionParallelFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=dtype, low_cpu_mem_usage=True
)
pipe.load_lora_weights("Insta360-Research/DiT360-Panorama-Image-Generation")
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

init_image = (
    Image.open("inference/inputs/Street_View_360_1.jpg")
    .convert("RGB")
    .resize((width, height))
)

inverted_latents, image_latents, latent_image_ids = pipe.invert(
    source_prompt=source_prompt,
    image=init_image,
    height=height,
    width=width,
    num_inversion_steps=timestep,
    gamma=1.0,
)

image = pipe(
    [prompt, new_prompt],
    inverted_latents=inverted_latents,
    image_latents=image_latents,
    latent_image_ids=latent_image_ids,
    height=height,
    width=width,
    start_timestep=start_timestep,
    stop_timestep=stop_timestep,
    num_inference_steps=timestep,
    eta=eta,
    guidance_scale=guidance_scale,
    generator=torch.Generator(device=device).manual_seed(seed),
    mask=None,
).images[1]

image.save("inference/outputs/street_view_night_edit.png")
print("Saved inference/outputs/street_view_night_edit.png")
