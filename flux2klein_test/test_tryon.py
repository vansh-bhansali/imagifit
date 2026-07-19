"""
Standalone test script for FLUX.2 [klein] 4B — NOT wired into app.py or the
running CatVTON server. Purpose: see actual output quality for a garment
try-on style edit (person photo + garment photo + instruction prompt) before
deciding whether this model is worth integrating anywhere.

Run with the isolated `flux2klein` conda env — do NOT run with the `catvton`
env, since this pulls diffusers from git main, which conflicts with the
pinned diffusers==0.31.0 that CatVTON needs (see PROJECT_NOTES.md #1).
"""
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

person_image = Image.open(
    "../CatVTON/resource/demo/example/person/men/model_5.png"
).convert("RGB")
garment_image = Image.open(
    "../CatVTON/resource/demo/example/condition/upper/21514384_52353349_1000.jpg"
).convert("RGB")

prompt = (
    "Put the garment shown in the second reference image onto the person "
    "shown in the first reference image. Preserve the person's face, "
    "identity, pose, body proportions, and the original background exactly. "
    "Do not change anything except the clothing region. Match the garment's "
    "exact color, pattern, and texture from the reference photo — do not "
    "invent or alter design details. Photorealistic result."
)

print("Loading pipeline (first run will download ~13GB of weights)...")
pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

print("Running inference...")
result = pipe(
    prompt=prompt,
    image=[person_image, garment_image],
    height=1024,
    width=768,
    num_inference_steps=28,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

out_path = "result_flux2klein.png"
result.save(out_path)
print(f"Saved result to {out_path}")
