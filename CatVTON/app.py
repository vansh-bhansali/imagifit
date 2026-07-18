import argparse
import os
from datetime import datetime

import gradio as gr
import gradio_client.utils as _gc_utils
import numpy as np
import torch

# gradio_client's get_type()/_json_schema_to_python_type() assume every JSON-schema
# node is a dict, but a bare `additionalProperties: true` node is a valid bool schema
# and crashes both ("argument of type 'bool' is not iterable" / APIInfoParseError),
# which breaks the app's root page on load.
_original_get_type = _gc_utils.get_type
def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _original_get_type(schema)
_gc_utils.get_type = _patched_get_type

_original_json_schema_to_python_type = _gc_utils._json_schema_to_python_type
def _patched_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    return _original_json_schema_to_python_type(schema, defs)
_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker, vis_mask
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPVisionModelWithProjection
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",  # Change to a copy repo as runawayml delete original repo
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


args = parse_args()

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

repo_path = snapshot_download(repo_id=args.resume_path)
# Pipeline
weight_dtype = init_weight_dtype(args.mixed_precision)
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    args.base_model_path,
    torch_dtype=weight_dtype,
    safety_checker=None,
).to(device)

# Load IP-Adapter from local cache to prevent hangs/downloads
ip_adapter_dir = snapshot_download(repo_id="h94/IP-Adapter", local_files_only=True)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    os.path.join(ip_adapter_dir, "models/image_encoder"),
    torch_dtype=weight_dtype,
    local_files_only=True,
).to(device)

pipeline.load_ip_adapter(
    ip_adapter_dir, 
    subfolder="models", 
    weight_name="ip-adapter_sd15.bin",
    image_encoder=image_encoder,
    local_files_only=True,
)
pipeline.set_ip_adapter_scale(0.7)

# Decode the VAE in slices/tiles so peak activation memory stays low at the
# cost of slightly slower decode — needed to fit in 18GB unified memory.
pipeline.vae.enable_slicing()
pipeline.vae.enable_tiling()
# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device,
)

def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):
    person_image, mask = person_image["background"], person_image["layers"][0]
    mask = Image.open(mask).convert("L")
    if len(np.unique(np.array(mask))) == 1:
        mask = None
    else:
        mask = np.array(mask)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        generator = torch.Generator(device=device).manual_seed(seed)

    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))
    
    # Process mask
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
    else:
        mask = automasker(
            person_image,
            cloth_type
        )['mask']
        mask = resize_and_crop(mask, (args.width, args.height))
    mask = mask_processor.blur(mask, blur_factor=9)

    # Inference
    # try:
    result_image = pipeline(
        prompt="photorealistic high-quality fashion catalog photo of the clothing, highly detailed fabric texture, realistic shadows and soft folds, perfectly fitted garment, seamless blending",
        negative_prompt="deformed clothing, warped patterns, generic design, lowres, blurry, bad anatomy",
        image=person_image,
        mask_image=mask,
        ip_adapter_image=cloth_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    # except Exception as e:
    #     raise gr.Error(
    #         "An error occurred. Please try again later: {}".format(e)
    #     )
    # MPS's caching allocator holds freed activation memory indefinitely,
    # so RSS keeps growing across generations unless we release it.
    if device == "mps":
        torch.mps.empty_cache()
    
    # Post-process
    # Composite the generated result back onto the original person image to guarantee
    # that all unmasked areas (face, hands, background, body shape) remain 100% identical.
    mask_gray = mask.convert("L")
    final_image = Image.composite(result_image, person_image, mask_gray)

    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, masked_person, cloth_image, final_image], 1, 4)
    save_result_image.save(result_save_path)
    if show_type == "result only":
        return final_image
    else:
        width, height = person_image.size
        if show_type == "input & result":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person , cloth_image], 3, 1)
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(final_image, (condition_width + 5, 0))
    return new_result_image


def person_example_fn(image_path):
    return image_path

HEADER = """
<h1 style="text-align: center;"> 👕 Stable Diffusion Inpainting + IP-Adapter Virtual Try-On </h1>
<div style="text-align: center; color: #808080;">
  Powered by Stable Diffusion v1.5 Inpainting & IP-Adapter on MPS
</div>
<br>
"""

def app_gradio():
    with gr.Blocks(title="Stable Diffusion Try-On") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                with gr.Row():
                    image_path = gr.Image(
                        type="filepath",
                        interactive=True,
                        visible=False,
                    )
                    person_image = gr.ImageEditor(
                        interactive=True, label="Person Image", type="filepath"
                    )

                with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image = gr.Image(
                            interactive=True, label="Condition Image", type="filepath"
                        )
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">Two ways to provide Mask:<br>1. Upload the person image and use the `🖌️` above to draw the Mask (higher priority)<br>2. Select the `Try-On Cloth Type` to generate automatically </span>'
                        )
                        cloth_type = gr.Radio(
                            label="Try-On Cloth Type",
                            choices=["upper", "lower", "overall"],
                            value="upper",
                        )


                submit = gr.Button("Submit")
                gr.Markdown(
                    '<center><span style="color: #FF0000">!!! Click only Once, Wait for Delay !!!</span></center>'
                )
                
                gr.Markdown(
                    '<span style="color: #808080; font-size: small;">Advanced options can adjust details:<br>1. `Inference Step` may enhance details;<br>2. `CFG` is highly correlated with saturation;<br>3. `Random seed` may improve pseudo-shadow.</span>'
                )
                with gr.Accordion("Advanced Options", open=False):
                    num_inference_steps = gr.Slider(
                        label="Inference Step", minimum=10, maximum=100, step=5, value=50
                    )
                    # Guidence Scale
                    guidance_scale = gr.Slider(
                        label="CFG Strenth", minimum=0.0, maximum=7.5, step=0.5, value=2.5
                    )
                    # Random Seed
                    seed = gr.Slider(
                        label="Seed", minimum=-1, maximum=10000, step=1, value=42
                    )
                    show_type = gr.Radio(
                        label="Show Type",
                        choices=["result only", "input & result", "input & mask & result"],
                        value="input & mask & result",
                    )

            with gr.Column(scale=2, min_width=500):
                result_image = gr.Image(interactive=False, label="Result")
                with gr.Row():
                    # Photo Examples
                    root_path = "resource/demo/example"
                    with gr.Column():
                        men_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "person", "men", _)
                                for _ in os.listdir(os.path.join(root_path, "person", "men"))
                            ],
                            examples_per_page=4,
                            inputs=image_path,
                            label="Person Examples ①",
                        )
                        women_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "person", "women", _)
                                for _ in os.listdir(os.path.join(root_path, "person", "women"))
                            ],
                            examples_per_page=4,
                            inputs=image_path,
                            label="Person Examples ②",
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">*Person examples come from the demos of <a href="https://huggingface.co/spaces/levihsu/OOTDiffusion">OOTDiffusion</a> and <a href="https://www.outfitanyone.org">OutfitAnyone</a>. </span>'
                        )
                    with gr.Column():
                        condition_upper_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "upper", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "upper"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Upper Examples",
                        )
                        condition_overall_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "overall", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "overall"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Overall Examples",
                        )
                        condition_person_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "person", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "person"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Reference Person Examples",
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">*Condition examples come from the Internet. </span>'
                        )

            image_path.change(
                person_example_fn, inputs=image_path, outputs=person_image
            )

            submit.click(
                submit_function,
                [
                    person_image,
                    cloth_image,
                    cloth_type,
                    num_inference_steps,
                    guidance_scale,
                    seed,
                    show_type,
                ],
                result_image,
            )
    demo.queue().launch(share=True, show_error=True)


if __name__ == "__main__":
    app_gradio()
