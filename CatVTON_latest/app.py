import argparse
import base64
import os
import uuid
from datetime import datetime

import gradio as gr
import gradio_client.utils as _gc_utils
import numpy as np
import torch
import threading
import urllib.request
import subprocess
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# gradio_client's get_type()/_json_schema_to_python_type() assume every JSON-schema
# node is a dict, but a bare `additionalProperties: true` node is a valid bool schema
# and crashes both ("argument of type 'bool' is not iterable"), which 500s the
# Gradio root page on every load (routes.py calls api_info() unconditionally).
# Upstream bug in gradio_client==1.3.0 — see PROJECT_NOTES.md "Issues Fixed" #4.
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
pipeline.enable_attention_slicing()
# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device,
)

# Serialize generation: both the Flask thread and the Gradio worker call into
# generation, and two concurrent diffusion runs would exhaust MPS memory.
generation_lock = threading.Lock()


def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):
    with generation_lock:
        return _generate_tryon(
            person_image,
            cloth_image,
            cloth_type,
            num_inference_steps,
            guidance_scale,
            seed,
            show_type,
        )


def _generate_tryon(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):
    if isinstance(person_image, dict):
        person_image_path, mask_path = person_image["background"], person_image["layers"][0]
        mask = Image.open(mask_path).convert("L")
        if len(np.unique(np.array(mask))) == 1:
            mask = None
        else:
            mask = np.array(mask)
            mask[mask > 0] = 255
            mask = Image.fromarray(mask)
        person_image = person_image_path
    else:
        # person_image is a string path from Flask webhook
        mask = None

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        generator = torch.Generator(device=device).manual_seed(seed)

    if person_image is None:
        raise gr.Error("Please provide a person image before submitting.")
    if cloth_image is None:
        raise gr.Error("Please provide a clothing image before submitting.")
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

    # MPS's caching allocator holds freed activation memory indefinitely,
    # so RSS keeps growing across generations unless we release it.
    if device == "mps":
        torch.mps.empty_cache()
    
    # Post-process — ensure all images are exactly the same size before compositing.
    # The SD pipeline VAE encode/decode round-trip and mask_processor.blur() can alter
    # dimensions slightly, which would cause Image.composite to crash with a size mismatch.
    person_size = person_image.size  # (width, height)
    result_image = result_image.resize(person_size, Image.LANCZOS)
    mask_gray = mask.convert("L").resize(person_size, Image.LANCZOS)
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

# --- Flask Server Integration ---
flask_app = Flask(__name__)
CORS(flask_app)  # Enable CORS for all routes so React can connect
UPLOAD_FOLDER = 'received_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@flask_app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.abspath(UPLOAD_FOLDER), filename)

# Persist the registered webhook across server restarts, so the client doesn't
# have to re-handshake every time this server relaunches. A stale URL is
# harmless: delivery retries fail gracefully and the client re-registers.
WEBHOOK_STATE_FILE = "registered_webhook.txt"
CLIENT_WEBHOOK_URL = None
if os.path.exists(WEBHOOK_STATE_FILE):
    try:
        CLIENT_WEBHOOK_URL = open(WEBHOOK_STATE_FILE).read().strip() or None
        if CLIENT_WEBHOOK_URL:
            print(f"🔗 Restored client webhook from previous session: {CLIENT_WEBHOOK_URL}")
    except OSError:
        pass
# Set by start_tunnel() once localtunnel reports its URL. May be a random
# subdomain if "imagifit-api-server" was still held by a dead tunnel, so the
# client must never hardcode it — we send full image URLs in the webhook payload.
PUBLIC_TUNNEL_URL = None

@flask_app.route('/register', methods=['POST'])
def register_webhook():
    global CLIENT_WEBHOOK_URL
    data = request.json
    if not data or 'client_webhook_url' not in data:
        return jsonify({"error": "Missing client_webhook_url"}), 400
    CLIENT_WEBHOOK_URL = data['client_webhook_url']
    try:
        with open(WEBHOOK_STATE_FILE, "w") as f:
            f.write(CLIENT_WEBHOOK_URL)
    except OSError as e:
        print(f"⚠️ Could not persist webhook URL: {e}")
    print(f"🔗 Registered Client Webhook: {CLIENT_WEBHOOK_URL}")
    return jsonify({"success": True, "message": "Webhook registered successfully"})

def process_and_send(client_path, clothing_path, clothing_id, cloth_type, job_id):
    try:
        print(f"⏳ Running AI generation for {job_id} in background (type: upper)...")
        result_img = submit_function(
            client_path, clothing_path,
            "upper", 50, 2.5, 42, "result only"
        )
        
        date_str = time.strftime("%Y%m%d%H%M%S")
        final_save_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, f"result_{date_str}.png"))
        result_img.save(final_save_path)
        
        import requests
        delivered = False
        for attempt in range(1, 11):
            # Fetch the webhook URL dynamically! If the client restarts their tunnel mid-generation,
            # this ensures we send it to their NEW tunnel url, not the dead one.
            current_webhook = CLIENT_WEBHOOK_URL
            
            if not current_webhook:
                print(f"⚠️ No webhook registered! Result saved locally at {final_save_path}")
                return
                
            print(f"📤 Sending result for {job_id} to {current_webhook} (attempt {attempt}/10)...")
            image_name = os.path.basename(final_save_path)
            payload = {
                "job_id": job_id,
                "clothing_id": clothing_id,
                "success": True,
                "generatedImageName": image_name,
                "generatedImageUrl": f"{PUBLIC_TUNNEL_URL}/images/{image_name}" if PUBLIC_TUNNEL_URL else None,
                "generatedImageLocalUrl": f"http://127.0.0.1:5050/images/{image_name}",
            }
            try:
                response = requests.post(
                    current_webhook,
                    json=payload,
                    headers={
                        'Bypass-Tunnel-Reminder': 'true',
                        'User-Agent': 'Imagifit-Server/1.0'
                    },
                    timeout=30
                )
                if 200 <= response.status_code < 300:
                    print(f"✅ Webhook delivered for {job_id} (attempt {attempt}, HTTP {response.status_code})")
                    delivered = True
                    break
                print(f"⚠️ Webhook attempt {attempt}/10 got HTTP {response.status_code} — "
                      f"client did not accept it. Body: {response.text[:200]!r}")
            except Exception as req_e:
                print(f"❌ Webhook attempt {attempt}/10 failed: {req_e}")
            time.sleep(2)
            
        if not delivered:
            print(f"❌ Webhook delivery FAILED for {job_id} after 10 attempts. "
                  f"Result is still available at {payload['generatedImageLocalUrl']} "
                  f"(and {payload['generatedImageUrl']} via tunnel).")

    except Exception as e:
        print(f"❌ Error during generation for {job_id}: {e}")
        current_webhook = CLIENT_WEBHOOK_URL
        if current_webhook:
            import requests
            payload = {
                "job_id": job_id,
                "clothing_id": clothing_id,
                "success": False,
                "error": str(e)
            }
            try:
                requests.post(
                    current_webhook, 
                    json=payload, 
                    headers={
                        'Bypass-Tunnel-Reminder': 'true',
                        'User-Agent': 'Imagifit-Server/1.0'
                    },
                    timeout=10
                )
            except:
                pass

def cleanup_old_files():
    """Deletes files in UPLOAD_FOLDER that are older than 5 minutes to prevent the folder from growing indefinitely."""
    now = time.time()
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                if os.stat(file_path).st_mtime < now - 300: # 5 minutes
                    try:
                        os.remove(file_path)
                    except:
                        pass
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

@flask_app.route('/process', methods=['POST'])
def process_image():
    cleanup_old_files()
    print("\n--- New Request Received! ---")
    print(f"Files received: {list(request.files.keys())}")
    print(f"Form data received: {list(request.form.keys())}")
    
    if 'clientImage' not in request.files or 'clothingImage' not in request.files:
        print("❌ Error: Missing clientImage or clothingImage! Returning 400 error.")
        return jsonify({"error": "Missing clientImage or clothingImage in request"}), 400
        
    client_image = request.files['clientImage']
    clothing_image = request.files['clothingImage']
    clothing_id = request.form.get('clothingId', 'unknown_clothing')
    cloth_type = request.form.get('clothType', 'upper')

    if client_image.filename == '' or clothing_image.filename == '':
        return jsonify({"error": "Empty filename for one of the images"}), 400

    # Prefix with a unique job id so concurrent requests uploading files with
    # the same name (e.g. "photo.jpg") don't overwrite each other mid-generation.
    # secure_filename can also return "" for non-ASCII names, so fall back.
    job_id = f"job_{uuid.uuid4().hex[:8]}"

    client_filename = f"{job_id}_{secure_filename(client_image.filename) or 'client.png'}"
    client_save_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, client_filename))
    client_image.save(client_save_path)

    clothing_filename = f"{job_id}_{secure_filename(clothing_image.filename) or 'clothing.png'}"
    clothing_save_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, clothing_filename))
    clothing_image.save(clothing_save_path)
    
    print(f"✅ Received Client Image: {client_filename}")
    print(f"✅ Received Clothing Image: {clothing_filename} (ID: {clothing_id})")
    
    # Spawn background thread and return immediately
    print(f"🔄 Spawning background generation for {job_id} (type: {cloth_type})...")
    threading.Thread(
        target=process_and_send, 
        args=(client_save_path, clothing_save_path, clothing_id, cloth_type, job_id)
    ).start()

    return jsonify({
        "success": True,
        "message": "Processing started",
        "job_id": job_id
    }), 202

def get_public_ip():
    try:
        url = 'https://ifconfig.me/ip'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read().decode('utf-8').strip()
    except Exception as e:
        return f"Unable to fetch public IP: {e}"

def start_tunnel():
    global PUBLIC_TUNNEL_URL
    print("🌍 Fetching public IP address for Localtunnel authorization password...")
    public_ip = get_public_ip()
    print(f"🔑 Your Localtunnel Password (Public IP): {public_ip}")
    print("🚀 Starting localtunnel on port 5050...")
    
    try:
        process = subprocess.Popen(
            ['npx', '-y', 'localtunnel', '--port', '5050', '--subdomain', 'imagifit-api-server'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            if "your url is:" in line:
                url = line.strip().replace("your url is: ", "")
                PUBLIC_TUNNEL_URL = url
                if "imagifit-api-server" not in url:
                    print("⚠️  WARNING: did NOT get the expected 'imagifit-api-server' subdomain "
                          "(old tunnel likely still holds it). Clients hardcoding the URL will break!")
                print("\n==================================================")
                print(f"🌐 Public Tunnel URL: {url}")
                print(f"⚙️  API Endpoint:     {url}/process")
                print(f"🔑 Password/IP:       {public_ip}")
                print("==================================================\n")
            else:
                print(f"[Tunnel] {line.strip()}")
    except Exception as e:
        print(f"❌ Failed to start localtunnel: {e}")

def run_flask():
    print("AI Server is listening on port 5050...")
    flask_app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)
# --- End Flask Integration ---

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
    # Start the Flask API and Localtunnel in background threads
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=start_tunnel, daemon=True).start()
    
    # Start the Gradio UI on the main thread
    app_gradio()
