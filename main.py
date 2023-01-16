import os
import glob
import base64
from datetime import datetime
import io as PythonIO
import subprocess

from interval_sdk import Interval, IO, action_ctx_var, io_var
from interval_sdk.classes.page import Page
from interval_sdk.classes.layout import Layout

from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline
import torch

interval = Interval(
    os.environ.get("INTERVAL_API_KEY"),
)

page = Page(name="Images")


@page.handle
async def handler(display: IO.Display):
    images = []
    for image_path in glob.glob("/workspace/outputs/*.png"):
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
            images.append(f"data:image/jpeg;base64,{data}")

    return Layout(
        title="Images",
        description="AI generated images that you've previously generated.",
        children=[
            display.grid(
                "Latest images",
                data=images,
                render_item=lambda x: {
                    "image": {
                        "url": x,
                        "aspectRatio": 1,
                    },
                },
                default_page_size=9,
                is_filterable=False,
            ),
        ],
    )


@page.action(slug="generate", name="Generate image")
async def generate(io: IO):

    prompt = await io.input.text(
        "What's the prompt?",
        help_text='To activate your trained images include "triggerword person" in the prompt.',
    )

    ctx = action_ctx_var.get()
    num_steps = 50
    await ctx.loading.start(
        title="Generating image...",
        description="This may take a while on a CPU."
        if not torch.cuda.is_available()
        else "",
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        "/workspace/trained_models/RyanCoppolo"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    image = pipe(
        prompt=prompt,
        negative_prompt="multiple people, ugly, deformed, malformed limbs, low quality, blurry, naked, out of frame",
        num_inference_steps=num_steps,
    ).images[0]

    now = datetime.now()
    image.save(
        f"/workspace/outputs/{now.strftime('%Y-%m-%d_%H-%M-%S')}.png", format="PNG"
    )

    img_bytes = PythonIO.BytesIO()
    image.save(img_bytes, format="PNG")

    await io.display.image(
        "Generated image",
        bytes=img_bytes.getvalue(),
        size="large",
    )

    return "All done!"


interval.routes.add("images", page)


@interval.action
async def train_model(io: IO):

    [_, token] = await io.group(
        io.display.markdown(
            "We're going to train a custom Stable Diffusion model on your images! First we'll download a pre-trained model to build off of. We'll need one of [your huggingface tokens](https://huggingface.co/settings/tokens) to download the base model."
        ),
        io.input.text("Enter your huggingface access token"),
    )

    ctx = action_ctx_var.get()
    await ctx.loading.start(
        title="Downloading model...",
    )

    model_path = hf_hub_download(
        repo_id="runwayml/stable-diffusion-v1-5",
        filename="v1-5-pruned-emaonly.ckpt",
        force_download=True,
        token=token,
    )

    proc = subprocess.Popen(["readlink", "-f", model_path], stdout=subprocess.PIPE)
    real_path = proc.stdout.read()
    subprocess.run(
        [
            "mv",
            real_path.strip(),
            "Dreambooth-Stable-Diffusion/stable-diffusion-v1-5.ckpt",
        ]
    )

    [_, trigger_word] = await io.group(
        io.display.markdown(
            "Model downloaded! Now we'll have you upload images (as soon as we build the I/O method)"
        ),
        io.input.text(
            "What trigger word do you want to train on?",
            help_text="Better to avoid spaces",
            placeholder="MyName",
        ),
    )

    await ctx.loading.start(title="Training model on your images...")

    subprocess.run(
        [
            "python",
            "Dreambooth-Stable-Diffusion/main.py",
            "--base",
            "Dreambooth-Stable-Diffusion/configs/stable-diffusion/v1-finetune_unfrozen.yaml",
            "-t",
            "--actual_resume",
            "Dreambooth-Stable-Diffusion/stable-diffusion-v1-5.ckpt",
            "--reg_data_root",
            "Dreambooth-Stable-Diffusion/regularization_images/person_ddim",
            "-n",
            trigger_word,
            "--gpus",
            "0",
            "--data_root",
            "Dreambooth-Stable-Diffusion/training_images",
            "--max_training_steps",
            "2000",
            "--class_word",
            "person",
            "--token",
            trigger_word,
            "--no-test",
        ]
    )

    await ctx.loading.start(title="Done training! Converting to diffusers format...")

    parent_dir = "/workspace/logs"
    pattern = "training_images*"
    latest_dir = max(glob.glob(os.path.join(parent_dir, pattern)), key=os.path.getmtime)

    subprocess.run(
        [
            "python",
            "diffusers/scripts/convert_stable_diffusion_checkpoint.py",
            "--checkpoint_path",
            f"{latest_dir}/checkpoints/last.ckpt",
            "--dump_path",
            f"/workspace/trained_models/{trigger_word}",
        ]
    )

    return "All done!"


interval.listen()
