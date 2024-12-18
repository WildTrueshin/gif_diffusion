import os
import torch
from model import DiffMorpherPipeline


os.makedirs("./saved", exist_ok=True)

pipeline = DiffMorpherPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float32)
pipeline.to("cuda")

images = pipeline(
    img_path_0="./photos/arsen.jpg",
    img_path_1="./photos/gimli.jpg",
    prompt_0="A hairy man with beard",
    prompt_1="A hairy men in helmet with beard",
    lamd=0.6,  # Lambda for self-attention replacement
    output_path="./saved",
    num_frames=16,
    save_intermediates=False,
)

images[0].save(f"./saved/output.gif", save_all=True, append_images=images[1:], duration=200, loop=0)
