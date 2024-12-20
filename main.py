import os
import torch
from model import DiffMorpherPipeline


os.makedirs("./saved", exist_ok=True)

pipeline = DiffMorpherPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float32)
pipeline.to("cuda")

images = pipeline(
    use_lora=True,
    img_path_0="./gif_diffusion/photos/house0.jpg",
    img_path_1="./gif_diffusion/photos/house1.jpg",
    prompt_0="simple house",
    prompt_1="Christmas house",
    lamd=0.6,  # Lambda for self-attention replacement
    output_path="./saved",
    num_frames=16,
    save_intermediates=False,
)

images[0].save(f"./saved/houses.gif", save_all=True, append_images=images[1:], duration=300, loop=0)
