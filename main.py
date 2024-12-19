import os
import torch
from model import DiffMorpherPipeline


os.makedirs("./saved", exist_ok=True)

pipeline = DiffMorpherPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float32)
pipeline.to("cuda")

images = pipeline(
    use_lora=True,
    img_path_0="./gif_diffusion/photos/wave_paint.png",
    img_path_1="./gif_diffusion/photos/wave_real.jpg",
    prompt_0="A picture of wave",
    prompt_1="A real wave",
    lamd=0.6,  # Lambda for self-attention replacement
    output_path="./saved",
    num_frames=20,
    save_intermediates=False,
)

images[0].save(f"./saved/output2.gif", save_all=True, append_images=images[1:], duration=200, loop=0)
