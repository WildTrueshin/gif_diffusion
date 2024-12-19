import torch
from torchinfo import summary
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from transformers import AutoTokenizer, PretrainedConfig
from transformers import CLIPTextModel
from torchvision import transforms
from PIL import Image
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch.nn.functional as F

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

model_path = "stabilityai/stable-diffusion-2-1-base"
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=None)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

unet.requires_grad_(False)
unet.to(device)

lora_rank = 16
unet_lora_attn_procs = {}
for name, attn_processor in unet.attn_processors.items():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]

    unet_lora_attn_procs[name] = LoRAAttnProcessor2_0(
        hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
    )

unet.set_attn_processor(unet_lora_attn_procs)
unet_lora_layers = AttnProcsLayers(unet.attn_processors)

tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", revision=None, use_fast=False)
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", revision=None)

text_encoder.requires_grad_(False)
text_encoder.to(device)

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=False):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

# !wget https://raw.githubusercontent.com/WildTrueshin/gif_diffusion/main/photos/house1.jpg
image = Image.open("house1.jpg").convert("RGB")

prompt = "a hairy man"
with torch.no_grad():
    text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None)
    text_embedding = encode_prompt(
        text_encoder, text_inputs.input_ids, text_inputs.attention_mask, text_encoder_use_attention_mask=False
    )


# initialize latent distribution
image_transforms = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        # transforms.RandomCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
image = image_transforms(image).to(device)
image = image.unsqueeze(dim=0)

noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=None)

vae.requires_grad_(False)
vae.to(device)


lora_steps = 200
lora_lr = 2e-4

params_to_optimize = unet_lora_layers.parameters()
optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=lora_lr,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-08,
)

lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=lora_steps,
    num_cycles=1,
    power=1.0,
)

accelerator = Accelerator(
    gradient_accumulation_steps=1,
    # mixed_precision='fp16'
)
set_seed(0)

# prepare accelerator
unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
optimizer = accelerator.prepare_optimizer(optimizer)
lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

latents_dist = vae.encode(image).latent_dist
for _ in tqdm(range(lora_steps), desc="Training LoRA..."):
    unet.train()
    model_input = latents_dist.sample() * vae.config.scaling_factor
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz, channels, height, width = model_input.shape
    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
    timesteps = timesteps.long()

    # Add noise to the model input according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

    # Predict the noise residual
    model_pred = unet(noisy_model_input, timesteps, text_embedding).sample

    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
    accelerator.backward(loss)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

LoraLoaderMixin.save_lora_weights(
    save_directory="./lora",
    unet_lora_layers=unet_lora_layers,
    text_encoder_lora_layers=None,
    weight_name="example_lora_0.ckpt",
    safe_serialization=False,
)

def load_lora(unet, lora_0, lora_1, alpha):
    lora = {}
    for key in lora_0:
        lora[key] = (1 - alpha) * lora_0[key] + alpha * lora_1[key]
    unet.load_attn_procs(lora)
    return unet


alpha = 0.8
lora_0 = torch.load("./lora/example_lora_0.ckpt", map_location="cpu")
lora_1 = torch.load("./lora/example_lora_0.ckpt", map_location="cpu")

lora_alpha = load_lora(unet, lora_0, lora_1, alpha)