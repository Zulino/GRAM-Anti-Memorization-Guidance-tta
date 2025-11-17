import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torchaudio
import torch.nn.functional as F
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference import amg_generation
from stable_audio_tools.inference.amg_generation import my_generate_diffusion_cond


device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0", )
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

# Enable float16 and move model to device
#model.half()
model = model.to(device)
#model.pretransform.to("cpu", dtype=torch.float32)

# Ensure output directory exists, and skip saving if the file already exists
audio_dir = "./"
os.makedirs(audio_dir, exist_ok=True)
audio_path = os.path.join(audio_dir, "audio.wav")

total_duration = 1.7143310657596371  # seconds
cfg_scale = 7
c1 = 4
c2 = 3
c3 = 100
c_gram = 100.0
lambda_min = 0.7
lambda_max = 0.8
denoising_steps = 100

#prompt = "An acoustic drum loop, 110 bpm"
prompt = "\"ATTACK loop 140 bpm-00.wav\" till \"ATTACK loop 140 bpm-31.wav\"\r\nare all part of the \"ATTACK LOOP 6\" sample package and belong together\r\nas they are all variations on the same 1 measure 4/4 140 bpm drumloop. The loop has a techno-trance\r\nfeel. The first four loops (00 till 03) contain some variations of the\r\npure drumloop, where 00 is the most minimal and 03 the fullest. All\r\nother variations add other sound effects, some of them being sounds\r\nwith a certain pitch, mostly C. These loop are suitable for your trance\r\nand techno productions. They were created using the Waldorf Attack VSTi within Cubase SX. Mastering (EQ, Stereo Enhancer, Multi-Band expand/compress/limit, dither, fades at start and/or end) done within Wavelab.\r\n"

# Set up text and timing conditioning
conditioning = [{
    "prompt": prompt,
    "seconds_start": 0, 
    "seconds_total": total_duration
}]
negative_conditioning = [{
    "prompt": "",
    "seconds_start": 0,
    "seconds_total": total_duration
}]

output = my_generate_diffusion_cond(
    model,
    steps=denoising_steps,
    cfg_scale=cfg_scale,
    conditioning=conditioning,
    negative_conditioning=negative_conditioning,
    sample_size=sample_size,
    sample_rate=sample_rate,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="my-dpmpp-3m-sde",
    device=device,
    c1=c1,
    c2=c2,
    c3=c3,
    c_gram=c_gram,
    lambda_min=lambda_min,
    lambda_max=lambda_max,
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

num_samples = int(conditioning[0]["seconds_total"] * sample_rate)
output = output[:, :num_samples]


# Peak normalize, clip, convert to int16, and save to file
output_float = output.to(torch.float32)
peak = output_float.abs().max().clamp_min(1e-6)
output_normalized = (output_float / peak).clamp(-1, 1)

# Compute cosine similarity between generated audio embedding and entry 5131
CLAP = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, device=device)
CLAP.load_ckpt()
with torch.no_grad():
    mono_for_embed = output_normalized.mean(dim=0, keepdim=True).to(device)
    generated_embedding = CLAP.get_audio_embedding_from_data(x=mono_for_embed, use_tensor=True)[0]
    target_embedding = torch.tensor(
        amg_generation.data["5131"]["embedding"],
        dtype=generated_embedding.dtype,
        device=device,
    )
    generated_embedding = F.normalize(generated_embedding, dim=0)
    target_embedding = F.normalize(target_embedding, dim=0)
    cosine_sim = torch.dot(generated_embedding, target_embedding).item()

print(f"[INFO] Cosine similarity vs ID 5131: {cosine_sim:.6f}")

output_to_save = output_normalized.mul(32767).to(torch.int16).cpu()


torchaudio.save(audio_path, output_to_save, sample_rate)
print(f"[INFO] Saved: {audio_path}")