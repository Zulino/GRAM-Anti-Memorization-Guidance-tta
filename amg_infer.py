import torch
import os
import random
import numpy as np
import logging
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Create a logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate a unique log file name with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/inference_{timestamp}.log"

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torchaudio
import torch.nn.functional as F
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference import amg_generation
from stable_audio_tools.inference.amg_generation import my_generate_diffusion_cond
import shutil

DEBUG_RUN_ID = 1

debug_dir = f"./debug_run_{DEBUG_RUN_ID}"
if os.path.exists(debug_dir):
    shutil.rmtree(debug_dir)
os.makedirs(debug_dir, exist_ok=True)


device = "cuda:1" if torch.cuda.is_available() else "cpu"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

# Download model
logger.info("Loading model...")
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0", )
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
logger.info("Model loaded.")
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
c1 = 0
c2 = 0
c3 = 0
c_gram = 1000.0
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

logger.info(f"--- STARTING DEBUG RUN {DEBUG_RUN_ID} ---")
logger.info(f"Saving debug tensors to: {debug_dir}")

# Get the logger
logger = logging.getLogger()

logger.info(f"Starting generation with seed: {seed}")
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
    gram_start_step=20,
    gram_use_normalized=False,
    lambda_min=lambda_min,
    lambda_max=lambda_max,
    seed=seed,
    logger=logger
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

logging.info(f"Cosine similarity vs ID 5131: {cosine_sim:.6f}")

output_to_save = output_normalized.mul(32767).to(torch.int16).cpu()


torchaudio.save(audio_path, output_to_save, sample_rate)
logging.info(f"Saved: {audio_path}")
logger.info(f"--- FINISHED DEBUG RUN {DEBUG_RUN_ID} ---")