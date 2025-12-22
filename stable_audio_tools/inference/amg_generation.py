import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch 
import typing as tp
import k_diffusion as K
#import CLAP.src.laion_clap as laion_clap
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass, field

from .utils import prepare_audio
from .sampling import sample_rf
from .amg_sampling import my_sample_k, make_cond_model_fn

import os, sys, json
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy import fft as scipy_fft

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..','..','..'))  
LOCAL_CLAP = os.path.join(ROOT, 'CLAP', 'src')

# 2) Insert it at the front of sys.path so it wins over the env package
sys.path.insert(0, LOCAL_CLAP)

# 3) Now import exactly your local code:
import laion_clap 



def apply_latent_lowpass_fft(tensor: torch.Tensor, cutoff_ratio: float = 0.25, soft_knee: float = 0.1):
    """
    Applica un Low-Pass filter direttamente nello spazio latente usando FFT.
    
    Args:
        tensor: [Batch, Channels, Time] (es. [1, 64, 1024])
        cutoff_ratio: 0.0 -> 1.0. Percentuale di frequenze da MANTENERE. 
                      0.25 significa: tieni i bassi (primo 25%), taglia il resto.
        soft_knee: Percentuale di transizione per evitare il taglio netto (riduce il ringing).
    """
    # 1. FFT sui Latents (Real FFT, solo frequenze positive)
    # tensor è reale, rfft è più efficiente
    fft_g = torch.fft.rfft(tensor, dim=-1, norm='ortho')
    
    # 2. Creazione della Maschera
    num_freqs = fft_g.shape[-1]
    cutoff_idx = int(num_freqs * cutoff_ratio)
    fade_len = int(num_freqs * soft_knee)
    
    # Inizializza maschera a zeri (tutto tagliato)
    mask = torch.zeros(num_freqs, device=tensor.device, dtype=tensor.dtype)
    
    # Passa-tutto fino al cutoff
    mask[:cutoff_idx] = 1.0
    
    # Dissolvenza (Linear Fade-out) per evitare il "Gibbs Phenomenon"
    if fade_len > 0 and cutoff_idx < num_freqs:
        end_fade = min(cutoff_idx + fade_len, num_freqs)
        actual_fade_len = end_fade - cutoff_idx
        # Crea una rampa che va da 1 a 0
        fade_curve = torch.linspace(1, 0, actual_fade_len, device=tensor.device, dtype=tensor.dtype)
        mask[cutoff_idx:end_fade] = fade_curve

    # 3. Applica Maschera (Broadcasting automatico su Batch e Channels)
    fft_g_filtered = fft_g * mask
    
    # 4. Inverse FFT per tornare al tempo
    # n=tensor.shape[-1] è CRUCIALE per gestire lunghezze dispari/pari correttamente
    filtered_tensor = torch.fft.irfft(fft_g_filtered, n=tensor.shape[-1], dim=-1, norm='ortho')
    
    return filtered_tensor



@dataclass
class GuidanceSpectralCollector:
    """
    Collector for guidance signals directly in LATENT space.
    Does NOT decode to audio, to preserve the modulation frequency analysis.
    """
    enabled: bool = False
    cfg_history: list = field(default_factory=list)  # Stored as latents
    amg_history: list = field(default_factory=list)  # Stored as latents
    sigma_history: list = field(default_factory=list)
    sample_rate_audio: int = 48000
    downsampling_ratio: int = 2048
    
    def reset(self):
        self.cfg_history = []
        self.amg_history = []
        self.sigma_history = []
    
    def set_pretransform(self, pretransform, audio_length: int):
        # We just need metadata, not the decode function
        self.sample_rate_audio = pretransform.sample_rate if hasattr(pretransform, 'sample_rate') else 48000
        self.downsampling_ratio = pretransform.downsampling_ratio if hasattr(pretransform, 'downsampling_ratio') else 2048
    
    @torch.no_grad()
    def collect(self, G_cfg: torch.Tensor, G_amg: torch.Tensor, sigma: float):
        if not self.enabled:
            return
        
        # G_cfg shape: [B, 64, 1024]
        # We assume batch size 1 or take mean. 
        # We keep the Channel dimension to average over it later, or keep it.
        # Let's average over Batch and Channels to get a single time-series per step representing "Global Activity"
        
        # Mean over Batch(0) and Channels(1) -> Result: [Time_Latent]
        # Oppure possiamo tenere i canali se vuoi vedere se certi canali sono più attivi. 
        # Per semplicità, mediamo sui canali per vedere l'energia spettrale globale.
        
        cfg_latent_profile = G_cfg.detach().mean(dim=(0, 1)).cpu().numpy()  # [1024]
        amg_latent_profile = G_amg.detach().mean(dim=(0, 1)).cpu().numpy()  # [1024]
        
        self.cfg_history.append(cfg_latent_profile)
        self.amg_history.append(amg_latent_profile)
        self.sigma_history.append(sigma)


def compute_and_save_guidance_spectrograms(
    collector: GuidanceSpectralCollector,
    output_dir: str,
    sample_rate: int = 48000, # Used only to calc latent rate
    latent_ratio: int = 2048,
    logger: logging.Logger = None
):
    if not collector.enabled or len(collector.cfg_history) == 0:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Stack: [Num_Steps, Latent_Time]
    cfg_stack = np.stack(collector.cfg_history, axis=0) 
    amg_stack = np.stack(collector.amg_history, axis=0)
    sigmas = np.array(collector.sigma_history)
    
    num_steps, latent_length = cfg_stack.shape
    
    # --- CALCOLO FREQUENZE LATENTI ---
    # SR_latent = SR_audio / Downsampling
    sr_latent = sample_rate / latent_ratio  # Es: 48000 / 2048 = 23.43 Hz
    nyquist_latent = sr_latent / 2          # Es: ~11.7 Hz
    
    if logger:
        logger.info(f"Computing LATENT spectrograms.")
        logger.info(f"  Latent SR: {sr_latent:.2f} Hz | Nyquist: {nyquist_latent:.2f} Hz")
    
    # FFT Latente (Real)
    # axis=1 è il tempo latente
    cfg_fft = scipy_fft.rfft(cfg_stack, axis=1)
    amg_fft = scipy_fft.rfft(amg_stack, axis=1)
    
    # Frequenze per l'asse Y
    freqs = scipy_fft.rfftfreq(latent_length, d=1.0/sr_latent)
    
    # Magnitudo dB
    cfg_db = 20 * np.log10(np.abs(cfg_fft) + 1e-10)
    amg_db = 20 * np.log10(np.abs(amg_fft) + 1e-10)
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # 1. CFG Spectrogram
    im0 = axes[0].imshow(
        cfg_db.T, 
        aspect='auto', 
        origin='lower',
        extent=[0, num_steps, 0, nyquist_latent], # Asse Y da 0 a ~11 Hz
        cmap='inferno'
    )
    axes[0].set_title(f'CFG Latent Spectrum (Normal Guidance)\nNyquist: {nyquist_latent:.2f} Hz')
    axes[0].set_ylabel('Latent Modulation Freq (Hz)')
    axes[0].set_xlabel('Denoising Step')
    plt.colorbar(im0, ax=axes[0], label='dB')

    # 2. AMG Spectrogram
    im1 = axes[1].imshow(
        amg_db.T, 
        aspect='auto', 
        origin='lower',
        extent=[0, num_steps, 0, nyquist_latent],
        cmap='inferno'
    )
    axes[1].set_title(f'AMG Latent Spectrum (Anti-Mem Guidance)\nShould see Cutoff if filtered')
    axes[1].set_ylabel('Latent Modulation Freq (Hz)')
    axes[1].set_xlabel('Denoising Step')
    plt.colorbar(im1, ax=axes[1], label='dB')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'latent_spectrograms.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    
    # Plot Profilo Medio (Per vedere chiaramente il taglio)
    fig_prof, ax_prof = plt.subplots(figsize=(10, 6))
    
    # Media su tutti gli step
    mean_cfg_spec = np.mean(np.abs(cfg_fft), axis=0)
    mean_amg_spec = np.mean(np.abs(amg_fft), axis=0)
    
    ax_prof.plot(freqs, 20*np.log10(mean_cfg_spec + 1e-10), label='CFG (Ref)', color='gray', alpha=0.5)
    ax_prof.plot(freqs, 20*np.log10(mean_amg_spec + 1e-10), label='AMG (Filtered)', color='red', linewidth=2)
    
    ax_prof.set_title('Average Spectral Profile (Latent Space)')
    ax_prof.set_xlabel('Latent Frequency (Hz)')
    ax_prof.set_ylabel('Magnitude (dB)')
    ax_prof.legend()
    ax_prof.grid(True, alpha=0.3)
    
    save_path_prof = os.path.join(output_dir, 'latent_spectrum_profile.png')
    fig_prof.savefig(save_path_prof, dpi=150)
    plt.close(fig_prof)

    return save_path, save_path_prof, None


def generalized_gram_volume(embeddings, return_log=False, logger=None):
    """
    Compute the parallelotope volume spanned by a set of embeddings.
    embeddings: Tensor with shape (K+1, D)
    """
    # 1. Save original dtype
    original_dtype = embeddings.dtype
    
    # 2. Convert everything to FLOAT64 (Double Precision) for calculation
    embeddings_64 = embeddings.to(torch.float64)

    # Compute the Gram matrix in float64
    G = embeddings_64 @ embeddings_64.T  # Shape (K+1, K+1)
    
    # Add jitter in float64
    G = G + torch.eye(G.shape[0], device=G.device, dtype=torch.float64) * 1e-6
    
    sign, log_det = torch.linalg.slogdet(G)
    
    if sign <= 0:
        return None
        
    log_volume = 0.5 * log_det 
    if logger is not None:
        # .item() converte in scalare python, quindi non importa il dtype
        logger.debug(f"[DEBUG] Gram Matrix Sign: {sign}, Log-Determinant: {log_det.item():.4f}")
    
    if return_log:
        # 3. Riconverti il risultato nel dtype originale (float32) per continuare il grafo
        return log_volume.to(original_dtype)
    
    return torch.exp(log_volume).to(original_dtype)


def my_generate_diffusion_cond(
        model,
        clap_model=None,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda:1",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        c1=5.0,
        c2=5.0,
        c3=5.0,
        c_gram =0.0,
        gram_radius=1.0,
        gram_start_step=0,
        gram_use_normalized=True,
        gram_neighborhood_scale=1.0,
        constrain_in_sphere=True,
        lambda_min=0.4,
        lambda_max=0.5,
        logger=None,
        debug_dir=None,
        guidance_rescale=0.0,
        enable_spectral_analysis=False,
        spectral_output_dir=None,
        amg_filter_enabled=False,
        amg_cutoff_ratio=0.25,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        c1, c2, c3: Coefficients for different CLAP guidance components.
        gram_radius: Radius for neighborhood in Gram calculation.
        gram_start_step: Step to start applying Gram-based guidance.
        gram_use_normalized: Whether to use normalized embeddings for Gram calculation.
        gram_neighborhood_scale: Scale for neighborhood in Gram calculation.
        constrain_in_sphere: Whether to constrain embeddings within a sphere.
        lambda_min, lambda_max: Min and max lambda values for Gram guidance.
        logger: Optional logger for debug messages.
        debug_dir: Optional directory to save debug tensors.
        guidance_rescale: Rescaling factor for guidance signals.
        enable_spectral_analysis: Whether to enable spectral analysis of guidance signals.
        spectral_output_dir: Directory to save spectral analysis outputs.
        amg_filter_enabled: Whether to enable AMG filtering on guidance signals.
        amg_cutoff_ratio: Cutoff ratio for AMG filtering (0.0-1.0).
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    if logger is None:
        logger = logging.getLogger()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # Initialize spectral collector
    spectral_collector = GuidanceSpectralCollector(enabled=enable_spectral_analysis)
    if enable_spectral_analysis:
        logger.info("Spectral analysis enabled for guidance signals (audio domain).")
    

    audio_sample_size = sample_size
    effective_audio_length = int(conditioning[0]["seconds_total"] * sample_rate)
    
    # Set pretransform for spectral collector to decode guidance to audio domain
    if model.pretransform is not None:
        if enable_spectral_analysis:
            spectral_collector.set_pretransform(model.pretransform, effective_audio_length)
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
          
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    logger.info(f"Using seed in my_generate_diffusion_cond: {seed}")
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level        

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}
    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective

    if diff_objective == "v":
        if clap_model is not None:
            CLAP = clap_model
        else:
            # Clap init
            CLAP = laion_clap.CLAP_Module(enable_fusion=False, device=device)
            CLAP.load_ckpt()
            CLAP.eval()
            # CLAP tokenizer
        e_prompt = CLAP.get_text_embedding([conditioning[0]["prompt"]])
        e_prompt = e_prompt[0]
        
        ####
        base_denoiser = K.external.VDenoiser(model.model)
        #despec_fn     = make_despec_fn(base_denoiser, e_prompt, s0=cfg_scale, c1=c1, c2=c2, c3=c3, lambda_min=lambda_min, lambda_max=lambda_max, CLAP=CLAP, device=device, length=effective_audio_length, model=model)
        despec_fn = make_despec_fn(
            base_denoiser,
            e_prompt,
            s0=cfg_scale,
            c1=c1,
            c2=c2,
            c3=c3,
            c_gram=c_gram,
            guidance_rescale=guidance_rescale,
            gram_start_step=gram_start_step,
            gram_use_normalized=gram_use_normalized,
            gram_neighborhood_scale=gram_neighborhood_scale,
            constrain_in_sphere=constrain_in_sphere,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            CLAP=CLAP,
            device=device,
            length=effective_audio_length,
            model=model,
            logger=logger,
            spectral_collector=spectral_collector,
            amg_filter_enabled=amg_filter_enabled,
            amg_cutoff_ratio=amg_cutoff_ratio
        )
        guided = make_cond_model_fn(base_denoiser, despec_fn, conditioning_inputs, negative_conditioning_tensors)
        sampler_kwargs['logger'] = logger

        sampled = my_sample_k(
            guided,
            noise,
            init_audio,
            steps,
            **sampler_kwargs,
            **conditioning_inputs,
            **negative_conditioning_tensors,
            cfg_scale=cfg_scale,
            batch_cfg=False,
            device=device,
            noise_seed=seed,
            debug_dir=debug_dir
        )

    elif diff_objective == "rectified_flow":

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, dist_shift=model.dist_shift, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    # if model.pretransform is not None and not return_latents:
    #     #cast sampled latents to pretransform dtype
    #     sampled = sampled.to(next(model.pretransform.parameters()).dtype)
    #     sampled = model.pretransform.decode(sampled)
    if model.pretransform is not None and not return_latents:
        
        # Store original device of the pretransform model
        # pretransform_original_device = next(model.pretransform.parameters()).device
        
        # Move pretransform model to CPU
        # model.pretransform.to('cpu')
        
        # Move sampled tensor to CPU
        sampled_on_cpu = sampled.detach()
        
        # Cast sampled latents to pretransform dtype (this operation will also be on CPU)
        sampled_on_cpu = sampled_on_cpu.to(next(model.pretransform.parameters()).dtype) # dtype is fine, parameters are now on CPU
        
        # Perform decode on CPU
        sampled = model.pretransform.decode(sampled_on_cpu) # Now both model and data are on CPU
        
        # Move pretransform model back to its original device
        # model.pretransform.to(pretransform_original_device)
        
        # 'sampled' is now on CPU. If you need it on the GPU for subsequent steps:
        sampled = sampled.to('cpu') # where 'device' is your target CUDA device
    
    # Compute and save spectral analysis if enabled
    if enable_spectral_analysis and spectral_collector.enabled:
        output_dir = spectral_output_dir if spectral_output_dir else (debug_dir if debug_dir else './spectral_analysis')
        latent_ratio = model.pretransform.downsampling_ratio if model.pretransform is not None else 1
        compute_and_save_guidance_spectrograms(
            collector=spectral_collector,
            output_dir=output_dir,
            sample_rate=model.sample_rate,
            latent_ratio=latent_ratio,
            logger=logger
        )
    
    # Return audio
    return sampled

# 1) Load once (outside your step loop!), convert embeddings to tensors:
with open('embeddings_new.json','r') as f:
    data = json.load(f)

# make a list of IDs and a single tensor of shape (N, D)
# Keep on CPU, will be moved to correct device when needed
ids         = sorted(list(data.keys()))
emb_matrix_cpu  = torch.stack([
    torch.tensor(data[sound_id]['embedding'], 
                dtype=torch.float32)
    for sound_id in ids
], dim=0)  # → (N, D) on CPU

# Cache for device-specific embedding matrices
_emb_matrix_cache = {}

def get_emb_matrix(device: str) -> torch.Tensor:
    """Get embedding matrix on the specified device (cached)."""
    if device not in _emb_matrix_cache:
        _emb_matrix_cache[device] = emb_matrix_cpu.to(device)
    return _emb_matrix_cache[device]

def make_despec_fn(
        base_model_fn,
        e_prompt,
        s0=7.5,
        c1=5.0,
        c2=5.0,
        c3=5.0,
        c_gram=0.0,
        #gram_radius=0.5,
        gram_neighborhood_scale=1.0,
        gram_start_step=0,
        gram_use_normalized=False,
        constrain_in_sphere=True,
        lambda_min=0.4,
        lambda_max=0.5,
        CLAP=None,
        device="cuda:1",
        length=2097152,
        model=None,
        logger=None,
        guidance_rescale=0.0,
        spectral_collector: GuidanceSpectralCollector = None,
        amg_filter_enabled=False,
        amg_cutoff_ratio=0.25

    ):
    """Return a cond_fn that applies AMG‐despecification at each step."""

    if logger is None:
        logger = logging.getLogger()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    neighbor_embeddings_cache = None
    cached_radius = None
    cached_start_et = None
    cached_min_dist = None
    step_counter = 0

    def despec_cond_fn(x, sigma, denoised,
                       conditioning_inputs, negative_conditioning_inputs, **_):
        nonlocal neighbor_embeddings_cache, cached_radius, cached_min_dist, step_counter, cached_start_et
        x.requires_grad_(True)
        # Unconditional and conditional x0 predictions (VDenoiser outputs x0)
        x0_uncond = denoised  # shape [B,C,L]
        x0_cond   = base_model_fn(x, sigma, **conditioning_inputs)

        # Compute alpha_bar (per batch) and derive epsilon estimates from x and x0
        alpha_bar = 1.0 / (1.0 + sigma.pow(2))  # [B]
        sqrt_ab   = alpha_bar.sqrt()
        sqrt_1mab = (1 - alpha_bar).sqrt().clamp_min(1e-8)

        # Broadcasting helper
        expand = lambda t: t.view(-1, *([1] * (x0_uncond.ndim - 1)))

        # Convert model x0 predictions to epsilon predictions: eps = (x_t - sqrt(alpha_bar)*x0)/sqrt(1-alpha_bar)
        eps_uncond = (x - expand(sqrt_ab) * x0_uncond) / expand(sqrt_1mab)
        eps_cond   = (x - expand(sqrt_ab) * x0_cond)   / expand(sqrt_1mab)

        latent_ratio  = model.pretransform.downsampling_ratio
        latent_length = length // latent_ratio
        x0_trim = x0_cond[:, :, :latent_length]
        x0_trim = model.pretransform.decode(x0_trim)

        audio_batch = x0_trim.mean(dim=1)  # mono
        # Normalize to [-1,1] scale for CLAP (avoid division by zero)
        peak = audio_batch.abs().max().clamp_min(1e-6)
        e_t = CLAP.get_audio_embedding_from_data(x=audio_batch/peak, use_tensor=True)
        e_t = e_t[0].to(device)

        G_sim = torch.zeros_like(x0_cond)
        G_spe = torch.zeros_like(x0_cond)
        G_dedup = torch.zeros_like(x0_cond)
        mask = 1.0

        # Nearest neighbour in embedding space
        if c1 > 0 or c2 > 0 or c3 > 0:
            emb_matrix = get_emb_matrix(device)
            with torch.no_grad():  # search doesn't need gradients
                dists = torch.linalg.norm(emb_matrix - e_t.unsqueeze(0), dim=1)
                best_idx = torch.argmin(dists).item()
                best_id  = ids[best_idx]
                best_dist= dists[best_idx].item()
            neighbour_cond = data[best_id]['conditioning']
            audio_embed = torch.tensor(data[best_id]['embedding'], device=device, dtype=e_t.dtype)

            # Conditional prediction for neighbour caption
            conditioning_tensors_N = model.conditioner([neighbour_cond], device)
            conditioning_inputs_N = model.get_conditioning_inputs(conditioning_tensors_N, negative=False)
            x0_cond_N = base_model_fn(x, sigma, **conditioning_inputs_N)
            eps_cond_N = (x - expand(sqrt_ab) * x0_cond_N) / expand(sqrt_1mab)

            # Similarity scalar (dot). (Could normalize embeddings if desired.)
            cos_sim = (e_t * audio_embed).sum(dim=-1)  # scalar
            sim_scalar = cos_sim.sum()

            if c3 > 0:
                # Gradient of similarity w.r.t. x (through x0_uncond -> decode -> embedding model)
                try:
                    grad_sigma = torch.autograd.grad(sim_scalar, x, retain_graph=True, allow_unused=True)[0]
                    if grad_sigma is not None:
                        G_sim = - c3 * torch.sqrt(1 - alpha_bar) * grad_sigma
                except RuntimeError:
                    pass  # Fallback: leave G_sim zeros if path not differentiable

            # Dynamic scales s1, s2 based on similarity
            s1 = (c1 * cos_sim).clamp(0, s0 - 1)
            s2 = (c2 * cos_sim).clamp(0, s0 - s1.item() - 1)

            # Guidance terms computed in epsilon space then mapped back to x0 space.
            delta_eps   = eps_cond   - eps_uncond
            delta_eps_N = eps_cond_N - eps_uncond
            scale_eps2x0 = - (sqrt_1mab / sqrt_ab)
            scale_b = expand(scale_eps2x0)
            delta_x0   = scale_b * delta_eps
            delta_x0_N = scale_b * delta_eps_N
            
            G_spe   = -s1 * delta_x0
            G_dedup = -s2 * delta_x0_N

            lambda_t = lambda_min + (lambda_max - lambda_min) * (alpha_bar ** 2)
            mask = (cos_sim > lambda_t).float().view(-1, *([1] * (G_spe.ndim - 1)))

        else:
            delta_eps   = eps_cond - eps_uncond
            scale_eps2x0 = - (sqrt_1mab / sqrt_ab)
            scale_b = expand(scale_eps2x0)
            delta_x0   = scale_b * delta_eps

        G_cfg = s0 * delta_x0

        #GRAM-AMG
        G_gram = torch.zeros_like(x0_cond) 
        total_gram_loss = 0.0
        
        enable_gram_guidance = c_gram > 0 and step_counter >= gram_start_step
        if enable_gram_guidance:
            logger.debug(f"[DEBUG] Gram guidance enabled at step {step_counter} with c_gram={c_gram}")

            if neighbor_embeddings_cache is None:
                emb_matrix = get_emb_matrix(device)
                with torch.no_grad():
                    e_t_detached = e_t.detach()
                    dists_to_et = torch.linalg.norm(emb_matrix - e_t_detached.unsqueeze(0), dim=1)
                    max_neighbors = min(emb_matrix.shape[0], 511)
                    k_neighbors = int(1 + round(gram_neighborhood_scale * (max_neighbors - 1)))
                    topk_vals, topk_indices = torch.topk(dists_to_et, k=k_neighbors, largest=False)
                    neighbor_embeddings_cache = emb_matrix[topk_indices]
                    cached_radius = topk_vals[-1].item()
                    cached_start_et = e_t_detached.clone()

                    if logger:
                        logger.debug(f"[DEBUG] Gram Init: Scale={gram_neighborhood_scale:.2f} -> Selected {k_neighbors} neighbors. Radius (K-th dist): {cached_radius:.4f}")


            if neighbor_embeddings_cache is not None:
                neighbor_embeddings = neighbor_embeddings_cache
                if gram_use_normalized:
                    e_t_for_volume = F.normalize(e_t, p=2, dim=0)
                    neighbors_for_volume = F.normalize(neighbor_embeddings, p=2, dim=1)
                else:
                    e_t_for_volume = e_t
                    neighbors_for_volume = neighbor_embeddings

                A = torch.cat([e_t_for_volume.unsqueeze(0), neighbors_for_volume], dim=0)
                log_volume = generalized_gram_volume(A, return_log=True) 

                if log_volume is not None:
                    # Maximize log(volume) -> minimize -log(volume)
                    total_gram_loss = -log_volume
                    if constrain_in_sphere:
                        current_dist = torch.linalg.norm(e_t - cached_start_et)
                        if current_dist > cached_radius:
                            diff = current_dist - cached_radius
                            dist_penalty = diff * 0.5
                            total_gram_loss += dist_penalty 
                    logger.debug(f"[DEBUG] Gram log-volume: {log_volume.item():.4f}, Total gram loss: {total_gram_loss.item():.4f}")
                else:
                    logger.debug("[DEBUG] Volume computation failed (None).")
            else:
                logger.debug("[DEBUG] Gram guidance enabled but neighbor cache empty; skip volume.")
             # Questo blocco è sicuro perché total_gram_loss è sempre definito
            if total_gram_loss != 0.0:
                try:
                    grad_gram_raw = torch.autograd.grad(total_gram_loss, x, retain_graph=True, allow_unused=True)[0]
                    
                    if grad_gram_raw is not None:
                        logger.debug(f"[DEBUG] Raw Grad Norm: {grad_gram_raw.norm().item():.4e}")
 
            
                        G_gram = -c_gram * expand(torch.sqrt(1 - alpha_bar)) * grad_gram_raw
                    
                    else:
                        logger.debug("[DEBUG] Grad calc failed (grad_gram_raw is None).")
 
                except RuntimeError as e:
                    logger.debug(f"[DEBUG] Grad calc ERROR: {e}")
                    pass

        # Parabolic gating based on alpha_bar
        # lambda_t = lambda_min + (lambda_max - lambda_min) * (alpha_bar ** 2)
        # mask = (cos_sim > lambda_t).float().view(-1, *([1] * (G_spe.ndim - 1)))

        if amg_filter_enabled and G_gram.abs().sum() > 1e-6:
            logger.debug(f"[FFT FILTER] Applying AMG FFT lowpass filter with cutoff ratio {amg_cutoff_ratio:.2f}, AMG gradient shape before filter: {G_gram.shape}")
            norm_before = G_gram.norm().item()
            G_gram = apply_latent_lowpass_fft(G_gram, cutoff_ratio=amg_cutoff_ratio)
            norm_after = G_gram.norm().item()
            logger.debug(f"[FFT FILTER] Cutoff: {amg_cutoff_ratio:.2f} | Norm: {norm_before:.2f} -> {norm_after:.2f} AMG Gradient shape: {G_gram.shape}")

        additional = G_spe + G_dedup + G_sim + G_gram
        #additional = G_spe + G_dedup + G_sim
        
        step_counter += 1

        # Collect guidance signals for spectral analysis
        if spectral_collector is not None and spectral_collector.enabled:
            sigma_val = sigma[0].item() if sigma.dim() > 0 and sigma.numel() > 1 else sigma.item()
            spectral_collector.collect(G_cfg, mask * additional, sigma_val)

        proposed_x0 = x0_uncond + G_cfg + mask * additional
        if guidance_rescale > 0.0:
            dims = list(range(1, proposed_x0.ndim))
            std_pos = x0_cond.std(dim=dims, keepdim=True)
            std_proposed = proposed_x0.std(dim=dims, keepdim=True)
            
            std_proposed = std_proposed.clamp(min=1e-6)
            factor = std_pos / std_proposed
            
            final_factor = guidance_rescale * factor + (1.0 - guidance_rescale)
            final_x0 = proposed_x0 * final_factor
            
            # Handle both scalar and batched sigma
            sigma_val = sigma[0].item() if sigma.dim() > 0 and sigma.numel() > 1 else sigma.item()
            logger.debug(f"[DEBUG S: {sigma_val:>6.2f}] RESCALE || "
                  f"Std Ref: {std_pos.mean().item():.4f} | "
                  f"Std Prop: {std_proposed.mean().item():.4f} | "
                  f"Factor: {final_factor.mean().item():.4f}")
        else:
            final_x0 = proposed_x0

        # Handle both scalar and batched sigma
        sigma_val = sigma[0].item() if sigma.dim() > 0 and sigma.numel() > 1 else sigma.item()
        logger.debug(f"[DEBUG S: {sigma_val:>6.2f}] NORMS || "
              f"G_cfg: {G_cfg.norm().item():.2f} | "
              f"G_gram: {G_gram.norm().item():.2f} | "
              f"Addit: {additional.norm().item():.2f} | "
              f"Final: {final_x0.norm().item():.2f}")

        return final_x0 - x0_uncond

    return despec_cond_fn
