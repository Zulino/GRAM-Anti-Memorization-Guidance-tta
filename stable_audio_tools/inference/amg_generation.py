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

def apply_latent_highpass_fft(tensor: torch.Tensor, cutoff_ratio: float = 0.25, soft_knee: float = 0.1):
    """
    Applica un High-Pass filter (Passa-Alto) nello spazio latente.
    Mantiene le frequenze SOPRA il cutoff (dettagli rapidi/rumore), rimuove la struttura.
    Utile per ablation studies: "Cosa succede se tengo solo la memorizzazione?"
    """
    # 1. FFT
    fft_g = torch.fft.rfft(tensor, dim=-1, norm='ortho')
    
    # 2. Setup Maschera
    seq_len = fft_g.shape[-1]
    cutoff_idx = int(seq_len * cutoff_ratio)
    fade_len = int(seq_len * soft_knee)
    
    # Inizializza a ZERI (tutto bloccato di default)
    mask = torch.zeros(seq_len, device=tensor.device, dtype=tensor.dtype)
    
    # 3. Logica High-Pass
    # Manteniamo da cutoff_idx fino alla fine
    mask[cutoff_idx:] = 1.0
    
    # Dissolvenza in entrata (Fade-In) per evitare click
    if fade_len > 0 and cutoff_idx > 0:
        start_fade = max(0, cutoff_idx - fade_len)
        # Rampa da 0 a 1 che finisce esattamente al cutoff
        ramp = torch.linspace(0, 1, cutoff_idx - start_fade, device=tensor.device, dtype=tensor.dtype)
        mask[start_fade:cutoff_idx] = ramp

    # 4. Applica e Inverti
    fft_filtered = fft_g * mask
    return torch.fft.irfft(fft_filtered, n=tensor.shape[-1], dim=-1, norm='ortho')

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
        amg_filter_mode='lowpass',
        save_latents=False,    
        latent_filename=None,
        latent_batch_start_idx=1,
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
            amg_cutoff_ratio=amg_cutoff_ratio,
            amg_filter_mode=amg_filter_mode,
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

    if save_latents:
        # Crea cartella 'latents' dentro debug_dir (o nella current dir)
        base_dir = debug_dir if debug_dir else "."
        latent_dir = os.path.join(base_dir, "latents")
        os.makedirs(latent_dir, exist_ok=True)
        
        # Salva ogni elemento del batch separatamente
        batch_size = sampled.shape[0]
        for i in range(batch_size):
            # Determina nome file per questo elemento
            if latent_filename:
                base_name = latent_filename
                # Rimuovi estensioni se presenti
                if base_name.endswith('.wav'): base_name = base_name[:-4]
                if base_name.endswith('.pt'): base_name = base_name[:-3]
                # Usa l'indice globale (batch_start_idx + i)
                fname = f"{base_name}_{latent_batch_start_idx + i}.pt"
            else:
                fname = f"latent_seed{seed}_{latent_batch_start_idx + i}.pt"
                
            save_path = os.path.join(latent_dir, fname)
            
            # Salva singolo tensore (sposta su CPU per risparmiare VRAM/compatibilità)
            torch.save(sampled[i].detach().cpu(), save_path)
            logger.info(f"[IO] Saved latent tensor to: {save_path}")

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
        amg_cutoff_ratio=0.25,
        amg_filter_mode='lowpass'

    ):
    """Return a cond_fn that applies AMG‐despecification at each step."""

    if logger is None:
        logger = logging.getLogger()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    step_counter = 0

    def despec_cond_fn(x, sigma, denoised,
                       conditioning_inputs, negative_conditioning_inputs, **_):
        nonlocal step_counter
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

        # CLAP processing: loop over batch elements to compute embeddings and gradients correctly
        latent_ratio  = model.pretransform.downsampling_ratio
        latent_length = length // latent_ratio
        batch_size = x.shape[0]
        
        # Initialize guidance tensors
        G_sim = torch.zeros_like(x0_cond)
        G_spe = torch.zeros_like(x0_cond)
        G_dedup = torch.zeros_like(x0_cond)
        mask = torch.ones(batch_size, 1, 1, device=device, dtype=x0_cond.dtype)
        
        # For AMG we need to process each batch element separately for correct gradients
        if c1 > 0 or c2 > 0 or c3 > 0:
            emb_matrix = get_emb_matrix(device)
            
            # ============================================================
            # PHASE 1: Compute CLAP embeddings WITHOUT gradients to determine
            # which batch elements exceed the similarity threshold (cheap).
            # ============================================================
            e_t_list_nograd = []
            cos_sim_list = []
            neighbour_cond_list = []
            audio_embed_list = []
            neighbor_embeddings_list = []  # Store K neighbors for each batch element (for GRAM)
            best_ids_list = []
            
            with torch.no_grad():
                for b in range(batch_size):
                    # Decode single element
                    x0_b = x0_cond[b:b+1, :, :latent_length]  # [1, C, latent_length]
                    x0_decoded = model.pretransform.decode(x0_b)  # [1, 2, audio_length]
                    
                    audio_mono = x0_decoded.mean(dim=1)  # [1, audio_length]
                    peak = audio_mono.abs().max().clamp_min(1e-6)
                    audio_normalized = audio_mono / peak  # [1, audio_length]
                    
                    # CLAP expects 1D audio, squeeze to remove batch dim
                    audio_for_clap = audio_normalized.squeeze(0)  # [audio_length]
                    e_t_b = CLAP.get_audio_embedding_from_data(x=[audio_for_clap], use_tensor=True)
                    e_t_b = e_t_b[0].to(device)  # [512]
                    e_t_list_nograd.append(e_t_b)
                    
                    # Find nearest neighbour for AMG (single closest)
                    dists = torch.linalg.norm(emb_matrix - e_t_b.unsqueeze(0), dim=1)
                    best_idx = torch.argmin(dists).item()
                    best_id = ids[best_idx]
                    best_ids_list.append(best_id)
                    
                    # Also find K neighbors for GRAM (if enabled)
                    if c_gram > 0:
                        max_neighbors = min(emb_matrix.shape[0], 511)
                        k_neighbors = int(1 + round(gram_neighborhood_scale * (max_neighbors - 1)))
                        _, topk_indices = torch.topk(dists, k=k_neighbors, largest=False)
                        neighbor_embs = emb_matrix[topk_indices]  # [K, 512]
                        neighbor_embeddings_list.append(neighbor_embs)
                    
                    neighbour_cond = data[best_id]['conditioning']
                    audio_embed = torch.tensor(data[best_id]['embedding'], device=device, dtype=e_t_b.dtype)
                    
                    neighbour_cond_list.append(neighbour_cond)
                    audio_embed_list.append(audio_embed)
                    
                    # Compute similarity for this element
                    cos_sim_b = (e_t_b * audio_embed).sum(dim=-1)
                    cos_sim_list.append(cos_sim_b)
            
            cos_sim = torch.stack(cos_sim_list, dim=0)  # [B]
            
            # Compute mask BEFORE expensive gradient operations
            lambda_t = lambda_min + (lambda_max - lambda_min) * (alpha_bar ** 2)
            mask = (cos_sim > lambda_t).float().view(-1, 1, 1)  # [B, 1, 1]
            
            if logger:
                # Handle both scalar and tensor lambda_t
                if hasattr(lambda_t, 'numel') and lambda_t.numel() > 1:
                    lambda_t_str = f"[{', '.join([f'{v:.3f}' for v in lambda_t.tolist()])}]"
                else:
                    lambda_t_val = lambda_t.item() if hasattr(lambda_t, 'item') else lambda_t
                    lambda_t_str = f"{lambda_t_val:.3f}"
                
                cos_sim_str = f"[{', '.join([f'{v:.2f}' for v in cos_sim.tolist()])}]"
                mask_active = int(mask.sum().item())
                logger.debug(f"[MASK DEBUG S:{step_counter:>3}] lambda_min={lambda_min:.3f} | lambda_max={lambda_max:.3f} | lambda_t={lambda_t_str} | cos_sim={cos_sim_str} | mask_active={mask_active}/{batch_size}")
            
            # Check if ANY element in the batch is active
            any_active = mask.sum().item() > 0
            
            # ============================================================
            # PHASE 2: Only if at least one element exceeds threshold,
            # recompute embeddings WITH gradients for active elements.
            # ============================================================
            if any_active:
                active_indices = [b for b in range(batch_size) if mask[b, 0, 0].item() > 0]
                
                # Recompute CLAP embeddings WITH gradients only for active elements
                e_t_list_grad = [None] * batch_size
                for b in active_indices:
                    x0_b = x0_cond[b:b+1, :, :latent_length]
                    x0_decoded = model.pretransform.decode(x0_b)
                    
                    audio_mono = x0_decoded.mean(dim=1)
                    peak = audio_mono.abs().max().clamp_min(1e-6)
                    audio_normalized = audio_mono / peak
                    
                    audio_for_clap = audio_normalized.squeeze(0)
                    e_t_b = CLAP.get_audio_embedding_from_data(x=[audio_for_clap], use_tensor=True)
                    e_t_b = e_t_b[0].to(device)
                    e_t_list_grad[b] = e_t_b
                
                # G_sim: gradient of cosine similarity for active elements only
                if c3 > 0:
                    try:
                        total_sim = torch.tensor(0.0, device=device)
                        for b in active_indices:
                            e_t_b = e_t_list_grad[b]
                            audio_embed_b = audio_embed_list[b]
                            cos_sim_b = (e_t_b * audio_embed_b).sum(dim=-1)
                            total_sim = total_sim + cos_sim_b
                        
                        grad_sigma = torch.autograd.grad(total_sim, x, retain_graph=True, allow_unused=True)[0]
                        if grad_sigma is None:
                            grad_sigma = torch.zeros_like(x)
                        G_sim = - c3 * expand(torch.sqrt(1 - alpha_bar)) * grad_sigma
                        
                        if step_counter % 10 == 0 and logger:
                            logger.debug(f"STEP {step_counter}: G_sim calculated for {len(active_indices)} active elements. Norm: {G_sim.norm().item()}")
                    except RuntimeError as e:
                        if logger: logger.error(f"STEP {step_counter}: Gradient Error: {e}")
                        G_sim = torch.zeros_like(x)
                
                # Build e_t_batch for GRAM: use no-grad embeddings for inactive, grad for active
                e_t_batch_list = []
                for b in range(batch_size):
                    if e_t_list_grad[b] is not None:
                        e_t_batch_list.append(e_t_list_grad[b])
                    else:
                        e_t_batch_list.append(e_t_list_nograd[b])
                e_t_batch = torch.stack(e_t_batch_list, dim=0)  # [B, 512]
                
                # Conditional prediction for neighbour captions
                conditioning_tensors_N = model.conditioner(neighbour_cond_list, device)
                conditioning_inputs_N = model.get_conditioning_inputs(conditioning_tensors_N, negative=False)
                x0_cond_N = base_model_fn(x, sigma, **conditioning_inputs_N)
                eps_cond_N = (x - expand(sqrt_ab) * x0_cond_N) / expand(sqrt_1mab)

                # Dynamic scales s1, s2 - per-element vectors
                cos_sim_expanded = cos_sim.view(-1, 1, 1)  # [B, 1, 1]
                s1 = torch.clamp(c1 * cos_sim_expanded, min=0, max=s0 - 1)  # [B, 1, 1]
                s2 = torch.clamp(c2 * cos_sim_expanded, min=0, max=s0 - 1)  # [B, 1, 1]
                s2 = torch.min(s2, s0 - s1 - 1)

                # Guidance terms computed in epsilon space then mapped back to x0 space
                delta_eps   = eps_cond   - eps_uncond
                delta_eps_N = eps_cond_N - eps_uncond
                scale_eps2x0 = - (sqrt_1mab / sqrt_ab)
                scale_b = expand(scale_eps2x0)
                delta_x0   = scale_b * delta_eps
                delta_x0_N = scale_b * delta_eps_N
                
                G_spe   = -s1 * delta_x0  # Per-element scaling [B, C, L]
                G_dedup = -s2 * delta_x0_N  # Per-element scaling [B, C, L]
            
            else:
                # No active elements - skip all expensive gradient computations
                if logger:
                    logger.debug(f"STEP {step_counter}: No active elements (all below threshold). Skipping gradient computations.")
                e_t_batch = torch.stack(e_t_list_nograd, dim=0)  # [B, 512]
                
                # Still need delta_x0 for G_cfg below
                delta_eps   = eps_cond   - eps_uncond
                scale_eps2x0 = - (sqrt_1mab / sqrt_ab)
                scale_b = expand(scale_eps2x0)
                delta_x0   = scale_b * delta_eps

        else:
            # No AMG - but still need e_t_batch and neighbors if GRAM is enabled
            e_t_batch = None
            neighbor_embeddings_list = []
            cos_sim = None
            if c_gram > 0:
                emb_matrix = get_emb_matrix(device)
                # Compute CLAP embedding for ALL batch elements for GRAM guidance
                e_t_list = []
                cos_sim_list = []
                for b in range(batch_size):
                    x0_b = x0_cond[b:b+1, :, :latent_length]
                    x0_decoded = model.pretransform.decode(x0_b)
                    audio_mono = x0_decoded.mean(dim=1)
                    peak = audio_mono.abs().max().clamp_min(1e-6)
                    audio_normalized = audio_mono / peak
                    # CLAP expects 1D audio, squeeze to remove batch dim
                    audio_for_clap = audio_normalized.squeeze(0)  # [audio_length]
                    e_t_b = CLAP.get_audio_embedding_from_data(x=[audio_for_clap], use_tensor=True)
                    e_t_b = e_t_b[0].to(device)  # [512]
                    e_t_list.append(e_t_b)
                    
                    # Find K neighbors for this batch element
                    with torch.no_grad():
                        dists = torch.linalg.norm(emb_matrix - e_t_b.unsqueeze(0), dim=1)
                        best_idx = torch.argmin(dists).item()
                        best_id = ids[best_idx]

                        audio_embed = torch.tensor(data[best_id]['embedding'], device=device, dtype=e_t_b.dtype)
                        cos_sim_b = (e_t_b * audio_embed).sum(dim=-1)
                        cos_sim_list.append(cos_sim_b)


                        max_neighbors = min(emb_matrix.shape[0], 511)
                        k_neighbors = int(1 + round(gram_neighborhood_scale * (max_neighbors - 1)))
                        _, topk_indices = torch.topk(dists, k=k_neighbors, largest=False)
                        neighbor_embs = emb_matrix[topk_indices]  # [K, 512]
                        neighbor_embeddings_list.append(neighbor_embs)
                        
                e_t_batch = torch.stack(e_t_list, dim=0)  # [B, 512]
                cos_sim = torch.stack(cos_sim_list, dim=0)  # [B]

                lambda_t = lambda_min + (lambda_max - lambda_min) * (alpha_bar ** 2)
                mask = (cos_sim > lambda_t).float().view(-1, 1, 1)  # [B, 1, 1]
                
                if logger:
                    # Handle both scalar and tensor lambda_t
                    if hasattr(lambda_t, 'numel') and lambda_t.numel() > 1:
                        lambda_t_str = f"[{', '.join([f'{v:.3f}' for v in lambda_t.tolist()])}]"
                    else:
                        lambda_t_val = lambda_t.item() if hasattr(lambda_t, 'item') else lambda_t
                        lambda_t_str = f"{lambda_t_val:.3f}"
                    
                    cos_sim_str = f"[{', '.join([f'{v:.2f}' for v in cos_sim.tolist()])}]"
                    mask_active = int(mask.sum().item())
                    logger.debug(f"[MASK DEBUG S:{step_counter:>3}] lambda_min={lambda_min:.3f} | lambda_max={lambda_max:.3f} | lambda_t={lambda_t_str} | cos_sim={cos_sim_str} | mask_active={mask_active}/{batch_size}")

            
            delta_eps   = eps_cond - eps_uncond
            scale_eps2x0 = - (sqrt_1mab / sqrt_ab)
            scale_b = expand(scale_eps2x0)
            delta_x0   = scale_b * delta_eps

        G_cfg = s0 * delta_x0

        #GRAM-AMG - iterate over all batch elements, each with its own neighbors
        G_gram = torch.zeros_like(x0_cond) 
        total_gram_loss = torch.tensor(0.0, device=device, dtype=x0_cond.dtype)
        
        enable_gram_guidance = c_gram > 0 and step_counter >= gram_start_step
        # Only compute GRAM gradients if guidance is enabled AND mask is active for at least one element
        any_mask_active = mask.sum().item() > 0 if enable_gram_guidance else False
        if enable_gram_guidance and any_mask_active:
            logger.debug(f"[DEBUG] Gram guidance enabled at step {step_counter} with c_gram={c_gram}")

            if e_t_batch is None or len(neighbor_embeddings_list) == 0:
                logger.warning("[DEBUG] GRAM guidance enabled but e_t_batch or neighbors missing - skipping")
            else:
                # Compute GRAM loss only for ACTIVE batch elements (those above threshold)
                active_gram_indices = [b for b in range(batch_size) if mask[b, 0, 0].item() > 0]
                
                for b in active_gram_indices:
                    e_t_b = e_t_batch[b]  # [512] - current generated embedding
                    neighbor_embeddings_b = neighbor_embeddings_list[b]  # [K, 512] - neighbors specific to this element
                    
                    if gram_use_normalized:
                        e_t_for_volume = F.normalize(e_t_b, p=2, dim=0)
                        neighbors_for_volume = F.normalize(neighbor_embeddings_b, p=2, dim=1)
                    else:
                        e_t_for_volume = e_t_b
                        neighbors_for_volume = neighbor_embeddings_b

                    # Build parallelotope matrix: [Generated, Neighbor1, Neighbor2, ...]
                    # Shape: (1+K, 512)
                    A = torch.cat([e_t_for_volume.unsqueeze(0), neighbors_for_volume], dim=0)
                    log_volume = generalized_gram_volume(A, return_log=True) 

                    if log_volume is not None:
                        # Maximize log(volume) -> minimize -log(volume)
                        # High volume = generated embedding is far from neighbors = good
                        gram_loss_b = -log_volume
                        total_gram_loss = total_gram_loss + gram_loss_b
                        
                if step_counter % 10 == 0 and logger:
                    logger.debug(f"[DEBUG] Total Gram loss for {len(active_gram_indices)} active elements: {total_gram_loss.item():.4f}")
        elif enable_gram_guidance and not any_mask_active:
            if logger:
                logger.debug(f"STEP {step_counter}: GRAM guidance skipped - no active elements (all below threshold).")

        # Compute gradient from summed GRAM loss (outside the if/elif so it runs when mask IS active)
        if enable_gram_guidance and total_gram_loss != 0.0 and not isinstance(total_gram_loss, float):
            try:
                grad_gram_raw = torch.autograd.grad(total_gram_loss, x, retain_graph=True, allow_unused=True)[0]
                
                if grad_gram_raw is not None:
                    if logger:
                        logger.debug(f"[DEBUG] Raw Grad Norm: {grad_gram_raw.norm().item():.4e}")
                    G_gram = -c_gram * expand(torch.sqrt(1 - alpha_bar)) * grad_gram_raw
                else:
                    if logger:
                        logger.debug("[DEBUG] Grad calc failed (grad_gram_raw is None).")

            except RuntimeError as e:
                if logger:
                    logger.debug(f"[DEBUG] Grad calc ERROR: {e}")
                pass

        if amg_filter_enabled and G_gram.abs().sum() > 1e-6:
            logger.debug(f"[FFT FILTER] Applying AMG FFT lowpass filter with cutoff ratio {amg_cutoff_ratio:.2f}, AMG gradient shape before filter: {G_gram.shape}")
            norm_before = G_gram.norm().item()
            if amg_filter_mode == 'highpass':
                G_gram = apply_latent_highpass_fft(G_gram, cutoff_ratio=amg_cutoff_ratio)
                filter_name = "HighPass"
            else:
                # Default a LowPass (quello che usavi prima)
                G_gram = apply_latent_lowpass_fft(G_gram, cutoff_ratio=amg_cutoff_ratio)
                filter_name = "LowPass"
            
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
