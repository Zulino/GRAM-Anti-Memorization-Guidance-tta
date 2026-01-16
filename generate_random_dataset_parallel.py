"""
Parallel Random Dataset Generator with Multiple Configurations

This script generates audio samples with different configurations in parallel,
distributing work across multiple GPUs.

Each configuration gets its own output directory under BASE_OUTPUT_DIR.

SELECTION MODES:
----------------
1. CLUSTER MODE: Set GENERATION_SOURCE to a cluster specification:
   - List of cluster IDs: [1, 5, 10, 20]
   - Range string: "3-20" (clusters 3 to 20 inclusive)
   - Tuple range: (3, 20) (clusters 3 to 20 inclusive)
   
   In cluster mode, the representative_id from cluster_representatives.csv
   is used for each cluster, and BATCH_SIZE generations are made per cluster.

2. RANDOM MODE: Set GENERATION_SOURCE to an integer N:
   - N random prompts are selected from embeddings_new.json
   - BATCH_SIZE generations are made per prompt.

Examples:
    GENERATION_SOURCE = 100          # 100 random prompts
    GENERATION_SOURCE = [1, 5, 10]   # Clusters 1, 5, and 10
    GENERATION_SOURCE = "0-30"       # Clusters 0 through 30
    GENERATION_SOURCE = (0, 60)      # Clusters 0 through 60
"""

import torch
import torch.multiprocessing as mp
import os
import sys
import io
import json
import csv
import random
import numpy as np
import torchaudio
from einops import rearrange
from tqdm import tqdm
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import queue
import time
import logging
from datetime import datetime

# Global log file path (set in main, used by workers)
GLOBAL_LOG_FILE = None

# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for a single generation experiment."""
    name: str                          # Directory name (e.g., "gram_rescale", "baseline_no_rescale")
    
    # Guidance parameters
    cfg_scale: float = 7.0             # Classifier-free guidance scale
    guidance_rescale: float = 0.0      # Rescale factor (0.0 = no rescale)
    
    # GRAM/AMG parameters
    c_gram: float = 0.0                # GRAM coefficient (0.0 = disabled)
    gram_neighborhood_scale: float = 0.0
    constrain_in_sphere: bool = False
    gram_use_normalized: bool = False
    gram_start_step: int = 0
    lambda_min: float = 0.7
    lambda_max: float = 0.8
    
    # Other AMG parameters
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    
    # AMG filter parameters
    amg_filter_enabled: bool = False
    amg_cutoff_ratio: float = 0.25
    amg_filter_mode: str = 'lowpass'
    
    # Sampling parameters
    steps: int = 100
    sigma_min: float = 0.3
    sigma_max: float = 500.0
    sampler_type: str = "my-dpmpp-3m-sde"
    
    # Spectral analysis parameters
    enable_spectral_analysis: bool = False
    spectral_output_dir: Optional[str] = None
    
    # Latent saving parameters
    save_latents: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict for logging."""
        return {
            'name': self.name,
            'cfg_scale': self.cfg_scale,
            'guidance_rescale': self.guidance_rescale,
            'c_gram': self.c_gram,
            'gram_neighborhood_scale': self.gram_neighborhood_scale,
            'constrain_in_sphere': self.constrain_in_sphere,
            'gram_use_normalized': self.gram_use_normalized,
            'gram_start_step': self.gram_start_step,
            'lambda_min': self.lambda_min,
            'lambda_max': self.lambda_max,
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3,
            'amg_filter_enabled': self.amg_filter_enabled,
            'amg_cutoff_ratio': self.amg_cutoff_ratio,
            'amg_filter_mode': self.amg_filter_mode,
            'steps': self.steps,
            'sampler_type': self.sampler_type,
            'enable_spectral_analysis': self.enable_spectral_analysis,
            'spectral_output_dir': self.spectral_output_dir,
            'save_latents': self.save_latents,
        }


# =============================================================================
# DEFINE YOUR CONFIGURATIONS HERE
# =============================================================================

CONFIGURATIONS: List[GenerationConfig] = [
    # 1. GRAM non filtrato (come in amg_infer.py ma senza filtro)
    # GenerationConfig(
    #     name="baseline",
    #     cfg_scale=7.0,
    #     guidance_rescale=0.0,
    #     c_gram=0.0,
    #     c1=6.0,
    #     c2=6.0,
    #     c3=1000.0,
    #     gram_neighborhood_scale=0.6,
    #     gram_use_normalized=False,
    #     gram_start_step=0,
    #     lambda_min=0.7,
    #     lambda_max=0.8,
    #     constrain_in_sphere=False,
    #     amg_filter_enabled=False,
    #     enable_spectral_analysis=False,
    #     save_latents=True,
    # ),
    
    # #2. GRAM filtrato lowpass (come in amg_infer.py)
    GenerationConfig(
        name="gram_no_filt_250_0.6",
        cfg_scale=7.0,
        guidance_rescale=0.0,
        c_gram=250.0,
        c1=0.0,
        c2=0.0,
        c3=0.0,
        gram_neighborhood_scale=0.6,
        gram_use_normalized=False,
        gram_start_step=0,
        lambda_min=0.7,
        lambda_max=0.8,
        constrain_in_sphere=False,
        amg_filter_enabled=False,
        amg_cutoff_ratio=0.25,
        amg_filter_mode='lowpass',
        enable_spectral_analysis=False,
        save_latents=True,
    ),

    GenerationConfig(
        name="gram_lp_250_0.3",
        cfg_scale=7.0,
        guidance_rescale=0.0,
        c_gram=250.0,
        c1=0.0,
        c2=0.0,
        c3=0.0,
        gram_neighborhood_scale=0.3,
        gram_use_normalized=False,
        gram_start_step=0,
        lambda_min=0.7,
        lambda_max=0.8,
        constrain_in_sphere=False,
        amg_filter_enabled=True,
        amg_cutoff_ratio=0.25,
        amg_filter_mode='lowpass',
        enable_spectral_analysis=False,
        save_latents=True,
    ),

    # GenerationConfig(
    #     name="gram_lp_1000_0",
    #     cfg_scale=7.0,
    #     guidance_rescale=0.0,
    #     c_gram=1000.0,
    #     c1=0.0,
    #     c2=0.0,
    #     c3=0.0,
    #     gram_neighborhood_scale=0.0,
    #     gram_use_normalized=False,
    #     gram_start_step=0,
    #     lambda_min=0.7,
    #     lambda_max=0.8,
    #     constrain_in_sphere=False,
    #     amg_filter_enabled=True,
    #     amg_cutoff_ratio=0.25,
    #     amg_filter_mode='lowpass',
    #     enable_spectral_analysis=False,
    #     save_latents=True,
    # ),

    GenerationConfig(
        name="gram_lp_1000_1",
        cfg_scale=7.0,
        guidance_rescale=0.0,
        c_gram=1000.0,
        c1=0.0,
        c2=0.0,
        c3=0.0,
        gram_neighborhood_scale=1.0,
        gram_use_normalized=False,
        gram_start_step=0,
        lambda_min=0.7,
        lambda_max=0.8,
        constrain_in_sphere=False,
        amg_filter_enabled=True,
        amg_cutoff_ratio=0.25,
        amg_filter_mode='lowpass',
        enable_spectral_analysis=False,
        save_latents=True,
    ),

    GenerationConfig(
        name="gram_hp_250_0.6",
        cfg_scale=7.0,
        guidance_rescale=0.0,
        c_gram=250.0,
        c1=0.0,
        c2=0.0,
        c3=0.0,
        gram_neighborhood_scale=0.6,
        gram_use_normalized=False,
        gram_start_step=0,
        lambda_min=0.7,
        lambda_max=0.8,
        constrain_in_sphere=False,
        amg_filter_enabled=True,
        amg_cutoff_ratio=0.25,
        amg_filter_mode='highpass',
        enable_spectral_analysis=False,
        save_latents=True,
    ),

    # GenerationConfig(
    #     name="gram_lp_500",
    #     cfg_scale=7.0,
    #     guidance_rescale=0.0,
    #     c_gram=500.0,
    #     c1=0.0,
    #     c2=0.0,
    #     c3=0.0,
    #     gram_neighborhood_scale=0.6,
    #     gram_use_normalized=False,
    #     gram_start_step=0,
    #     lambda_min=0.7,
    #     lambda_max=0.8,
    #     constrain_in_sphere=False,
    #     amg_filter_enabled=True,
    #     amg_cutoff_ratio=0.25,
    #     amg_filter_mode='lowpass',
    #     enable_spectral_analysis=False,
    #     save_latents=True,
    # ),

    # GenerationConfig(
    #     name="gram_lp_500_norm",
    #     cfg_scale=7.0,
    #     guidance_rescale=0.0,
    #     c_gram=500.0,
    #     c1=0.0,
    #     c2=0.0,
    #     c3=0.0,
    #     gram_neighborhood_scale=0.6,
    #     gram_use_normalized=True,
    #     gram_start_step=0,
    #     lambda_min=0.7,
    #     lambda_max=0.8,
    #     constrain_in_sphere=False,
    #     amg_filter_enabled=True,
    #     amg_cutoff_ratio=0.25,
    #     amg_filter_mode='lowpass',
    #     enable_spectral_analysis=False,
    #     save_latents=True,
    # ),

    # GenerationConfig(
    #     name="gram_lp_250_norm",
    #     cfg_scale=7.0,
    #     guidance_rescale=0.0,
    #     c_gram=250.0,
    #     c1=0.0,
    #     c2=0.0,
    #     c3=0.0,
    #     gram_neighborhood_scale=0.6,
    #     gram_use_normalized=True,
    #     gram_start_step=0,
    #     lambda_min=0.7,
    #     lambda_max=0.8,
    #     constrain_in_sphere=False,
    #     amg_filter_enabled=True,
    #     amg_cutoff_ratio=0.25,
    #     amg_filter_mode='lowpass',
    #     enable_spectral_analysis=False,
    #     save_latents=True,
    # ),
    
    # 3. No AMG (baseline solo CFG)
    # GenerationConfig(
    #     name="no_amg",
    #     cfg_scale=7.0,
    #     guidance_rescale=0.0,
    #     c_gram=0.0,
    #     c1=0.0,
    #     c2=0.0,
    #     c3=0.0,
    #     gram_neighborhood_scale=0.6,
    #     gram_use_normalized=False,
    #     gram_start_step=0,
    #     lambda_min=0.7,
    #     lambda_max=0.8,
    #     constrain_in_sphere=False,
    #     amg_filter_enabled=False,
    #     enable_spectral_analysis=False,
    #     save_latents=True,
    # ),

    # 3. No AMG (baseline solo CFG)
    # GenerationConfig(
    #     name="baseline",
    #     cfg_scale=7.0,
    #     guidance_rescale=0.0,
    #     c_gram=0.0,
    #     c1=6.0,
    #     c2=6.0,
    #     c3=1000.0,
    #     gram_neighborhood_scale=0.6,
    #     gram_use_normalized=False,
    #     gram_start_step=0,
    #     lambda_min=0.7,
    #     lambda_max=0.8,
    #     constrain_in_sphere=False,
    #     amg_filter_enabled=False,
    #     enable_spectral_analysis=False,
    #     save_latents=True,
    # ),
]

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

INPUT_JSON = 'embeddings_new.json'
CLUSTER_REPRESENTATIVES_CSV = 'cluster_representatives.csv'
BASE_OUTPUT_DIR = "./gram_lp_sphere_studies_full_clusters_6k"
LOG_DIR = "./logs"

# GENERATION SOURCE:
# - Integer N: Select N random prompts from embeddings_new.json
# - List [1, 5, 10]: Use representatives from clusters 1, 5, 10
# - String "3-20": Use representatives from clusters 3 to 20 (inclusive)
# - Tuple (3, 20): Use representatives from clusters 3 to 20 (inclusive)
GENERATION_SOURCE: Union[int, List[int], str, tuple] = (0, 19)  # Single audio for spectral analysis

BATCH_SIZE = 100                       # Number of audio to generate per prompt/cluster
MINI_BATCH_SIZE = 4                  # Number of audio to generate at once (reduces VRAM usage)
SEED = -1                          # -1 for random (consistent across configs)
SELECTION_SEED = -1                # Seed for ID selection (-1 = random, use fixed value to reproduce same selection)
GPUS = [0, 1]                      # List of GPU indices to use

def parse_generation_source(source: Union[int, List[int], str, tuple]) -> tuple:
    """
    Parse GENERATION_SOURCE and return (mode, ids_or_count).
    
    Returns:
        ('random', count) for random mode
        ('cluster', list_of_cluster_ids) for cluster mode
    """
    if isinstance(source, int):
        return ('random', source)
    
    elif isinstance(source, list):
        # List of cluster IDs
        return ('cluster', source)
    
    elif isinstance(source, str):
        # Parse range string like "3-20"
        if '-' in source:
            parts = source.split('-')
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            return ('cluster', list(range(start, end + 1)))
        else:
            # Single cluster ID as string
            return ('cluster', [int(source.strip())])
    
    elif isinstance(source, tuple) and len(source) == 2:
        # Tuple range like (3, 20)
        start, end = source
        return ('cluster', list(range(start, end + 1)))
    
    else:
        raise ValueError(f"Invalid GENERATION_SOURCE format: {source}")


def load_cluster_representatives(csv_path: str) -> Dict[int, str]:
    """
    Load cluster representatives from CSV file.
    
    Returns:
        Dict mapping cluster_id -> representative_id (as string)
    """
    representatives = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster_id = int(row['cluster_id'])
            representative_id = row['representative_id']
            representatives[cluster_id] = representative_id
    return representatives


# =============================================================================
# WORKER FUNCTION
# =============================================================================

def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, 
                   data: Dict, sample_seeds: Dict[str, int]):
    """
    Worker process that runs on a single GPU and processes tasks from the queue.
    
    Args:
        gpu_id: GPU index to use
        task_queue: Queue of (config, sound_id) tuples to process
        result_queue: Queue to put results (success/failure)
        data: The full dataset dictionary
        sample_seeds: Dict mapping sound_id to seed for reproducibility
    """
    device = f"cuda:{gpu_id}"
    
    try:
        # Import inside worker to avoid CUDA initialization in main process
        from stable_audio_tools import get_pretrained_model
        from stable_audio_tools.inference import amg_generation
        from stable_audio_tools.inference.amg_generation import my_generate_diffusion_cond
        
        # Load models on this GPU
        print(f"[GPU {gpu_id}] Loading Stable Audio model...")
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        sample_rate = model_config["sample_rate"]
        downsampling_ratio = model.pretransform.downsampling_ratio if hasattr(model, 'pretransform') else 1
        model = model.to(device)
        
        print(f"[GPU {gpu_id}] Loading CLAP model...")
        CLAP = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, device=device)
        CLAP.load_ckpt()
        CLAP.eval()
        
        print(f"[GPU {gpu_id}] Ready, waiting for tasks...")
        
        while True:
            try:
                task = task_queue.get(timeout=5)
            except queue.Empty:
                # Check if we should exit
                if task_queue.empty():
                    break
                continue
            
            if task is None:  # Poison pill
                break
            
            config, sound_id = task
            
            try:
                # Get info for this sample
                info = data[sound_id]['conditioning']
                prompt = info['prompt']
                duration = info['seconds_total']
                
                # Calculate sample size
                requested_samples = int(duration * sample_rate)
                generation_sample_size = math.ceil(requested_samples / downsampling_ratio) * downsampling_ratio
                
                # Conditioning
                conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": duration}]
                negative_conditioning = [{"prompt": "", "seconds_start": 0, "seconds_total": duration}]
                
                # Use consistent seed across configurations
                file_seed = sample_seeds[sound_id]
                
                # Generate
                output = my_generate_diffusion_cond(
                    model,
                    steps=config.steps,
                    cfg_scale=config.cfg_scale,
                    conditioning=conditioning,
                    negative_conditioning=negative_conditioning,
                    sample_size=generation_sample_size,
                    sample_rate=sample_rate,
                    sigma_min=config.sigma_min,
                    sigma_max=config.sigma_max,
                    sampler_type=config.sampler_type,
                    device=device,
                    seed=file_seed,
                    guidance_rescale=config.guidance_rescale,
                    clap_model=CLAP,
                    c_gram=config.c_gram,
                    gram_neighborhood_scale=config.gram_neighborhood_scale,
                    constrain_in_sphere=config.constrain_in_sphere,
                    gram_use_normalized=config.gram_use_normalized,
                    c1=config.c1,
                    c2=config.c2,
                    c3=config.c3,
                    enable_spectral_analysis=config.enable_spectral_analysis,
                    spectral_output_dir=config.spectral_output_dir,
                )
                
                # Post-processing
                output = rearrange(output, "b d n -> d (b n)")
                output = output[:, :requested_samples]
                
                output_float = output.to(torch.float32)
                peak = output_float.abs().max().clamp_min(1e-6)
                output_normalized = (output_float / peak).clamp(-1, 1)
                
                # Save
                output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
                filename = os.path.join(output_dir, f"sound_{sound_id}.wav")
                output_int16 = output_normalized.mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, output_int16, sample_rate)
                
                result_queue.put(('success', config.name, sound_id))
                
            except Exception as e:
                result_queue.put(('error', config.name, sound_id, str(e)))
                
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error: {e}")
        result_queue.put(('fatal', gpu_id, str(e)))


class TeeWriter:
    """Write to both terminal and log file."""
    def __init__(self, log_file_path, original_stream):
        self.log_file_path = log_file_path
        self.original_stream = original_stream
        self.buffer = ""
    
    def write(self, text):
        if text:
            self.original_stream.write(text)
            self.original_stream.flush()
            # Append to log file
            try:
                with open(self.log_file_path, 'a') as f:
                    f.write(text)
            except:
                pass
    
    def flush(self):
        self.original_stream.flush()


def run_sequential_on_gpu(gpu_id: int, configs: List[GenerationConfig], 
                          selected_ids: List[str], data: Dict, 
                          sample_seeds: Dict[str, int],
                          log_file_path: str,
                          mode: str = 'random',
                          cluster_info: Dict[str, int] = None):
    """
    Run generation sequentially on a single GPU for assigned configurations.
    This is a simpler alternative to the queue-based approach.
    
    Args:
        mode: 'random' or 'cluster'
        cluster_info: Dict mapping sound_id -> cluster_id (only for cluster mode)
    """
    import traceback
    
    device = f"cuda:{gpu_id}"
    
    # Redirect stdout and stderr to also write to log file
    sys.stdout = TeeWriter(log_file_path, sys.__stdout__)
    sys.stderr = TeeWriter(log_file_path, sys.__stderr__)
    
    def log_msg(msg, level="INFO"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"{timestamp} | {level:8s} | [GPU {gpu_id}] {msg}"
        print(line)
    
    try:
        # Import inside function to control CUDA initialization
        from stable_audio_tools import get_pretrained_model
        from stable_audio_tools.inference import amg_generation
        from stable_audio_tools.inference.amg_generation import my_generate_diffusion_cond
        
        # Load models
        log_msg("Loading Stable Audio model...")
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        sample_rate = model_config["sample_rate"]
        downsampling_ratio = model.pretransform.downsampling_ratio if hasattr(model, 'pretransform') else 1
        model = model.to(device)
        
        log_msg("Loading CLAP model...")
        CLAP = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, device=device)
        CLAP.load_ckpt()
        CLAP.eval()
        
        log_msg("Models loaded successfully!")
        
    except Exception as e:
        log_msg(f"FATAL: Failed to load models: {e}", "ERROR")
        log_msg(traceback.format_exc(), "ERROR")
        return {'success': 0, 'error': 0, 'errors': [{'fatal': str(e)}]}
    
    results = {'success': 0, 'error': 0, 'errors': []}
    
    for config in configs:
        log_msg(f"Processing config: {config.name}")
        
        for sound_id in tqdm(selected_ids, desc=f"[GPU {gpu_id}] {config.name}"):
            try:
                # Determine output directory based on mode
                if mode == 'cluster' and cluster_info and sound_id in cluster_info:
                    cluster_id = cluster_info[sound_id]
                    output_dir = os.path.join(BASE_OUTPUT_DIR, config.name, str(cluster_id))
                else:
                    output_dir = os.path.join(BASE_OUTPUT_DIR, config.name, "random_prompts")
                
                # Create directory if needed
                os.makedirs(output_dir, exist_ok=True)
                
                # Check if all batch files already exist
                all_exist = True
                for batch_idx in range(BATCH_SIZE):
                    if BATCH_SIZE == 1:
                        filename = os.path.join(output_dir, f"gen_{batch_idx + 1}.wav")
                    else:
                        filename = os.path.join(output_dir, f"gen_{batch_idx + 1}.wav")
                    if not os.path.exists(filename):
                        all_exist = False
                        break
                
                if all_exist:
                    log_msg(f"Skipping {config.name}/{sound_id} (all {BATCH_SIZE} files exist)", "DEBUG")
                    results['success'] += BATCH_SIZE
                    continue
                
                # Get info
                info = data[sound_id]['conditioning']
                prompt = info['prompt']
                duration = info['seconds_total']
                
                log_msg(f"Generating {config.name}/{sound_id} x{BATCH_SIZE}: '{prompt[:50]}...' ({duration}s)", "DEBUG")
                
                # Calculate sample size
                requested_samples = int(duration * sample_rate)
                generation_sample_size = math.ceil(requested_samples / downsampling_ratio) * downsampling_ratio
                
                # Conditioning
                conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": duration}]
                negative_conditioning = [{"prompt": "", "seconds_start": 0, "seconds_total": duration}]
                
                file_seed = sample_seeds[sound_id]
                
                # Generate in mini-batches to avoid OOM
                global_batch_idx = 0
                num_mini_batches = math.ceil(BATCH_SIZE / MINI_BATCH_SIZE)
                
                for mini_batch_num in range(num_mini_batches):
                    # Calculate how many to generate in this mini-batch
                    remaining = BATCH_SIZE - global_batch_idx
                    current_mini_batch_size = min(MINI_BATCH_SIZE, remaining)
                    
                    if current_mini_batch_size <= 0:
                        break
                    
                    # Use different seed offset for each mini-batch for variety
                    mini_batch_seed = file_seed + mini_batch_num * 1000
                    
                    # Generate with mini_batch_size
                    # Determine spectral output dir for this specific generation
                    if config.enable_spectral_analysis:
                        if config.spectral_output_dir:
                            spectral_dir = config.spectral_output_dir
                        else:
                            spectral_dir = os.path.join(output_dir, "plots")
                    else:
                        spectral_dir = None
                    
                    # Determina il nome del file latent in base alla modalità
                    if mode == 'cluster' and cluster_info and sound_id in cluster_info:
                        latent_fname = "gen"
                        latent_start_idx = global_batch_idx + 1
                    else:
                        # Random mode: usa sound_id nel nome per evitare sovrascritture
                        if BATCH_SIZE == 1:
                            latent_fname = f"sound_{sound_id}"
                            latent_start_idx = 1  # Non serve indice se BATCH_SIZE=1
                        else:
                            latent_fname = f"sound_{sound_id}"
                            latent_start_idx = global_batch_idx + 1
                    
                    output = my_generate_diffusion_cond(
                        model,
                        steps=config.steps,
                        cfg_scale=config.cfg_scale,
                        conditioning=conditioning,
                        negative_conditioning=negative_conditioning,
                        sample_size=generation_sample_size,
                        sample_rate=sample_rate,
                        sigma_min=config.sigma_min,
                        sigma_max=config.sigma_max,
                        sampler_type=config.sampler_type,
                        device=device,
                        seed=mini_batch_seed,
                        batch_size=current_mini_batch_size,
                        guidance_rescale=config.guidance_rescale,
                        clap_model=CLAP,
                        c_gram=config.c_gram,
                        gram_neighborhood_scale=config.gram_neighborhood_scale,
                        gram_use_normalized=config.gram_use_normalized,
                        gram_start_step=config.gram_start_step,
                        lambda_min=config.lambda_min,
                        lambda_max=config.lambda_max,
                        constrain_in_sphere=config.constrain_in_sphere,
                        c1=config.c1,
                        c2=config.c2,
                        c3=config.c3,
                        amg_filter_enabled=config.amg_filter_enabled,
                        amg_cutoff_ratio=config.amg_cutoff_ratio,
                        amg_filter_mode=config.amg_filter_mode,
                        enable_spectral_analysis=config.enable_spectral_analysis,
                        spectral_output_dir=spectral_dir,
                        save_latents=config.save_latents,
                        latent_filename=latent_fname,
                        latent_batch_start_idx=latent_start_idx,
                        debug_dir=output_dir,  # Salva latents in output_dir/latents/
                    )
                    
                    # output shape: [mini_batch_size, channels, samples]
                    # Save each item separately
                    for i in range(current_mini_batch_size):
                        batch_idx = global_batch_idx + i
                        
                        # Extract single audio from batch
                        single_output = output[i:i+1]  # Keep batch dim for rearrange
                        single_output = rearrange(single_output, "b d n -> d (b n)")
                        single_output = single_output[:, :requested_samples]
                        
                        output_float = single_output.to(torch.float32)
                        peak = output_float.abs().max().clamp_min(1e-6)
                        output_normalized = (output_float / peak).clamp(-1, 1)
                        
                        # Save file - use gen_N.wav format for cluster mode, sound_id format for random mode
                        if mode == 'cluster' and cluster_info and sound_id in cluster_info:
                            filename = os.path.join(output_dir, f"gen_{batch_idx + 1}.wav")
                            log_label = f"{config.name}/cluster_{cluster_info[sound_id]}/gen_{batch_idx + 1}"
                        else:
                            if BATCH_SIZE == 1:
                                filename = os.path.join(output_dir, f"sound_{sound_id}.wav")
                            else:
                                filename = os.path.join(output_dir, f"sound_{sound_id}_{batch_idx + 1}.wav")
                            log_label = f"{config.name}/sound_{sound_id}_{batch_idx + 1}"
                        
                        output_int16 = output_normalized.mul(32767).to(torch.int16).cpu()
                        torchaudio.save(filename, output_int16, sample_rate)
                        
                        results['success'] += 1
                    
                    global_batch_idx += current_mini_batch_size
                    
                    # Clear GPU cache after each mini-batch to free memory
                    del output
                    torch.cuda.empty_cache()
                
                log_msg(f"Completed {config.name}/{sound_id}: {global_batch_idx} files", "DEBUG")
                
            except Exception as e:
                results['error'] += 1
                error_info = {'config': config.name, 'id': sound_id, 'error': str(e)}
                results['errors'].append(error_info)
                log_msg(f"ERROR {config.name}/{sound_id}: {e}", "ERROR")
                log_msg(traceback.format_exc(), "ERROR")
    
    log_msg(f"Completed! Success: {results['success']}, Errors: {results['error']}")
    return results


def setup_logging():
    """Setup logging to file and console. Returns (logger, log_file_path)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"parallel_generation_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("parallel_generation")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_filename}")
    
    # Also redirect main process stdout/stderr to log
    sys.stdout = TeeWriter(log_filename, sys.__stdout__)
    sys.stderr = TeeWriter(log_filename, sys.__stderr__)
    
    return logger, log_filename


def main():
    """Main function to orchestrate parallel generation."""
    
    # Setup logging
    logger, log_file_path = setup_logging()
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Load data first (needed for cluster mode directory creation)
    logger.info(f"Loading prompts from {INPUT_JSON}...")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    all_ids = list(data.keys())
    total_available = len(all_ids)
    
    # Parse generation source
    mode, source_data = parse_generation_source(GENERATION_SOURCE)
    
    if mode == 'cluster':
        # Cluster mode: load representatives and get IDs
        logger.info(f"CLUSTER MODE: Loading representatives from {CLUSTER_REPRESENTATIVES_CSV}...")
        representatives = load_cluster_representatives(CLUSTER_REPRESENTATIVES_CSV)
        
        cluster_ids = source_data
        selected_ids = []
        cluster_info = {}  # Maps sound_id -> cluster_id
        
        for cluster_id in cluster_ids:
            if cluster_id in representatives:
                rep_id = representatives[cluster_id]
                if rep_id in data:
                    selected_ids.append(rep_id)
                    cluster_info[rep_id] = cluster_id
                    logger.info(f"  Cluster {cluster_id}: representative ID {rep_id}")
                else:
                    logger.warning(f"  Cluster {cluster_id}: representative ID {rep_id} not found in embeddings!")
            else:
                logger.warning(f"  Cluster {cluster_id} not found in representatives CSV!")
        
        logger.info(f"Selected {len(selected_ids)} cluster representatives from clusters {cluster_ids}")
        
    else:
        # Random mode
        num_generations = source_data
        
        if num_generations > total_available:
            logger.warning(f"Requested {num_generations}, available {total_available}.")
            selected_ids = all_ids
        else:
            logger.info(f"RANDOM MODE: Selecting {num_generations} random prompts...")
            if SELECTION_SEED != -1:
                random.seed(SELECTION_SEED)
                logger.info(f"Using fixed selection seed: {SELECTION_SEED}")
            else:
                random.seed()
                logger.info("Using random selection (different IDs each run)")
            selected_ids = random.sample(all_ids, num_generations)
        
        cluster_info = {}  # Not used in random mode
    
    # Create output directories
    for config in CONFIGURATIONS:
        output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config info
        config_file = os.path.join(output_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # In cluster mode, create subdirectories for each cluster with info
        if mode == 'cluster':
            for sound_id, cluster_id in cluster_info.items():
                cluster_dir = os.path.join(output_dir, str(cluster_id))
                os.makedirs(cluster_dir, exist_ok=True)
                
                # Save cluster info
                cluster_info_file = os.path.join(cluster_dir, 'cluster_info.json')
                prompt = data[sound_id]['conditioning']['prompt']
                duration = data[sound_id]['conditioning']['seconds_total']
                cluster_data = {
                    'cluster_id': cluster_id,
                    'representative_id': sound_id,
                    'prompt': prompt,
                    'duration': duration,
                    'batch_size': BATCH_SIZE
                }
                with open(cluster_info_file, 'w') as f:
                    json.dump(cluster_data, f, indent=2)
        else:
            # In random mode, create random_prompts subdirectory
            random_dir = os.path.join(output_dir, "random_prompts")
            os.makedirs(random_dir, exist_ok=True)
    
    # Generate consistent seeds for each sample
    sample_seeds = {}
    for i, sound_id in enumerate(selected_ids):
        if SEED == -1:
            sample_seeds[sound_id] = random.randint(0, 2**32 - 1)
        else:
            sample_seeds[sound_id] = SEED + i
    
    # Save sample list and seeds for reproducibility
    manifest = {
        'mode': mode,
        'generation_source': str(GENERATION_SOURCE),
        'batch_size': BATCH_SIZE,
        'selected_ids': selected_ids,
        'cluster_info': cluster_info if mode == 'cluster' else None,
        'seeds': sample_seeds,
        'configurations': [c.to_dict() for c in CONFIGURATIONS],
    }
    manifest_path = os.path.join(BASE_OUTPUT_DIR, 'generation_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest to {manifest_path}")
    
    # Log all configurations
    logger.debug("=" * 60)
    logger.debug("CONFIGURATIONS:")
    for config in CONFIGURATIONS:
        logger.debug(f"  {config.name}: {config.to_dict()}")
    logger.debug("=" * 60)
    
    # Distribute configurations across GPUs
    num_gpus = len(GPUS)
    configs_per_gpu = [[] for _ in range(num_gpus)]
    
    for i, config in enumerate(CONFIGURATIONS):
        gpu_idx = i % num_gpus
        configs_per_gpu[gpu_idx].append(config)
    
    logger.info(f"\nDistribution across {num_gpus} GPUs:")
    for gpu_idx, gpu_id in enumerate(GPUS):
        config_names = [c.name for c in configs_per_gpu[gpu_idx]]
        logger.info(f"  GPU {gpu_id}: {config_names}")
    
    total_files = len(selected_ids) * len(CONFIGURATIONS) * BATCH_SIZE
    logger.info(f"\nStarting generation:")
    logger.info(f"  - {len(selected_ids)} prompts")
    logger.info(f"  - {len(CONFIGURATIONS)} configurations")
    logger.info(f"  - {BATCH_SIZE} generations per prompt")
    logger.info(f"  - Total files: {total_files}")
    
    # Run in parallel processes
    processes = []
    for gpu_idx, gpu_id in enumerate(GPUS):
        if not configs_per_gpu[gpu_idx]:
            continue
        
        p = mp.Process(
            target=run_sequential_on_gpu,
            args=(gpu_id, configs_per_gpu[gpu_idx], selected_ids, data, sample_seeds, log_file_path, mode, cluster_info)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    # Log completion summary
    logger.info("\n" + "="*60)
    logger.info("GENERATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nOutput directories:")
    total_generated = 0
    for config in CONFIGURATIONS:
        config_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        if mode == 'cluster':
            # Count files in all cluster subdirectories
            count = 0
            if os.path.exists(config_dir):
                for subdir in os.listdir(config_dir):
                    subdir_path = os.path.join(config_dir, subdir)
                    if os.path.isdir(subdir_path):
                        count += len([f for f in os.listdir(subdir_path) if f.endswith('.wav')])
            total_generated += count
            logger.info(f"  {config.name}: {count} files (across {len(cluster_info)} clusters)")
        else:
            # Count files in random_prompts subdirectory
            random_dir = os.path.join(config_dir, "random_prompts")
            count = len([f for f in os.listdir(random_dir) if f.endswith('.wav')]) if os.path.exists(random_dir) else 0
            total_generated += count
            logger.info(f"  {config.name}: {count} files")
    
    logger.info(f"\nTotal files generated: {total_generated}")
    logger.info(f"Expected: {total_files}")


if __name__ == '__main__':
    main()
