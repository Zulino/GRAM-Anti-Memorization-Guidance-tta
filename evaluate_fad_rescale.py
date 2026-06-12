"""Compute Fréchet Audio Distance (FAD) between soundDataset (baseline) and random_gen/rescale (evaluation).

This script:
1. Reads generated audio files from random_gen/rescale/
2. Finds corresponding baseline files in soundDataset/ using the same ID (e.g., sound_476.wav)
3. Computes FAD between the matched pairs

Usage:
    python evaluate_fad_rescale.py --model vggish
    python evaluate_fad_rescale.py --model clap-laion-music
    python evaluate_fad_rescale.py --model MERT-v1-95M
    python evaluate_fad_rescale.py --model all  # Run all three models

Available models:
    - vggish: VGGish (AudioSet baseline)
    - clap-2023: Microsoft CLAP original
    - clap-laion-audio: CLAP trained on general audio
    - clap-laion-music: CLAP music (HTSAT-base) with custom music_audioset checkpoint
    - MERT-v1-95M: MERT full model (all layers concatenated, recommended)
    - MERT-v1-95M-{1-11}: MERT individual layers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


def extract_id_from_filename(filename: str) -> Optional[str]:
    """Extract the numeric ID from a filename like sound_476.wav -> '476'"""
    match = re.match(r'sound_(\d+)\.wav', filename)
    if match:
        return match.group(1)
    return None


def find_matched_files(eval_dir: Path, baseline_dir: Path) -> Tuple[List[str], List[str]]:
    """Find evaluation files and their corresponding baseline files.
    
    Returns:
        Tuple of (eval_files, ref_files) where both lists are matched by ID
    """
    eval_files = []
    ref_files = []
    
    # Get all evaluation wav files
    eval_wavs = sorted(eval_dir.glob('*.wav'))
    
    for eval_wav in eval_wavs:
        audio_id = extract_id_from_filename(eval_wav.name)
        if audio_id is None:
            print(f"[WARN] Could not extract ID from: {eval_wav.name}")
            continue
        
        # Find corresponding baseline file
        baseline_wav = baseline_dir / f'sound_{audio_id}.wav'
        if baseline_wav.exists():
            eval_files.append(str(eval_wav))
            ref_files.append(str(baseline_wav))
        else:
            print(f"[WARN] No baseline file found for ID {audio_id}")
    
    return eval_files, ref_files


# Minimum audio duration in seconds for VGGish (uses 0.96s windows)
VGGISH_MIN_DURATION = 1.0


def _pad_audio_if_needed(file_path: str, model_name: str, target_sr: int, min_duration: float, temp_dir: str) -> str:
    """Pad audio with zeros if too short for the model.
    
    Returns the original path if no padding needed, or a path to a padded temp file.
    """
    import soundfile as sf
    
    # Only VGGish needs padding (other models handle short audio internally)
    if model_name.lower() != 'vggish':
        return file_path
    
    try:
        data, sr = sf.read(file_path)
        
        # Convert stereo to mono if needed
        if data.ndim == 2:
            data = data.mean(axis=1)
        
        duration = len(data) / sr
        
        if duration >= min_duration:
            return file_path
        
        # Calculate samples needed for target duration at target sample rate
        # Resample if needed
        if sr != target_sr:
            import scipy.signal
            num_samples = int(len(data) * target_sr / sr)
            data = scipy.signal.resample(data, num_samples)
            sr = target_sr
        
        # Calculate padding needed
        target_samples = int(min_duration * sr)
        current_samples = len(data)
        
        if current_samples < target_samples:
            # Pad with zeros at the end
            padding = np.zeros(target_samples - current_samples, dtype=data.dtype)
            data = np.concatenate([data, padding])
        
        # Save to temp file
        temp_path = os.path.join(temp_dir, os.path.basename(file_path))
        sf.write(temp_path, data, sr)
        return temp_path
        
    except Exception:
        # If anything fails, return original path and let fadtk handle it
        return file_path


def compute_fad_clap_music(
    ref_files: List[str],
    eval_files: List[str],
    verbose: bool = True,
    reduce_dim: Optional[int] = None,
    reduce_method: str = 'pca',
    reduce_fit: str = 'reference',
) -> float:
    """Compute FAD using CLAP laion-music (HTSAT-base) with custom checkpoint.

    The model is loaded manually via laion_clap instead of fadtk,
    using the music_audioset checkpoint stored locally.
    """
    import torch
    import torchaudio
    from stable_audio_tools.inference import amg_generation
    from fadtk.fad import calc_embd_statistics, calc_frechet_distance

    if len(ref_files) == 0:
        raise SystemExit('No reference files provided.')
    if len(eval_files) == 0:
        raise SystemExit('No evaluation audio files found.')

    if verbose:
        print(f"[STEP] Starting FAD compute | model='clap-laion-music' (custom checkpoint), "
              f"eval_files={len(eval_files)}, ref_files={len(ref_files)}")

    # --- Load CLAP model with custom checkpoint ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        'model_cache/clap_checkpoints/music_audioset_epoch_15_esc_90.14.pt'
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CLAP checkpoint not found at {checkpoint_path}")

    CLAP = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    CLAP.load_ckpt(checkpoint_path)
    CLAP.eval()
    for param in CLAP.parameters():
        param.requires_grad = False

    if verbose:
        print(f"[STEP] Loaded CLAP music model (HTSAT-base, custom checkpoint from {checkpoint_path})")

    # --- Helper: extract embedding from a single audio file ---
    def _extract_embedding(file_path: str) -> Optional[np.ndarray]:
        try:
            audio, sr = torchaudio.load(file_path)
            if sr != 48000:
                audio = torchaudio.functional.resample(audio, sr, 48000)
            audio = audio.to(torch.float32)
            peak = audio.abs().max().clamp_min(1e-6)
            audio = (audio / peak).clamp(-1, 1)
            mono_audio = audio.mean(dim=0, keepdim=True).to(device)
            with torch.no_grad():
                emb = CLAP.get_audio_embedding_from_data(mono_audio, use_tensor=True)
            return emb.cpu().numpy().flatten()
        except Exception as e:
            if verbose:
                print(f"[WARN] Error extracting embedding from {file_path}: {e}")
            return None

    # --- Extract embeddings for eval and ref sets ---
    def _extract_all(files: List[str], label: str) -> np.ndarray:
        embs: List[np.ndarray] = []
        errors = 0
        if verbose:
            print(f"[STEP] Extracting {label} embeddings... files={len(files)}")
        for f in files:
            vec = _extract_embedding(f)
            if vec is not None:
                embs.append(vec)
            else:
                errors += 1
        if not embs:
            raise SystemExit(f'No {label} embeddings could be extracted.')
        mat = np.stack(embs, axis=0)
        if verbose:
            print(f"[STEP] {label.capitalize()} embeddings ready | loaded={len(embs)}, errors={errors}, shape={mat.shape}")
        return mat

    eval_mat = _extract_all(eval_files, 'evaluation')
    ref_mat = _extract_all(ref_files, 'reference')

    # --- Optional dimensionality reduction (same logic as compute_fad) ---
    if reduce_dim is not None and reduce_dim > 0 and reduce_dim < ref_mat.shape[1]:
        orig_D = ref_mat.shape[1]
        fit_source = reduce_fit.lower()

        if reduce_method.lower() == 'pca':
            if fit_source == 'both':
                fit_mat = np.vstack([ref_mat, eval_mat])
            else:
                fit_mat = ref_mat
            mean = fit_mat.mean(axis=0)
            Xc = fit_mat - mean
            try:
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            except np.linalg.LinAlgError:
                cov = np.cov(Xc, rowvar=False)
                S, V = np.linalg.eigh(cov)
                idx = np.argsort(S)[::-1]
                Vt = V[:, idx].T
            W = Vt[:reduce_dim].T
            ref_mat = (ref_mat - mean) @ W
            eval_mat = (eval_mat - mean) @ W
        elif reduce_method.lower() in ('random', 'rp'):
            from numpy.random import default_rng
            rng = default_rng(42)
            W = rng.standard_normal((orig_D, reduce_dim))
            W, _ = np.linalg.qr(W)
            W = W[:, :reduce_dim]
            ref_mat = ref_mat @ W
            eval_mat = eval_mat @ W

        if verbose:
            print(f"[STEP] Dim reduction: {reduce_method}, {orig_D} -> {ref_mat.shape[1]}")

    # --- Compute FAD ---
    mu_ref, cov_ref = calc_embd_statistics(ref_mat)
    mu_eval, cov_eval = calc_embd_statistics(eval_mat)

    if verbose:
        print(f"[STEP] Stats | mu_ref={mu_ref.shape}, cov_ref={cov_ref.shape}")

    fad_score = float(calc_frechet_distance(mu_ref, cov_ref, mu_eval, cov_eval))
    return fad_score


def compute_fad(
    ref_files: List[str],
    eval_files: List[str],
    model_name: str = 'vggish',
    verbose: bool = True,
    reduce_dim: Optional[int] = None,
    reduce_method: str = 'pca',
    reduce_fit: str = 'reference',
) -> float:
    """Compute FAD using fadtk."""
    
    if len(ref_files) == 0:
        raise SystemExit('No reference files provided.')
    if len(eval_files) == 0:
        raise SystemExit('No evaluation audio files found.')
    
    if verbose:
        print(f"[STEP] Starting FAD compute | model='{model_name}', eval_files={len(eval_files)}, ref_files={len(ref_files)}")
    
    try:
        from fadtk.model_loader import get_all_models
        from fadtk.fad import FrechetAudioDistance, calc_embd_statistics, calc_frechet_distance
        
        all_models = get_all_models()
        target = str(model_name).lower()
        ml = None
        for m in all_models:
            if getattr(m, 'name', '').lower() == target:
                ml = m
                break
        
        if ml is None:
            available = ', '.join(sorted(getattr(m, 'name', '?') for m in all_models))
            raise SystemExit(f'Model "{model_name}" not found. Available in fadtk: {available}')
        
        if verbose:
            print(f"[STEP] Using fadtk model loader: '{getattr(ml, 'name', ml)}'")
        
        fad = FrechetAudioDistance(ml)
        
    except Exception as e:
        raise SystemExit(f"fadtk is not available or get_all_models failed. Original error: {e}")
    
    # Create temp directory for padded files (only used for VGGish)
    temp_dir = None
    file_mapping = {}  # Maps original path -> potentially padded path
    
    try:
        if model_name.lower() == 'vggish':
            temp_dir = tempfile.mkdtemp(prefix='fad_padded_')
            if verbose:
                print(f"[STEP] Checking audio durations for VGGish (min {VGGISH_MIN_DURATION}s)...")
            
            padded_count = 0
            for f in ref_files + eval_files:
                padded_path = _pad_audio_if_needed(f, model_name, ml.sr, VGGISH_MIN_DURATION, temp_dir)
                file_mapping[f] = padded_path
                if padded_path != f:
                    padded_count += 1
            
            if verbose and padded_count > 0:
                print(f"[STEP] Zero-padded {padded_count} short audio files to {VGGISH_MIN_DURATION}s")
        else:
            # No padding needed for other models
            for f in ref_files + eval_files:
                file_mapping[f] = f
        
        # Cache embeddings
        if verbose:
            print(f"[STEP] Caching reference embeddings... files={len(ref_files)}")
        for f in ref_files:
            try:
                fad.cache_embedding_file(file_mapping[f])
            except Exception as ex:
                if verbose:
                    print(f"[WARN] Error caching ref {f}: {ex}")
        
        if verbose:
            print(f"[STEP] Caching evaluation embeddings... files={len(eval_files)}")
        for f in eval_files:
            try:
                fad.cache_embedding_file(file_mapping[f])
            except Exception as ex:
                if verbose:
                    print(f"[WARN] Error caching eval {f}: {ex}")
        
        def _to_2d(a: np.ndarray) -> np.ndarray:
            """Ensure embeddings are 2D [T, D]."""
            a = np.asarray(a)
            if a.ndim == 1:
                return a.reshape(1, -1)
            if a.ndim > 2:
                return a.reshape(-1, a.shape[-1])
            return a
        
        # Load evaluation embeddings
        eval_embs: List[np.ndarray] = []
        eval_D: Optional[int] = None
        _eval_loaded = 0
        _eval_mismatch = 0
        _eval_errors = 0
        
        for f in eval_files:
            try:
                arr = fad.read_embedding_file(file_mapping[f])
                arr2 = _to_2d(arr)
                if eval_D is None:
                    eval_D = arr2.shape[1]
                elif arr2.shape[1] != eval_D:
                    _eval_mismatch += 1
                    continue
                mean_vec = arr2.mean(axis=0, keepdims=True)
                eval_embs.append(mean_vec)
                _eval_loaded += 1
            except Exception:
                _eval_errors += 1
        
        if not eval_embs:
            raise SystemExit('No evaluation embeddings could be loaded.')
        
        eval_mat = np.concatenate(eval_embs, axis=0)
        if verbose:
            print(f"[STEP] Eval embeddings ready | loaded={_eval_loaded}, mismatch={_eval_mismatch}, errors={_eval_errors}, shape={eval_mat.shape}")
        
        # Load reference embeddings
        ref_embs: List[np.ndarray] = []
        ref_D: Optional[int] = None
        _ref_loaded = 0
        _ref_mismatch = 0
        _ref_errors = 0
        
        for f in ref_files:
            try:
                arr = fad.read_embedding_file(file_mapping[f])
                arr2 = _to_2d(arr)
                if ref_D is None:
                    ref_D = arr2.shape[1]
                elif arr2.shape[1] != ref_D:
                    _ref_mismatch += 1
                    continue
                mean_vec = arr2.mean(axis=0, keepdims=True)
                ref_embs.append(mean_vec)
                _ref_loaded += 1
            except Exception:
                _ref_errors += 1
        
        if not ref_embs:
            raise SystemExit('No reference embeddings could be loaded.')
        
        ref_mat = np.concatenate(ref_embs, axis=0)
        if verbose:
            print(f"[STEP] Ref embeddings ready | loaded={_ref_loaded}, mismatch={_ref_mismatch}, errors={_ref_errors}, shape={ref_mat.shape}")
        
        # Optional dimensionality reduction
        if reduce_dim is not None and reduce_dim > 0 and reduce_dim < ref_mat.shape[1]:
            orig_D = ref_mat.shape[1]
            fit_source = reduce_fit.lower()
            
            if reduce_method.lower() == 'pca':
                if fit_source == 'both':
                    fit_mat = np.vstack([ref_mat, eval_mat])
                else:
                    fit_mat = ref_mat
                mean = fit_mat.mean(axis=0)
                Xc = fit_mat - mean
                try:
                    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                except np.linalg.LinAlgError:
                    cov = np.cov(Xc, rowvar=False)
                    S, V = np.linalg.eigh(cov)
                    idx = np.argsort(S)[::-1]
                    Vt = V[:, idx].T
                W = Vt[:reduce_dim].T
                ref_mat = (ref_mat - mean) @ W
                eval_mat = (eval_mat - mean) @ W
            elif reduce_method.lower() in ('random', 'rp'):
                from numpy.random import default_rng
                rng = default_rng(42)
                W = rng.standard_normal((orig_D, reduce_dim))
                W, _ = np.linalg.qr(W)
                W = W[:, :reduce_dim]
                ref_mat = ref_mat @ W
                eval_mat = eval_mat @ W
            
            if verbose:
                print(f"[STEP] Dim reduction: {reduce_method}, {orig_D} -> {ref_mat.shape[1]}")
        
        # Compute statistics and FAD
        mu_ref, cov_ref = calc_embd_statistics(ref_mat)
        mu_eval, cov_eval = calc_embd_statistics(eval_mat)
        
        if verbose:
            print(f"[STEP] Stats | mu_ref={mu_ref.shape}, cov_ref={cov_ref.shape}")
        
        fad_score = float(calc_frechet_distance(mu_ref, cov_ref, mu_eval, cov_eval))
        return fad_score
    
    finally:
        # Clean up temporary directory with padded files
        if temp_dir is not None and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    # --------------- Argument parsing --------------- #
    parser = argparse.ArgumentParser(
        description='Compute FAD between baseline and generated audio files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  vggish            VGGish (AudioSet baseline, requires ~1s audio)
  clap-2023         Microsoft CLAP original
  clap-laion-audio  CLAP trained on general audio (LAION-Audio-630K)
  clap-laion-music  CLAP music (HTSAT-base) with custom music_audioset checkpoint
  MERT-v1-95M       MERT full model - all layers concatenated (recommended)
  MERT-v1-95M-{N}   MERT individual layer N (1-11, lower=acoustic, higher=semantic)
  all               Run vggish, clap-laion-music, and MERT-v1-95M

Examples:
  python evaluate_fad_rescale.py --model vggish
  python evaluate_fad_rescale.py --model clap-laion-music --eval-dir ./my_generated/
  python evaluate_fad_rescale.py --model all
        """
    )
    parser.add_argument('--model', '-m', type=str, default='vggish',
                        help='Model to use for FAD computation (default: vggish). "all" runs a set of common models.')
    parser.add_argument('--eval-dir', '-e', type=str, default=None,
                        help='Directory with generated audio files to evaluate')
    parser.add_argument('--baseline-dir', '-b', type=str, default=None,
                        help='Directory with baseline/reference audio files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file path (default: fad_rescale_{model}.json)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--reduce-dim', type=int, default=None,
                        help='Optional dimensionality reduction (e.g., 128, 256)')
    
    args = parser.parse_args()
    
    # --------------- Configuration --------------- #
    WORKSPACE = Path('/mnt/media/HDD_4TB/riccardo/GRAM-AMG')
    EVAL_DIR = Path(args.eval_dir) if args.eval_dir else WORKSPACE / 'rescale_studies' / 'gram_rescale2' / 'random_prompts'
    BASELINE_DIR = Path(args.baseline_dir) if args.baseline_dir else WORKSPACE / 'soundDataset'
    
    SR = 16000
    REDUCE_DIM: Optional[int] = args.reduce_dim
    REDUCE_METHOD: str = 'pca'
    REDUCE_FIT: str = 'reference'
    QUIET = args.quiet
    
    # Handle 'all' model option
    if args.model.lower() == 'all':
        models_to_run = ['vggish', 'clap-laion-audio', 'clap-laion-music', 'MERT-v1-95M']
    else:
        models_to_run = [args.model]
    
    # --------------- Find matched files --------------- #
    if not QUIET:
        print(f'[INFO] Evaluation dir: {EVAL_DIR}')
        print(f'[INFO] Baseline dir:   {BASELINE_DIR}')
    
    eval_files, ref_files = find_matched_files(EVAL_DIR, BASELINE_DIR)
    
    if not QUIET:
        print(f'[INFO] Matched files: {len(eval_files)}')
    
    if len(eval_files) == 0:
        raise SystemExit('No matched files found!')
    
    # --------------- FAD computation for each model --------------- #
    results = {}
    
    for MODEL in models_to_run:
        if not QUIET:
            print(f'\n{"="*60}')
            print(f'[INFO] Computing FAD with model: {MODEL}')
            print(f'{"="*60}')
        
        try:
            if MODEL == 'clap-laion-music':
                fad_score = compute_fad_clap_music(
                    ref_files,
                    eval_files,
                    verbose=(not QUIET),
                    reduce_dim=REDUCE_DIM,
                    reduce_method=REDUCE_METHOD,
                    reduce_fit=REDUCE_FIT,
                )
            else:
                fad_score = compute_fad(
                    ref_files,
                    eval_files,
                    model_name=MODEL,
                    verbose=(not QUIET),
                    reduce_dim=REDUCE_DIM,
                    reduce_method=REDUCE_METHOD,
                    reduce_fit=REDUCE_FIT,
                )
            
            print(f'[RESULT] FAD ({MODEL}, sr={SR}) = {fad_score:.6f}')
            results[MODEL] = fad_score
            
        except Exception as e:
            print(f'[ERROR] Failed to compute FAD with {MODEL}: {e}')
            results[MODEL] = None
    
    # --------------- Output results --------------- #
    if args.output:
        output_path = Path(args.output)
    elif len(models_to_run) == 1:
        output_path = WORKSPACE / f'fad_rescale_{models_to_run[0].replace("-", "_")}.json'
    else:
        output_path = WORKSPACE / 'fad_rescale_all.json'
    
    os.makedirs(output_path.parent, exist_ok=True)
    payload = {
        'eval_dir': str(EVAL_DIR),
        'baseline_dir': str(BASELINE_DIR),
        'sample_rate': SR,
        'n_eval_files': len(eval_files),
        'n_ref_files': len(ref_files),
        'results': results,
    }
    # For backwards compatibility when single model
    if len(models_to_run) == 1:
        payload['model'] = models_to_run[0]
        payload['fad'] = results.get(models_to_run[0])
    
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    if not QUIET:
        print(f'\n[INFO] Saved JSON: {output_path}')
    
    # Print summary if multiple models
    if len(models_to_run) > 1:
        print(f'\n{"="*60}')
        print('SUMMARY')
        print(f'{"="*60}')
        for model, score in results.items():
            if score is not None:
                print(f'  {model:20s}: {score:.6f}')
            else:
                print(f'  {model:20s}: FAILED')


if __name__ == '__main__':
    main()
