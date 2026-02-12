import torch
import os
import json
import re
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import sys

# Import CLAP for audio embeddings
from stable_audio_tools.inference import amg_generation

# --- CONFIGURATION ---
RANDOM_GEN_DIR = "./spectralAnalysis_fad/gram_lp_250_0.3"
EMBEDDINGS_FILE = 'embeddings_new.json'
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

def extract_id_from_filename(filename):
    """Extract the sound ID from filename like 'sound_545.wav' -> '545'"""
    match = re.match(r'sound_(\d+)\.wav', filename)
    if match:
        return match.group(1)
    return None

def main():
    print(f"Using device: {DEVICE}")
    
    # --- LOAD EMBEDDINGS DATA ---
    print("Loading embeddings data...")
    with open(EMBEDDINGS_FILE, 'r') as f:
        embeddings_data = json.load(f)
    
    # --- LOAD CLAP MODEL ---
    print("Loading CLAP model...")
    CLAP = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, device=DEVICE)
    CLAP.load_ckpt()
    
    # --- GET ALL SUBDIRECTORIES ---
    subdirs = []
    for item in os.listdir(RANDOM_GEN_DIR):
        item_path = os.path.join(RANDOM_GEN_DIR, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    
    subdirs = sorted(subdirs)
    print(f"Found {len(subdirs)} subdirectories to process")
    
    results = {}
    
    for subdir in subdirs:
        subdir_path = os.path.join(RANDOM_GEN_DIR, subdir)
        
        # Get all wav files
        wav_files = [f for f in os.listdir(subdir_path) if f.endswith('.wav') and f.startswith('sound_')]
        
        if not wav_files:
            print(f"  {subdir}: No wav files found, skipping...")
            continue
        
        print(f"\nProcessing {subdir} ({len(wav_files)} files)...")
        
        prompt_adherence_scores = []
        processed_count = 0
        skipped_count = 0
        
        for wav_file in tqdm(wav_files, desc=f"  {subdir}"):
            sound_id = extract_id_from_filename(wav_file)
            
            if sound_id is None:
                skipped_count += 1
                continue
            
            if sound_id not in embeddings_data:
                skipped_count += 1
                continue
            
            # Get the prompt for this sound ID
            prompt = embeddings_data[sound_id]['conditioning']['prompt']
            
            # Load the generated audio
            wav_path = os.path.join(subdir_path, wav_file)
            try:
                audio, sr = torchaudio.load(wav_path)
            except Exception as e:
                print(f"    Error loading {wav_file}: {e}")
                skipped_count += 1
                continue
            
            # Resample to 48000 if needed (CLAP expects 48000)
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000)
                audio = resampler(audio)
            
            # Normalize audio
            audio = audio.to(torch.float32)
            peak = audio.abs().max().clamp_min(1e-6)
            audio = (audio / peak).clamp(-1, 1)
            
            # Get mono audio
            mono_audio = audio.mean(dim=0, keepdim=True).to(DEVICE)
            
            with torch.no_grad():
                # Get audio embedding
                audio_emb = CLAP.get_audio_embedding_from_data(x=mono_audio, use_tensor=True)[0]
                audio_emb_norm = F.normalize(audio_emb.unsqueeze(0), p=2, dim=1)
                
                # Get text embedding for the prompt
                text_emb = CLAP.get_text_embedding([prompt], use_tensor=True)
                if isinstance(text_emb, torch.Tensor):
                    text_emb = text_emb.to(DEVICE)
                else:
                    text_emb = torch.tensor(text_emb).to(DEVICE)
                text_emb_norm = F.normalize(text_emb, p=2, dim=1)
                
                # Calculate cosine similarity (prompt adherence)
                similarity = torch.mm(audio_emb_norm, text_emb_norm.T).item()
                prompt_adherence_scores.append(similarity)
                processed_count += 1
        
        if prompt_adherence_scores:
            avg_adherence = sum(prompt_adherence_scores) / len(prompt_adherence_scores)
            results[subdir] = {
                'avg_prompt_adherence': avg_adherence,
                'processed_files': processed_count,
                'skipped_files': skipped_count,
                'min_adherence': min(prompt_adherence_scores),
                'max_adherence': max(prompt_adherence_scores)
            }
        else:
            results[subdir] = {
                'avg_prompt_adherence': 0.0,
                'processed_files': 0,
                'skipped_files': skipped_count,
                'min_adherence': 0.0,
                'max_adherence': 0.0
            }
    
    # --- PRINT RESULTS ---
    print("\n" + "="*60)
    print("       PROMPT ADHERENCE RESULTS       ")
    print("="*60)
    print(f"{'Subdirectory':<35} {'Avg Adherence':>12} {'Files':>8}")
    print("-"*60)
    
    # Sort by avg_prompt_adherence descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_prompt_adherence'], reverse=True)
    
    for subdir, data in sorted_results:
        print(f"{subdir:<35} {data['avg_prompt_adherence']:>12.4f} {data['processed_files']:>8}")
    
    print("="*60)
    
    # --- SAVE RESULTS TO JSON ---
    output_file = os.path.join(RANDOM_GEN_DIR, "prompt_adherence_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
