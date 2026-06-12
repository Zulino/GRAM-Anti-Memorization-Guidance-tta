import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
import numpy as np

# Importa lo stesso modulo usato per la guidance per garantire coerenza
from stable_audio_tools.inference import amg_generation

def load_clap_model(device):
    """Carica il modello CLAP usato per la AMG."""
    print(f"[INFO] Loading CLAP model on {device}...")
    # enable_fusion=False è lo standard usato in amg_generation
    model = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, device=device)
    model.load_ckpt()
    model.eval()
    return model

def get_audio_embedding(model, audio_path, device):
    """Carica l'audio, lo preprocessa e calcola l'embedding CLAP."""
    try:
        # Carica audio
        wav, sr = torchaudio.load(audio_path)
        
        # CLAP vuole 48kHz
        if sr != 48000:
            resampler = T.Resample(sr, 48000)
            wav = resampler(wav)
        
        # Converti a Mono se necessario
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        # Peak Normalization (come in amg_infer.py)
        # Assicura che il volume non influenzi troppo l'embedding
        peak = wav.abs().max().clamp_min(1e-6)
        wav = (wav / peak).clamp(-1, 1)

        # Sposta su GPU
        wav = wav.to(device)

        # Calcola embedding
        with torch.no_grad():
            # x vuole [batch, time], quindi aggiungiamo dimensione batch se serve
            # get_audio_embedding_from_data gestisce internamente il formato, 
            # ma passiamo use_tensor=True per velocità
            emb = model.get_audio_embedding_from_data(x=wav, use_tensor=True)
            
            # Normalizza il vettore (importante per cosine similarity)
            emb = F.normalize(emb, p=2, dim=1)
            
        return emb

    except Exception as e:
        print(f"[ERROR] Skipping {audio_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Find the most similar audio in a folder using CLAP.")
    parser.add_argument("--ref", type=str, required=True, help="Path to the reference audio file.")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing candidate audio files.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top matches to display.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()

    # Verifica percorsi
    if not os.path.exists(args.ref):
        raise FileNotFoundError(f"Reference file not found: {args.ref}")
    if not os.path.exists(args.folder):
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    # Carica Modello
    model = load_clap_model(args.device)

    # 1. Calcola embedding di riferimento
    print(f"[INFO] Processing reference: {os.path.basename(args.ref)}")
    ref_emb = get_audio_embedding(model, args.ref, args.device)
    if ref_emb is None:
        return

    # 2. Scansiona la cartella
    valid_extensions = ('.wav', '.mp3', '.flac', '.ogg')
    candidates = []
    
    files = [f for f in os.listdir(args.folder) if f.lower().endswith(valid_extensions)]
    print(f"[INFO] Scanning {len(files)} files in {args.folder}...")

    results = []

    for fname in tqdm(files):
        fpath = os.path.join(args.folder, fname)
        
        # Calcola embedding candidato
        cand_emb = get_audio_embedding(model, fpath, args.device)
        
        if cand_emb is not None:
            # Calcola Cosine Similarity: dot product tra vettori normalizzati
            # ref_emb shape: [1, 512], cand_emb shape: [1, 512]
            score = torch.mm(ref_emb, cand_emb.T).item()
            results.append((fname, score))

    # 3. Ordina e mostra risultati
    # Ordina decrescente per score (più alto = più simile)
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*50}")
    print(f"TOP {args.top_k} MOST SIMILAR FILES")
    print(f"{'='*50}")
    print(f"Reference: {os.path.basename(args.ref)}")
    print(f"{'-'*50}")
    print(f"{'Score':<10} | {'Filename'}")
    print(f"{'-'*50}")
    
    for i, (fname, score) in enumerate(results[:args.top_k]):
        print(f"{score:.4f}     | {fname}")
        
    # Opzionale: salva il path del vincitore in un file txt per uso futuro
    if results:
        best_file = results[0][0]
        print(f"\n[WINNER] {best_file} (Score: {results[0][1]:.4f})")

if __name__ == "__main__":
    main()