import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from scipy import signal
import math

def compute_metrics_for_latent(latent_path):
    """
    Carica un latent e calcola HFER e Spectral Entropy.
    """
    try:
        # Carica il tensore (map_location gestisce cpu/gpu)
        latent = torch.load(latent_path, map_location='cpu')
        
        # Gestione dimensioni: vogliamo [Channels, Time] o [Time]
        # Spesso è [1, Channels, Time] -> squeeze
        if latent.ndim == 3:
            latent = latent.squeeze(0)
        
        latent_np = latent.numpy()
        
        # Calcola PSD media sui canali usando Welch
        # nperseg=256 è standard per finestre brevi
        freqs, psd = signal.welch(latent_np, axis=-1, nperseg=256)
        
        # Media sui canali se presenti (shape [C, Freqs] -> [Freqs])
        if psd.ndim > 1:
            mean_psd = np.mean(psd, axis=0)
        else:
            mean_psd = psd

        # Evitiamo divisioni per zero
        total_energy = np.sum(mean_psd) + 1e-12
        
        # --- 1. HFER (High Frequency Energy Ratio) ---
        # Cutoff al 25% della banda (come il tuo filtro LP)
        cutoff_idx = int(len(mean_psd) * 0.25)
        high_freq_energy = np.sum(mean_psd[cutoff_idx:])
        hfer = high_freq_energy / total_energy
        
        # --- 2. Spectral Entropy ---
        # Normalizza PSD come una distribuzione di probabilità
        psd_prob = mean_psd / total_energy
        # Shannon Entropy (in bit)
        spectral_entropy = -np.sum(psd_prob * np.log2(psd_prob + 1e-12))
        # Normalizzata tra 0 e 1 (divisa per log2(N_bins))
        norm_spectral_entropy = spectral_entropy / np.log2(len(mean_psd))

        return hfer, norm_spectral_entropy

    except Exception as e:
        return None, None

def find_latent_folders(base_path, cluster_range=None):
    """
    Trova ricorsivamente tutte le cartelle 'latents' all'interno del path base.
    Se cluster_range è specificato (es. [20, 30]), filtra solo i cluster in quel range.
    Ritorna una lista di percorsi alle cartelle latents.
    """
    latent_folders = []
    for root, dirs, files in os.walk(base_path):
        if 'latents' in dirs:
            latent_path = os.path.join(root, 'latents')
            # Controlla se il parent folder è un numero (cluster ID)
            parent_name = os.path.basename(root)
            if cluster_range is not None and parent_name.isdigit():
                cluster_id = int(parent_name)
                if cluster_id < cluster_range[0] or cluster_id > cluster_range[1]:
                    continue  # Salta cluster fuori dal range
            latent_folders.append(latent_path)
        # Se la cartella stessa si chiama 'latents' e contiene file .pt
        if os.path.basename(root) == 'latents':
            pt_files = [f for f in files if f.endswith('.pt') or f.endswith('.pth')]
            if pt_files and root not in latent_folders:
                # Controlla parent del parent per cluster ID
                parent_name = os.path.basename(os.path.dirname(root))
                if cluster_range is not None and parent_name.isdigit():
                    cluster_id = int(parent_name)
                    if cluster_id < cluster_range[0] or cluster_id > cluster_range[1]:
                        continue
                latent_folders.append(root)
    return latent_folders

def analyze_folder(folder_path, label, limit=None, cluster_range=None):
    if not folder_path or not os.path.exists(folder_path):
        return None

    # Cerca cartelle latents ricorsivamente
    latent_folders = find_latent_folders(folder_path, cluster_range)
    
    # Se non trova sottocartelle latents, usa il path diretto
    if not latent_folders:
        latent_folders = [folder_path]
    
    files = []
    for lf in latent_folders:
        for root, _, filenames in os.walk(lf):
            for f in filenames:
                if f.endswith('.pt') or f.endswith('.pth'):
                    files.append(os.path.join(root, f))
    
    if limit:
        import random
        random.shuffle(files)
        files = files[:limit]
        
    print(f"\nProcessing [{label}]: found {len(files)} files in {len(latent_folders)} latent folder(s)...")
    
    hfer_list = []
    entropy_list = []
    
    for f in tqdm(files, leave=False):
        h, e = compute_metrics_for_latent(f)
        if h is not None:
            hfer_list.append(h)
            entropy_list.append(e)
            
    if not hfer_list:
        print("  -> No valid data computed.")
        return None

    return {
        'label': label,
        'count': len(hfer_list),
        'hfer_mean': np.mean(hfer_list),
        'hfer_std': np.std(hfer_list),
        'entr_mean': np.mean(entropy_list),
        'entr_std': np.std(entropy_list)
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate Spectral Metrics for Latents")
    
    # Accetta fino a 4 configurazioni opzionali
    parser.add_argument("--conf1", type=str, help="Path for Config 1")
    parser.add_argument("--label1", type=str, default="Config 1", help="Label for Config 1")
    
    parser.add_argument("--conf2", type=str, help="Path for Config 2")
    parser.add_argument("--label2", type=str, default="Config 2", help="Label for Config 2")
    
    parser.add_argument("--conf3", type=str, help="Path for Config 3")
    parser.add_argument("--label3", type=str, default="Config 3", help="Label for Config 3")
    
    parser.add_argument("--conf4", type=str, help="Path for Config 4")
    parser.add_argument("--label4", type=str, default="Config 4", help="Label for Config 4")
    
    parser.add_argument("--limit", type=int, default=None, help="Limit max files per folder (optional)")
    parser.add_argument("--clusters", type=int, nargs=2, default=None, metavar=('START', 'END'),
                       help="Range di cluster da analizzare [START, END] inclusi (es. --clusters 20 30)")
    
    args = parser.parse_args()
    
    cluster_range = args.clusters  # Sarà None o [start, end]
    
    configs = [
        (args.conf1, args.label1),
        (args.conf2, args.label2),
        (args.conf3, args.label3),
        (args.conf4, args.label4)
    ]
    
    results = []
    for path, label in configs:
        if path:
            res = analyze_folder(path, label, args.limit, cluster_range)
            if res:
                results.append(res)
    
    # --- STAMPA TABELLA LEGGIBILE ---
    print("\n" + "="*85)
    print(f"{'CONFIGURATION':<25} | {'N. SAMPLES':<10} | {'HFER (Mean ± Std)':<22} | {'ENTROPY (Mean ± Std)':<22}")
    print("-" * 85)
    
    for r in results:
        hfer_str = f"{r['hfer_mean']:.4f} ± {r['hfer_std']:.4f}"
        entr_str = f"{r['entr_mean']:.4f} ± {r['entr_std']:.4f}"
        print(f"{r['label']:<25} | {r['count']:<10} | {hfer_str:<22} | {entr_str:<22}")
    print("="*85 + "\n")

    # --- STAMPA FORMATO LATEX ---
    print("--- LaTeX Body Snippet ---\n")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{\\textbf{Latent Spectral Analysis.} Comparison of High-Frequency Energy Ratio (HFER) and Spectral Entropy (SE).}")
    print("\\label{tab:spectral_metrics}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("\\textbf{Configuration} & \\textbf{HFER} ($\\downarrow$) & \\textbf{Spectral Entropy} ($\\downarrow$) \\\\")
    print("\\midrule")
    
    for r in results:
        # Grassetto automatico per il valore più basso (migliore) tra raw e lp se vuoi, 
        # qui stampiamo semplice
        print(f"{r['label']} & {r['hfer_mean']:.3f} $\\pm$ {r['hfer_std']:.3f} & {r['entr_mean']:.3f} $\\pm$ {r['entr_std']:.3f} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()