import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import interp1d

# Colori per le 4 configurazioni
COLORS = ['#7f7f7f', '#2ca02c', '#d62728', '#ff7f0e']
LINESTYLES = ['--', '-', '-', '-.']

# Griglia di frequenze comune per interpolazione
COMMON_FREQ_BINS = 129  # Output standard di Welch con nperseg=256


def find_latent_folders(base_path, cluster_range=None):
    """
    Trova ricorsivamente tutte le cartelle 'latents' all'interno del path base.
    Se cluster_range è specificato (es. [20, 30]), filtra solo i cluster in quel range.
    """
    latent_folders = []
    for root, dirs, files in os.walk(base_path):
        if 'latents' in dirs:
            latent_path = os.path.join(root, 'latents')
            parent_name = os.path.basename(root)
            if cluster_range is not None and parent_name.isdigit():
                cluster_id = int(parent_name)
                if cluster_id < cluster_range[0] or cluster_id > cluster_range[1]:
                    continue
            latent_folders.append(latent_path)
        if os.path.basename(root) == 'latents':
            pt_files = [f for f in files if f.endswith('.pt') or f.endswith('.pth')]
            if pt_files and root not in latent_folders:
                parent_name = os.path.basename(os.path.dirname(root))
                if cluster_range is not None and parent_name.isdigit():
                    cluster_id = int(parent_name)
                    if cluster_id < cluster_range[0] or cluster_id > cluster_range[1]:
                        continue
                latent_folders.append(root)
    return latent_folders


def load_latents_and_compute_psd(folder_path, label, limit=None, cluster_range=None):
    """
    Carica i file latent da tutte le cartelle latents trovate,
    calcola la PSD media e restituisce i risultati.
    """
    if not folder_path or not os.path.exists(folder_path):
        print(f"Warning: Path {folder_path} does not exist")
        return None
    
    # Cerca cartelle latents ricorsivamente
    latent_folders = find_latent_folders(folder_path, cluster_range)
    
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
    
    psd_accumulator = []
    hfer_scores = []
    
    cutoff_ratio = 0.25
    common_freqs = np.linspace(0, 0.5, COMMON_FREQ_BINS)
    
    for fpath in tqdm(files, leave=False):
        try:
            latent = torch.load(fpath, map_location='cpu')
            if len(latent.shape) == 3:
                latent = latent.squeeze(0)
            
            latent_np = latent.numpy()
            
            # Salta latent troppo corti (minimo 8 timestep)
            seq_len = latent_np.shape[-1]
            if seq_len < 8:
                continue
            
            # Adatta nperseg alla lunghezza del segnale
            nperseg = min(256, seq_len)
            
            freqs, psd = signal.welch(latent_np, axis=-1, nperseg=nperseg)
            freqs_normalized = freqs / (2 * freqs[-1]) if freqs[-1] > 0 else freqs
            
            mean_channel_psd = np.mean(psd, axis=0)
            mean_channel_psd /= (np.sum(mean_channel_psd) + 1e-10)
            
            if len(freqs_normalized) > 1:
                interp_func = interp1d(freqs_normalized, mean_channel_psd, 
                                       kind='linear', fill_value='extrapolate')
                interpolated_psd = interp_func(common_freqs)
                interpolated_psd = np.clip(interpolated_psd, 0, None)
            else:
                continue
            
            psd_accumulator.append(interpolated_psd)
            
            cutoff_idx = int(len(interpolated_psd) * cutoff_ratio)
            high_freq_energy = np.sum(interpolated_psd[cutoff_idx:])
            total_energy = np.sum(interpolated_psd)
            hfer_scores.append(high_freq_energy / (total_energy + 1e-10))
            
        except Exception as e:
            pass  # Silently skip errors

    if len(psd_accumulator) == 0:
        print(f"  -> No valid latents found for {label}")
        return None
    
    psds = np.array(psd_accumulator)
    mean_psd = np.mean(psds, axis=0)
    std_psd = np.std(psds, axis=0)
    
    print(f"  Used {len(psd_accumulator)}/{len(files)} latents")
    
    return {
        'label': label,
        'freqs': common_freqs,
        'mean': mean_psd,
        'std': std_psd,
        'hfer': np.mean(hfer_scores),
        'count': len(psd_accumulator)
    }


def plot_spectral_analysis(results, output_file, title_suffix=None):
    """Plotta tutti gli spettri delle configurazioni."""
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    for i, res in enumerate(results):
        freqs = res['freqs']
        mean = res['mean']
        std = res['std']
        color = COLORS[i % len(COLORS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]
        
        plt.plot(freqs, mean, 
                 label=f"{res['label']} (HFER: {res['hfer']:.2f})",
                 color=color, linestyle=linestyle, linewidth=2)
        
        plt.fill_between(freqs, mean - std/2, mean + std/2, 
                         color=color, alpha=0.15)

    plt.axvline(x=0.25 * 0.5, color='k', linestyle=':', alpha=0.5, label='LP Cutoff (25%)')
    
    title = 'Global Latent Space Spectral Density'
    if title_suffix:
        title += f' {title_suffix}'
    plt.title(title)
    plt.xlabel('Normalized Frequency (Nyquist=0.5)')
    plt.ylabel('Power Spectral Density (Normalized)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    base_path = os.path.splitext(output_file)[0]
    plt.savefig(f"{base_path}.pdf", dpi=300)
    plt.savefig(f"{base_path}.png", dpi=300)
    print(f"Saved spectrum plots to {base_path}.pdf and {base_path}.png")
    plt.close()


def plot_differential_analysis(results, output_file, title_suffix=None):
    """Plotta le differenze rispetto alla prima configurazione (riferimento)."""
    if len(results) < 2:
        print("Need at least 2 configurations for differential plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    ref = results[0]
    ref_mean = ref['mean']
    freqs = ref['freqs']
    
    for i, res in enumerate(results[1:], start=1):
        diff = res['mean'] - ref_mean
        color = COLORS[i % len(COLORS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]
        
        plt.plot(freqs, diff, 
                 label=f"{res['label']} - {ref['label']}",
                 color=color, linestyle=linestyle, linewidth=2)

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0.25 * 0.5, color='k', linestyle=':', alpha=0.5, label='LP Cutoff (25%)')
    
    title = f'Spectral Energy Excess relative to {ref["label"]}'
    if title_suffix:
        title += f' {title_suffix}'
    plt.title(title)
    plt.xlabel('Normalized Frequency (Nyquist=0.5)')
    plt.ylabel('Relative Power (Δ PSD)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    base_path = os.path.splitext(output_file)[0]
    plt.savefig(f"{base_path}_diff.pdf", dpi=300)
    plt.savefig(f"{base_path}_diff.png", dpi=300)
    print(f"Saved differential plots to {base_path}_diff.pdf and {base_path}_diff.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze Latent Spectrum across configurations")
    
    # Accetta fino a 4 configurazioni
    parser.add_argument("--conf1", type=str, help="Path for Config 1 (reference)")
    parser.add_argument("--label1", type=str, default="Config 1", help="Label for Config 1")
    
    parser.add_argument("--conf2", type=str, help="Path for Config 2")
    parser.add_argument("--label2", type=str, default="Config 2", help="Label for Config 2")
    
    parser.add_argument("--conf3", type=str, help="Path for Config 3")
    parser.add_argument("--label3", type=str, default="Config 3", help="Label for Config 3")
    
    parser.add_argument("--conf4", type=str, help="Path for Config 4")
    parser.add_argument("--label4", type=str, default="Config 4", help="Label for Config 4")
    
    parser.add_argument("--output", type=str, default="latent_spectrum_analysis",
                       help="Output file base name (without extension)")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit max files per configuration (optional)")
    parser.add_argument("--clusters", type=int, nargs=2, default=None, metavar=('START', 'END'),
                       help="Range di cluster da analizzare [START, END] inclusi")
    
    args = parser.parse_args()
    
    configs = [
        (args.conf1, args.label1),
        (args.conf2, args.label2),
        (args.conf3, args.label3),
        (args.conf4, args.label4)
    ]
    
    cluster_range = args.clusters
    
    results = []
    for path, label in configs:
        if path:
            res = load_latents_and_compute_psd(path, label, args.limit, cluster_range)
            if res:
                results.append(res)
    
    if not results:
        print("No valid data to plot!")
        return
    
    # Crea directory di output se necessario
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Genera i due grafici
    plot_spectral_analysis(results, args.output)
    plot_differential_analysis(results, args.output)
    
    # Stampa tabella riassuntiva
    print("\n" + "="*85)
    print(f"{'CONFIGURATION':<30} | {'N. SAMPLES':<12} | {'HFER (Mean)':<15}")
    print("-" * 85)
    
    for r in results:
        print(f"{r['label']:<30} | {r['count']:<12} | {r['hfer']:.4f}")
    print("="*85)
    
    # LaTeX table
    print("\n--- LaTeX Table Data ---")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("\\textbf{Configuration} & \\textbf{N. Samples} & \\textbf{HFER} \\\\")
    print("\\midrule")
    for r in results:
        print(f"{r['label']} & {r['count']} & {r['hfer']:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
