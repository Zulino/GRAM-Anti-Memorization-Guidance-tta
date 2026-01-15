import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm

def compute_psd(tensor):
    # Tensor shape: [Channels, Time] o [1, Channels, Time]
    if tensor.dim() == 3: tensor = tensor.squeeze(0)
    
    # FFT Real-to-Complex sull'asse temporale
    fft = torch.fft.rfft(tensor, dim=-1, norm='ortho')
    
    # Potenza (Magnitude squared)
    power = fft.abs() ** 2
    
    # Media sui 64 canali (vogliamo l'energia globale del latent)
    avg_power = power.mean(dim=0) 
    return avg_power.numpy()

def aggregate_latent_analysis(latents_dir, output_dir='analysis_aggregated'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Liste per accumulare le differenze
    deltas_raw = []   # (Raw - Baseline)
    deltas_filt = []  # (Filtered - Baseline)
    
    # Trova tutti i file baseline per usarli come "ancora"
    # Assumiamo pattern: "nome_baseline.pt"
    baseline_files = glob.glob(os.path.join(latents_dir, "*_no_amg.pt")) 
    
    print(f"Found {len(baseline_files)} baseline samples. Computing aggregates...")

    latent_sr = 0
    freqs = None

    for base_path in tqdm(baseline_files):
        # Ricostruisci i nomi dei file corrispondenti (Raw e Filtered)
        # Esempio: "song1_no_amg.pt" -> "song1_gram_no_filter.pt"
        # Adatta questa parte alla tua nomenclatura esatta!
        prefix = base_path.replace("_no_amg.pt", "")
        raw_path = f"{prefix}_gram_no_filter.pt"
        filt_path = f"{prefix}_gram_filtered.pt" # o _gram_0.25_filtered.pt

        if not os.path.exists(raw_path) or not os.path.exists(filt_path):
            continue

        # Carica tensori
        t_base = torch.load(base_path, map_location='cpu').float()
        t_raw = torch.load(raw_path, map_location='cpu').float()
        t_filt = torch.load(filt_path, map_location='cpu').float()
        
        # Calcola PSD
        psd_base = compute_psd(t_base)
        psd_raw = compute_psd(t_raw)
        psd_filt = compute_psd(t_filt)
        
        # Inizializza asse frequenze (una volta sola)
        if freqs is None:
            n_samples = t_base.shape[-1]
            # Sostituisci con il tuo SR corretto calcolato (48000 / 2048 = 23.43)
            latent_sr = 23.4375 
            freqs = np.fft.rfftfreq(n_samples, d=1/latent_sr)

        # Calcola Differenze in dB
        # Aggiungiamo epsilon per evitare log(0)
        eps = 1e-12
        db_base = 10 * np.log10(psd_base + eps)
        db_raw = 10 * np.log10(psd_raw + eps)
        db_filt = 10 * np.log10(psd_filt + eps)
        
        deltas_raw.append(db_raw - db_base)
        deltas_filt.append(db_filt - db_base)

    # --- AGGREGAZIONE ---
    # Stack in matrici [N_samples, N_freqs] e fai la media
    avg_delta_raw = np.mean(np.stack(deltas_raw), axis=0)
    std_delta_raw = np.std(np.stack(deltas_raw), axis=0) # Per le confidence bands
    
    avg_delta_filt = np.mean(np.stack(deltas_filt), axis=0)
    std_delta_filt = np.std(np.stack(deltas_filt), axis=0)
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    
    # Disegna intervalli di confidenza (Mean +/- StdErr o StdDev)
    # Usiamo StdDev/sqrt(N) per lo Standard Error della media se N è grande
    sem_raw = std_delta_raw / np.sqrt(len(deltas_raw))
    sem_filt = std_delta_filt / np.sqrt(len(deltas_filt))

    # Plot Raw AMG
    plt.plot(freqs, avg_delta_raw, label='Raw AMG vs Baseline', color='blue', linewidth=2)
    plt.fill_between(freqs, avg_delta_raw - sem_raw, avg_delta_raw + sem_raw, color='blue', alpha=0.1)
    
    # Plot Filtered AMG
    plt.plot(freqs, avg_delta_filt, label='Filtered AMG vs Baseline', color='red', linewidth=2)
    plt.fill_between(freqs, avg_delta_filt - sem_filt, avg_delta_filt + sem_filt, color='red', alpha=0.1)
    
    # Linea Zero (Baseline)
    plt.axhline(0, color='black', linestyle='--', label='Baseline Reference')
    
    # Linea Cutoff (Visualizzazione)
    # Assumendo cutoff 0.25 * Nyquist (11.7) ~ 2.9 Hz
    plt.axvline(2.9, color='orange', linestyle=':', label='Filter Cutoff')

    plt.title(f'Aggregated Latent Spectral Impact (Average over {len(deltas_raw)} generations)')
    plt.xlabel('Latent Modulation Frequency (Hz)')
    plt.ylabel('Power Difference vs Baseline (dB)')
    plt.xlim(0, 11.7) # Nyquist limit
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'aggregated_latent_impact.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved aggregated plot to {save_path}")
    plt.show()


def compare_two_folders(folder_a, folder_b, label_a='Method A', label_b='Method B', output_dir='analysis_comparison'):
    """
    Confronta i latent tra due cartelle con gli stessi nomi file.
    
    Args:
        folder_a: Prima cartella (es. spectralAnalysis/gram_lp/43)
        folder_b: Seconda cartella (es. spectralAnalysis/gram_no_filter/43)
        label_a: Etichetta per folder_a nel plot
        label_b: Etichetta per folder_b nel plot
        output_dir: Cartella di output per i plot
        
    Struttura attesa:
        folder_a/latents/gen_1.pt, gen_2.pt, ...
        folder_b/latents/gen_1.pt, gen_2.pt, ...
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Percorsi alle sottocartelle latents
    latents_dir_a = os.path.join(folder_a, 'latents')
    latents_dir_b = os.path.join(folder_b, 'latents')
    
    # Verifica esistenza
    if not os.path.exists(latents_dir_a):
        # Prova senza sottocartella latents
        latents_dir_a = folder_a
    if not os.path.exists(latents_dir_b):
        latents_dir_b = folder_b
    
    # Trova tutti i .pt nella prima cartella
    files_a = sorted(glob.glob(os.path.join(latents_dir_a, "*.pt")))
    
    print(f"Found {len(files_a)} files in {latents_dir_a}")
    
    psd_list_a = []
    psd_list_b = []
    freqs = None
    matched_count = 0
    
    for file_a in tqdm(files_a):
        filename = os.path.basename(file_a)
        file_b = os.path.join(latents_dir_b, filename)
        
        if not os.path.exists(file_b):
            print(f"Warning: {filename} not found in {latents_dir_b}, skipping...")
            continue
        
        matched_count += 1
        
        # Carica tensori
        t_a = torch.load(file_a, map_location='cpu', weights_only=False).float()
        t_b = torch.load(file_b, map_location='cpu', weights_only=False).float()
        
        # Calcola PSD
        psd_a = compute_psd(t_a)
        psd_b = compute_psd(t_b)
        
        # Inizializza frequenze
        if freqs is None:
            n_samples = t_a.shape[-1]
            latent_sr = 23.4375  # 48000 / 2048
            freqs = np.fft.rfftfreq(n_samples, d=1/latent_sr)
        
        # Converti in dB
        eps = 1e-12
        psd_list_a.append(10 * np.log10(psd_a + eps))
        psd_list_b.append(10 * np.log10(psd_b + eps))
    
    print(f"Successfully matched {matched_count} file pairs")
    
    if matched_count == 0:
        print("No matching files found!")
        return
    
    # Aggregazione
    avg_psd_a = np.mean(np.stack(psd_list_a), axis=0)
    avg_psd_b = np.mean(np.stack(psd_list_b), axis=0)
    std_a = np.std(np.stack(psd_list_a), axis=0)
    std_b = np.std(np.stack(psd_list_b), axis=0)
    sem_a = std_a / np.sqrt(len(psd_list_a))
    sem_b = std_b / np.sqrt(len(psd_list_b))
    
    # --- PLOT 1: PSD Assolute ---
    plt.figure(figsize=(12, 6))
    
    plt.plot(freqs, avg_psd_a, label=label_a, color='blue', linewidth=2)
    plt.fill_between(freqs, avg_psd_a - sem_a, avg_psd_a + sem_a, color='blue', alpha=0.15)
    
    plt.plot(freqs, avg_psd_b, label=label_b, color='red', linewidth=2)
    plt.fill_between(freqs, avg_psd_b - sem_b, avg_psd_b + sem_b, color='red', alpha=0.15)
    
    plt.title(f'Latent PSD Comparison ({matched_count} samples)')
    plt.xlabel('Latent Modulation Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.xlim(0, freqs[-1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'psd_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved to {save_path}")
    plt.show()
    
    # --- PLOT 2: Differenza ---
    plt.figure(figsize=(12, 6))
    diff = avg_psd_a - avg_psd_b
    plt.plot(freqs, diff, color='green', linewidth=2, label=f'{label_a} - {label_b}')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.fill_between(freqs, diff - np.sqrt(sem_a**2 + sem_b**2), 
                     diff + np.sqrt(sem_a**2 + sem_b**2), color='green', alpha=0.15)
    
    plt.title(f'Spectral Difference: {label_a} vs {label_b}')
    plt.xlabel('Latent Modulation Frequency (Hz)')
    plt.ylabel('Power Difference (dB)')
    plt.xlim(0, freqs[-1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path2 = os.path.join(output_dir, 'psd_difference.png')
    plt.savefig(save_path2, dpi=300)
    print(f"Saved to {save_path2}")
    plt.show()
    
    return {
        'freqs': freqs,
        'avg_psd_a': avg_psd_a,
        'avg_psd_b': avg_psd_b,
        'difference': diff,
        'n_samples': matched_count
    }


# Esempio di utilizzo:
# aggregate_latent_analysis("path/to/all_latents_folder")

# Per confrontare due cartelle:
# compare_two_folders(
#     'spectralAnalysis/gram_lp/43',
#     'spectralAnalysis/gram_no_filter/43',
#     label_a='GRAM LP Filtered',
#     label_b='GRAM No Filter',
#     output_dir='spectral_comparison'
# )

#python -c "from latents_diff_analyzer import compare_two_folders; compare_two_folders('spectralAnalysis/gram_lp/43', 'spectralAnalysis/gram_no_filter/43', label_a='GRAM LP Filtered', label_b='GRAM No Filter', output_dir='spectral_comparison')"