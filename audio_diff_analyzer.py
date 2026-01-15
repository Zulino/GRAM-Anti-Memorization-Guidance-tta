import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

def analyze_audio_difference(path_a, path_b, output_dir='analysis_results'):
    """
    Confronta due file audio e visualizza le differenze spettrali.
    Path A: Solitamente il file 'Base' (es. con CFG standard / Memorizzato)
    Path B: Solitamente il file 'Trattato' (es. con AMG Filtrata)
    """
    # Creazione cartella output
    os.makedirs(output_dir, exist_ok=True)
    name_a = os.path.basename(path_a)
    name_b = os.path.basename(path_b)
    
    print(f"--- Loading Files ---\nA: {name_a}\nB: {name_b}")
    
    # 1. Caricamento Audio
    # Usa sr=None per mantenere il sample rate nativo (es. 44.1k o 48k)
    y_a, sr_a = librosa.load(path_a, sr=None)
    y_b, sr_b = librosa.load(path_b, sr=None)
    
    if sr_a != sr_b:
        raise ValueError(f"Sample rates mismatch: {sr_a} vs {sr_b}")
    
    # 2. Allineamento Lunghezza
    min_len = min(len(y_a), len(y_b))
    y_a = y_a[:min_len]
    y_b = y_b[:min_len]
    
    # 3. Calcolo STFT (Spettrogramma)
    n_fft = 2048
    hop_length = 512
    
    # Calcoliamo la Magnitudo Lineare
    S_a = np.abs(librosa.stft(y_a, n_fft=n_fft, hop_length=hop_length))
    S_b = np.abs(librosa.stft(y_b, n_fft=n_fft, hop_length=hop_length))
    
    # Convertiamo in dB
    S_a_db = librosa.amplitude_to_db(S_a, ref=np.max)
    S_b_db = librosa.amplitude_to_db(S_b, ref=np.max)
    
    # 4. Calcolo Differenza (A - B)
    # Valori POSITIVI = A ha più energia (es. rumore rimosso in B)
    # Valori NEGATIVI = B ha più energia (es. struttura aggiunta in B)
    diff_db = S_a_db - S_b_db
    
    # --- CALCOLO METRICHE ---
    
    # A. Differenza Media per Frequenza (Profilo Spettrale)
    # Media lungo l'asse temporale (axis=1)
    mean_diff_per_freq = np.mean(diff_db, axis=1)
    freqs = librosa.fft_frequencies(sr=sr_a, n_fft=n_fft)
    
    # B. Centroide Spettrale della DIFFERENZA (Dove si concentra il cambiamento?)
    # Usiamo la magnitudo assoluta della differenza per capire dove c'è "azione"
    abs_diff_mag = np.abs(S_a - S_b) 
    spec_cent = librosa.feature.spectral_centroid(S=abs_diff_mag, sr=sr_a, n_fft=n_fft)[0]
    avg_centroid_diff = np.mean(spec_cent)
    
    # C. Energy Ratios (Bande di Frequenza)
    # Definiamo bande: Low (<2kHz), Mid (2-8kHz), High (>8kHz)
    band_low = (freqs < 2000)
    band_mid = (freqs >= 2000) & (freqs < 8000)
    band_high = (freqs >= 8000)
    
    # Energia totale della differenza (somma magnitudo diff)
    total_diff_energy = np.sum(abs_diff_mag)
    
    e_low = np.sum(abs_diff_mag[band_low, :]) / total_diff_energy * 100
    e_mid = np.sum(abs_diff_mag[band_mid, :]) / total_diff_energy * 100
    e_high = np.sum(abs_diff_mag[band_high, :]) / total_diff_energy * 100

    print(f"\n--- Analysis Results ---")
    print(f"Spectral Centroid of Change: {avg_centroid_diff:.2f} Hz")
    print(f"Energy of Change Distribution:")
    print(f"  - Low freq (<2kHz):  {e_low:.2f}%")
    print(f"  - Mid freq (2-8kHz): {e_mid:.2f}%")
    print(f"  - High freq (>8kHz): {e_high:.2f}%")
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    
    # Plot A (Original)
    ax1 = fig.add_subplot(gs[0, 0])
    librosa.display.specshow(S_a_db, sr=sr_a, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax1, cmap='magma')
    ax1.set_title(f'A: {name_a}')
    
    # Plot B (Modified)
    ax2 = fig.add_subplot(gs[0, 1])
    librosa.display.specshow(S_b_db, sr=sr_b, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
    ax2.set_title(f'B: {name_b}')
    
    # Plot Difference Spectrogram
    ax3 = fig.add_subplot(gs[1, :])
    # Usa mappa divergente (bwr: Blue-White-Red) centrata a 0
    # Rosso = A > B (Energia Rimossa), Blu = B > A (Energia Aggiunta)
    divnorm = plt.Normalize(vmin=-20, vmax=20) 
    img_diff = librosa.display.specshow(diff_db, sr=sr_a, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax3, cmap='bwr', vmin=-20, vmax=20)
    ax3.set_title('Difference Spectrogram (A - B) [Red = Removed from A, Blue = Added to B]')
    plt.colorbar(img_diff, ax=ax3, label='dB Diff')
    
    # Plot Profilo Medio delle Frequenze
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(freqs, mean_diff_per_freq, color='black', linewidth=1.5)
    
    # Colora le aree sotto la curva
    ax4.fill_between(freqs, mean_diff_per_freq, 0, where=(mean_diff_per_freq > 0), interpolate=True, color='red', alpha=0.3, label='Energy Removed (A > B)')
    ax4.fill_between(freqs, mean_diff_per_freq, 0, where=(mean_diff_per_freq < 0), interpolate=True, color='blue', alpha=0.3, label='Energy Added (B > A)')
    
    ax4.axvline(avg_centroid_diff, color='green', linestyle='--', label=f'Centroid of Change: {avg_centroid_diff:.0f}Hz')
    ax4.set_xscale('linear') # O 'log' se preferisci
    ax4.set_xlim([0, sr_a/2])
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Avg Difference (dB)')
    ax4.set_title('Average Spectral Difference Profile')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'diff_analysis_{name_a}_vs_{name_b}.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()

analyze_audio_difference('/mnt/media/HDD_4TB/riccardo/GRAM-AMG/soundDataset/sound_5131.wav', 'spectral_analysis/no_amg.wav')