#!/usr/bin/env python3
"""
Script per generare mel spectrogram di tutti i file audio in una cartella.

Usage:
    python generate_mel_spectrograms.py <input_folder> <output_folder> [--sr SAMPLE_RATE] [--n_mels N_MELS] [--hop_length HOP_LENGTH]

Example:
    python generate_mel_spectrograms.py ./audio_files ./spectrograms
    python generate_mel_spectrograms.py ./audio_files ./spectrograms --sr 22050 --n_mels 128
"""

import argparse
import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.aiff', '.aif'}


def find_audio_files(input_path_str: str) -> list:
    """Trova tutti i file audio nella cartella specificata o ritorna il file singolo."""
    audio_files = []
    input_path = Path(input_path_str)
    
    # Se è un file singolo, verifica che sia audio e ritornalo
    if input_path.is_file():
        if input_path.suffix.lower() in AUDIO_EXTENSIONS:
            return [input_path]
        else:
            print(f"Errore: {input_path} non è un file audio valido.")
            return []
    
    # Se è una cartella, cerca tutti i file audio ricorsivamente
    if input_path.is_dir():
        for file_path in input_path.rglob('*'):
            if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                audio_files.append(file_path)
        return sorted(audio_files)
    
    return []


def generate_mel_spectrogram(
    audio_path: Path,
    output_path: Path,
    sr: int = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
    figsize: tuple = (12, 4),
    dpi: int = 300,
    reference_length: int = None
):
    """Genera e salva il mel spectrogram di un file audio."""
    try:
        # Carica l'audio (sr=None usa il sample rate nativo del file)
        y, sr_loaded = librosa.load(audio_path, sr=sr)
        
        # Se specificato, adatta la lunghezza dell'audio al riferimento
        if reference_length is not None:
            if len(y) > reference_length:
                # Tronca l'audio
                y = y[:reference_length]
                print(f"  Audio troncato da {len(y)} a {reference_length} campioni")
            elif len(y) < reference_length:
                # Pad con zeri
                y = np.pad(y, (0, reference_length - len(y)), mode='constant')
                print(f"  Audio paddato da {len(y)} a {reference_length} campioni")
        
        # Calcola il mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr_loaded,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )
        
        # Converti in decibel
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Crea la figura con impostazioni font migliorati
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        })
        
        fig, ax = plt.subplots(figsize=figsize)
        
        img = librosa.display.specshow(
            mel_spec_db,
            x_axis='time',
            y_axis='mel',
            sr=sr_loaded,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            ax=ax,
            cmap='magma'
        )
        
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(f'Mel Spectrogram: {audio_path.name}')
        
        # Salva la figura con alta qualità
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format=output_path.suffix[1:])
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"Errore nel processare {audio_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Genera mel spectrogram per tutti i file audio in una cartella.'
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help='Cartella contenente i file audio o percorso a un singolo file audio'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Cartella dove salvare i mel spectrogram'
    )
    parser.add_argument(
        '--sr', '--sample-rate',
        type=int,
        default=None,
        help='Sample rate per caricare l\'audio (default: None, usa il sample rate nativo del file)'
    )
    parser.add_argument(
        '--n_mels',
        type=int,
        default=128,
        help='Numero di bande mel (default: 128)'
    )
    parser.add_argument(
        '--n_fft',
        type=int,
        default=2048,
        help='Lunghezza FFT (default: 2048)'
    )
    parser.add_argument(
        '--hop_length',
        type=int,
        default=512,
        help='Hop length per STFT (default: 512)'
    )
    parser.add_argument(
        '--fmin',
        type=float,
        default=0.0,
        help='Frequenza minima in Hz (default: 0.0)'
    )
    parser.add_argument(
        '--fmax',
        type=float,
        default=None,
        help='Frequenza massima in Hz (default: sr/2)'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[12, 4],
        help='Dimensioni della figura (larghezza, altezza) (default: 12 4)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI per le immagini salvate (default: 300)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'jpg', 'pdf', 'svg'],
        help='Formato immagine output (default: png)'
    )
    parser.add_argument(
        '--reference',
        type=str,
        default=None,
        help='File audio di riferimento: gli audio processati saranno adattati alla sua lunghezza (troncati o paddati)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Cerca file audio anche nelle sottocartelle'
    )
    
    args = parser.parse_args()
    
    # Verifica che l'input esista
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"Errore: '{args.input_folder}' non esiste.")
        return
    
    # Crea la cartella output se non esiste
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Trova tutti i file audio
    is_single_file = input_path.is_file()
    if is_single_file:
        print(f"Processando file singolo: '{args.input_folder}'")
    else:
        print(f"Cercando file audio in '{args.input_folder}'...")
    
    audio_files = find_audio_files(args.input_folder)
    
    if not audio_files:
        print("Nessun file audio trovato.")
        return
    
    print(f"Trovati {len(audio_files)} file audio.")
    
    # Se specificato un file di riferimento, carica e ottieni la sua lunghezza
    reference_length = None
    if args.reference:
        reference_path = Path(args.reference)
        if not reference_path.exists():
            print(f"Errore: il file di riferimento '{args.reference}' non esiste.")
            return
        print(f"Caricando file di riferimento: {args.reference}")
        try:
            y_ref, sr_ref = librosa.load(reference_path, sr=args.sr)
            reference_length = len(y_ref)
            print(f"Lunghezza di riferimento: {reference_length} campioni ({reference_length/sr_ref:.2f} secondi @ {sr_ref} Hz)")
        except Exception as e:
            print(f"Errore nel caricare il file di riferimento: {e}")
            return
    
    print(f"Generando mel spectrogram...")
    
    # Processa ogni file
    success_count = 0
    error_count = 0
    
    for audio_file in tqdm(audio_files, desc="Processando"):
        # Costruisci il path di output
        if is_single_file:
            # Per file singolo, usa solo il nome del file nell'output
            output_file = output_path / audio_file.with_suffix(f'.{args.format}').name
        else:
            # Per cartella, mantieni la struttura relativa
            relative_path = audio_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix(f'.{args.format}')
        
        # Crea le sottocartelle se necessario
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Genera il mel spectrogram
        success = generate_mel_spectrogram(
            audio_path=audio_file,
            output_path=output_file,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            fmin=args.fmin,
            fmax=args.fmax,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            reference_length=reference_length
        )
        
        if success:
            success_count += 1
        else:
            error_count += 1
    
    print(f"\nCompletato!")
    print(f"  - Successi: {success_count}")
    print(f"  - Errori: {error_count}")
    print(f"  - Mel spectrogram salvati in: {output_path}")


if __name__ == '__main__':
    main()
