#!/usr/bin/env python3
"""
Script per rilevare file audio che sembrano rumore.

Questo script analizza una cartella di file audio e identifica quelli che
potrebbero essere rumore basandosi su diverse metriche:
1. Zero Crossing Rate (ZCR) - alto per il rumore
2. Spectral Flatness - alto per il rumore (vicino a 1)
3. RMS Energy variance - basso per rumore bianco stazionario
4. Crest Factor - basso per il rumore
5. Spectral Centroid stability - rumore ha centroide più stabile

Usage:
    python detect_noise_audio.py <cartella> [--threshold 0.7] [--move-to <cartella_rumore>]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import json
import shutil

import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')


def compute_audio_features(audio_path: str, sr: int = 22050) -> Dict[str, float]:
    """
    Calcola le features audio per determinare se è rumore.
    
    Returns:
        Dict con le features calcolate
    """
    try:
        # Carica l'audio
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        
        # Se l'audio è troppo corto, skip
        if len(y) < sr * 0.5:  # almeno 0.5 secondi
            return None
        
        # Normalizza
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        features = {}
        
        # 1. Zero Crossing Rate medio
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # 2. Spectral Flatness (Wiener entropy) - 1 = rumore bianco, 0 = tono puro
        spec_flat = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = float(np.mean(spec_flat))
        features['spectral_flatness_std'] = float(np.std(spec_flat))
        
        # 3. RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['rms_cv'] = float(np.std(rms) / (np.mean(rms) + 1e-8))  # coefficient of variation
        
        # 4. Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spec_cent))
        features['spectral_centroid_std'] = float(np.std(spec_cent))
        features['spectral_centroid_cv'] = float(np.std(spec_cent) / (np.mean(spec_cent) + 1e-8))
        
        # 5. Spectral Bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
        
        # 6. Spectral Rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spec_rolloff))
        
        # 7. Crest Factor (Peak to RMS ratio) - basso per rumore
        peak = np.max(np.abs(y))
        rms_total = np.sqrt(np.mean(y**2))
        features['crest_factor'] = float(peak / (rms_total + 1e-8))
        
        # 8. Autocorrelation - rumore ha bassa autocorrelazione
        autocorr = np.correlate(y[:sr], y[:sr], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-8)
        # Media dei primi 100 lag (escludendo il primo)
        features['autocorr_mean'] = float(np.mean(np.abs(autocorr[1:min(100, len(autocorr))])))
        
        # 9. Spectral Entropy
        S = np.abs(librosa.stft(y))
        S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-8)
        spectral_entropy = -np.sum(S_norm * np.log2(S_norm + 1e-8), axis=0)
        features['spectral_entropy_mean'] = float(np.mean(spectral_entropy))
        
        # 10. Controlla se ci sono pattern ritmici (rumore non ne ha)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        features['detected_tempo'] = float(tempo) if isinstance(tempo, (int, float, np.number)) else float(tempo[0]) if len(tempo) > 0 else 0.0
        features['num_beats'] = int(len(beats))
        
        return features
        
    except Exception as e:
        print(f"Errore processando {audio_path}: {e}")
        return None


def compute_noise_score(features: Dict[str, float]) -> float:
    """
    Calcola uno score da 0 a 1 che indica quanto l'audio sembra rumore.
    1 = sicuramente rumore, 0 = sicuramente non rumore
    """
    score = 0.0
    weights_sum = 0.0
    
    # Spectral Flatness alta = rumore (peso alto, molto indicativo)
    if features['spectral_flatness_mean'] > 0.1:
        score += 3.0 * min(features['spectral_flatness_mean'] / 0.5, 1.0)
        weights_sum += 3.0
    else:
        weights_sum += 3.0
    
    # ZCR alto = rumore
    if features['zcr_mean'] > 0.1:
        score += 2.0 * min(features['zcr_mean'] / 0.3, 1.0)
        weights_sum += 2.0
    else:
        weights_sum += 2.0
    
    # Bassa autocorrelazione = rumore
    autocorr_score = 1.0 - min(features['autocorr_mean'] / 0.3, 1.0)
    score += 2.0 * autocorr_score
    weights_sum += 2.0
    
    # Crest factor basso = rumore (rumore bianco ha CF ~3-4, musica ha CF più alto)
    if features['crest_factor'] < 5:
        score += 1.5 * (1.0 - features['crest_factor'] / 5.0)
        weights_sum += 1.5
    else:
        weights_sum += 1.5
    
    # Pochi beat rilevati = possibile rumore
    if features['num_beats'] < 5:
        score += 1.0
    weights_sum += 1.0
    
    # Spectral centroid CV basso = rumore stazionario
    if features['spectral_centroid_cv'] < 0.3:
        score += 1.0 * (1.0 - features['spectral_centroid_cv'] / 0.3)
        weights_sum += 1.0
    else:
        weights_sum += 1.0
    
    # Alta entropia spettrale = rumore
    if features['spectral_entropy_mean'] > 5:
        score += 1.0 * min((features['spectral_entropy_mean'] - 5) / 3, 1.0)
        weights_sum += 1.0
    else:
        weights_sum += 1.0
    
    return score / weights_sum


def analyze_folder(
    folder_path: str,
    threshold: float = 0.6,
    extensions: List[str] = ['.wav', '.mp3', '.flac', '.ogg', '.m4a'],
    recursive: bool = True
) -> List[Tuple[str, float, Dict]]:
    """
    Analizza tutti i file audio in una cartella.
    
    Returns:
        Lista di tuple (path, noise_score, features) per i file considerati rumore
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Errore: la cartella {folder_path} non esiste")
        return []
    
    # Trova tutti i file audio
    audio_files = []
    if recursive:
        for ext in extensions:
            audio_files.extend(folder.rglob(f"*{ext}"))
            audio_files.extend(folder.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            audio_files.extend(folder.glob(f"*{ext}"))
            audio_files.extend(folder.glob(f"*{ext.upper()}"))
    
    audio_files = sorted(set(audio_files))
    
    if not audio_files:
        print(f"Nessun file audio trovato in {folder_path}")
        return []
    
    print(f"Trovati {len(audio_files)} file audio da analizzare...")
    
    noise_files = []
    all_results = []
    
    for i, audio_file in enumerate(audio_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Analizzando {i + 1}/{len(audio_files)}: {audio_file.name}")
        
        features = compute_audio_features(str(audio_file))
        
        if features is None:
            continue
        
        noise_score = compute_noise_score(features)
        
        result = {
            'path': str(audio_file),
            'noise_score': noise_score,
            'features': features
        }
        all_results.append(result)
        
        if noise_score >= threshold:
            noise_files.append((str(audio_file), noise_score, features))
    
    return noise_files, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Rileva file audio che sembrano rumore in una cartella"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Cartella contenente i file audio da analizzare"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.6,
        help="Soglia noise score (0-1) sopra la quale un file è considerato rumore (default: 0.6)"
    )
    parser.add_argument(
        "--move-to", "-m",
        type=str,
        default=None,
        help="Cartella dove spostare i file rumore (opzionale)"
    )
    parser.add_argument(
        "--copy-to", "-c",
        type=str,
        default=None,
        help="Cartella dove copiare i file rumore (opzionale)"
    )
    parser.add_argument(
        "--output-json", "-o",
        type=str,
        default=None,
        help="File JSON dove salvare i risultati completi dell'analisi"
    )
    parser.add_argument(
        "--recursive", "-r",
        type=lambda x: x.lower() in ('true', '1', 'yes', 'si', 's'),
        default=True,
        metavar="BOOL",
        help="Se True cerca anche nelle sottocartelle, se False solo nella cartella specificata (default: True)"
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="Mostra tutti i file con i loro score, non solo quelli rumore"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Analisi rumore audio")
    print(f"{'='*60}")
    print(f"Cartella: {args.folder}")
    print(f"Soglia: {args.threshold}")
    print(f"Ricerca ricorsiva: {args.recursive}")
    print(f"{'='*60}\n")
    
    noise_files, all_results = analyze_folder(
        args.folder,
        threshold=args.threshold,
        recursive=args.recursive
    )
    
    # Ordina tutti i risultati per noise score
    all_results = sorted(all_results, key=lambda x: x['noise_score'], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"RISULTATI")
    print(f"{'='*60}")
    print(f"File analizzati: {len(all_results)}")
    print(f"File classificati come rumore (score >= {args.threshold}): {len(noise_files)}")
    
    if noise_files:
        print(f"\n{'='*60}")
        print("FILE RUMORE RILEVATI:")
        print(f"{'='*60}")
        
        noise_files = sorted(noise_files, key=lambda x: x[1], reverse=True)
        
        for path, score, features in noise_files:
            print(f"\n📢 {Path(path).name}")
            print(f"   Path: {path}")
            print(f"   Noise Score: {score:.3f}")
            print(f"   Spectral Flatness: {features['spectral_flatness_mean']:.3f}")
            print(f"   ZCR: {features['zcr_mean']:.3f}")
            print(f"   Autocorr: {features['autocorr_mean']:.3f}")
            print(f"   Beats rilevati: {features['num_beats']}")
    
    if args.list_all:
        print(f"\n{'='*60}")
        print("TUTTI I FILE (ordinati per noise score):")
        print(f"{'='*60}")
        for result in all_results:
            status = "🔴 RUMORE" if result['noise_score'] >= args.threshold else "🟢 OK"
            print(f"{status} [{result['noise_score']:.3f}] {Path(result['path']).name}")
    
    # Salva JSON se richiesto
    if args.output_json:
        avg_noise_score = np.mean([r['noise_score'] for r in all_results]) if all_results else 0.0
        with open(args.output_json, 'w') as f:
            json.dump({
                'folder': args.folder,
                'threshold': args.threshold,
                'total_files': len(all_results),
                'noise_files_count': len(noise_files),
                'average_noise_score': float(avg_noise_score),
                'results': all_results
            }, f, indent=2)
        print(f"\nRisultati salvati in: {args.output_json}")
    
    # Sposta file se richiesto
    if args.move_to and noise_files:
        move_dir = Path(args.move_to)
        move_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSpostamento file rumore in: {args.move_to}")
        
        for path, score, _ in noise_files:
            src = Path(path)
            dst = move_dir / src.name
            # Se esiste già, aggiungi numero
            counter = 1
            while dst.exists():
                dst = move_dir / f"{src.stem}_{counter}{src.suffix}"
                counter += 1
            shutil.move(str(src), str(dst))
            print(f"  Spostato: {src.name} -> {dst}")
    
    # Copia file se richiesto
    if args.copy_to and noise_files:
        copy_dir = Path(args.copy_to)
        copy_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCopia file rumore in: {args.copy_to}")
        
        for path, score, _ in noise_files:
            src = Path(path)
            dst = copy_dir / src.name
            counter = 1
            while dst.exists():
                dst = copy_dir / f"{src.stem}_{counter}{src.suffix}"
                counter += 1
            shutil.copy2(str(src), str(dst))
            print(f"  Copiato: {src.name} -> {dst}")
    
    print(f"\n{'='*60}")
    print("Analisi completata!")
    print(f"{'='*60}\n")
    
    return len(noise_files)


if __name__ == "__main__":
    sys.exit(main())
