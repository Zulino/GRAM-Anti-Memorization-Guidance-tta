import sys
import os

# --- FIX IMPORTAZIONE ---
# Ottieni il percorso assoluto della cartella in cui si trova questo script
current_script_path = os.path.dirname(os.path.abspath(__file__))
# Aggiungi questa cartella al path di sistema di Python
sys.path.append(current_script_path)

import torch
import torchaudio
import numpy as np
from transformers import ClapProcessor, ClapModel
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

# --- IMPORTAZIONE DELLA TUA LIBRERIA ---
# Assicurati che mert_method.py sia nella stessa directory
try:
    from mert_method import get_mert_file_embedding
except ImportError as e:
    print(f"\nERRORE CRITICO: Non riesco a trovare 'mert_method.py'.")
    print(f"Python sta cercando in: {sys.path}")
    print(f"Assicurati che 'mert_method.py' sia esattamente in: {current_script_path}\n")
    raise e
# --- CONFIGURAZIONE ---
PATH_DATASET = "soundDataset" 
PATH_GENERATED = "spectralAnalysis_fad/no_amg/random_prompts"

# ID Modello CLAP (MERT viene gestito da mert_method.py)
CLAP_MODEL_ID = "laion/clap-htsat-unfused"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AudioEvaluator:
    def __init__(self):
        print("Caricamento modello CLAP...")
        # CLAP setup
        self.clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        self.clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device)
        self.clap_sr = 48000 

    def load_audio_for_clap(self, path):
        """Carica, converte in mono e ricampiona l'audio specificamente per CLAP."""
        try:
            waveform, sr = torchaudio.load(path)
            # Converti in mono se stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Resampling per CLAP (48kHz)
            if sr != self.clap_sr:
                resampler = torchaudio.transforms.Resample(sr, self.clap_sr)
                waveform = resampler(waveform)
            return waveform.squeeze()
        except Exception as e:
            print(f"Errore caricamento CLAP {path}: {e}")
            return None

    def get_clap_embedding(self, audio_tensor):
        """Estrae embedding CLAP (solo audio encoder)."""
        if audio_tensor is None: return None
        # CLAP input processing
        inputs = self.clap_processor(audios=audio_tensor.cpu().numpy(), sampling_rate=self.clap_sr, return_tensors="pt").to(device)
        with torch.no_grad():
            audio_embed = self.clap_model.get_audio_features(**inputs)
        return audio_embed

# def find_file_recursive(root_dir, filename):
#     """Cerca un file ricorsivamente in una directory."""
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         if filename in filenames:
#             return os.path.join(dirpath, filename)
#     return None

def find_file_in_dir(root_dir, filename):
    """Cerca un file solo nella directory specificata (non ricorsivo)."""
    path = os.path.join(root_dir, filename)
    if os.path.isfile(path):
        return path
    return None

def main():
    # evaluator = AudioEvaluator()
    
    # mert_scores = []
    # clap_scores = []
    
    # dataset_files = [f for f in os.listdir(PATH_DATASET) if f.endswith('.wav')]
    # print(f"Trovati {len(dataset_files)} file nel dataset originale.")
    # print("Inizio calcolo similarità...\n")

    # for filename in tqdm(dataset_files):
    #     path_orig = os.path.join(PATH_DATASET, filename)
    #     path_gen = find_file_recursive(PATH_GENERATED, filename)
        
    #     if not path_gen:
    #         continue

    evaluator = AudioEvaluator()
    
    mert_scores = []
    clap_scores = []
    
    dataset_files = [f for f in os.listdir(PATH_DATASET) if f.endswith('.wav')]
    print(f"Trovati {len(dataset_files)} file nel dataset originale.")
    print("Inizio calcolo similarità...\n")

    for filename in tqdm(dataset_files):
        path_orig = os.path.join(PATH_DATASET, filename)
        path_gen = find_file_in_dir(PATH_GENERATED, filename)
        
        if not path_gen:
            continue
            
        # ---------------------------------------------------------
        # 1. CALCOLO MERT (Usando la tua libreria)
        # ---------------------------------------------------------
        # mert_method restituisce numpy array, lo convertiamo in tensor per cosine_similarity
        # La funzione get_mert_file_embedding fa già la media temporale internamente.
        try:
            emb_orig_np = get_mert_file_embedding(path_orig, feature='both', device=device)
            emb_gen_np = get_mert_file_embedding(path_gen, feature='both', device=device)
            
            # Convertiamo in tensori PyTorch e aggiungiamo dimensione batch (unsqueeze)
            t_orig = torch.from_numpy(emb_orig_np).unsqueeze(0).to(device)
            t_gen = torch.from_numpy(emb_gen_np).unsqueeze(0).to(device)
            
            score_mert = cosine_similarity(t_orig, t_gen).item()
            mert_scores.append(score_mert)
        except Exception as e:
            print(f"Errore MERT su {filename}: {e}")

        # ---------------------------------------------------------
        # 2. CALCOLO CLAP (Gestito internamente qui)
        # ---------------------------------------------------------
        try:
            audio_orig_clap = evaluator.load_audio_for_clap(path_orig)
            audio_gen_clap = evaluator.load_audio_for_clap(path_gen)
            
            if audio_orig_clap is not None and audio_gen_clap is not None:
                emb_orig_clap = evaluator.get_clap_embedding(audio_orig_clap)
                emb_gen_clap = evaluator.get_clap_embedding(audio_gen_clap)
                score_clap = cosine_similarity(emb_orig_clap, emb_gen_clap).item()
                clap_scores.append(score_clap)
        except Exception as e:
            print(f"Errore CLAP su {filename}: {e}")

    # --- RISULTATI ---
    print("\n" + "="*30)
    print("RISULTATI FINALI DI MEMORIZZAZIONE")
    print("="*30)
    
    if mert_scores:
        print(f"Media MERT Cosine Similarity: {np.mean(mert_scores):.4f} (std: {np.std(mert_scores):.4f})")
    else:
        print("Nessun punteggio MERT calcolato.")
        
    if clap_scores:
        print(f"Media CLAP Cosine Similarity: {np.mean(clap_scores):.4f} (std: {np.std(clap_scores):.4f})")
    else:
        print("Nessun punteggio CLAP calcolato.")

if __name__ == "__main__":
    main()