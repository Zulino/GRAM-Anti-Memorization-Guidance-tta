import torch
import os
import json
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Import per CLAP
from stable_audio_tools.inference import amg_generation

# --- CONFIGURAZIONE ---
FOLDERS_TO_EVALUATE = [
    "./random_gen/no_rescale",
    "./random_gen/rescale"
]

INPUT_JSON = 'embeddings_new.json'
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def get_text_embedding(clap_model, prompt, device):
    """
    Calcola l'embedding del testo usando il modello CLAP.
    Usa encode_text direttamente per evitare bug di compatibilità.
    """
    # Tokenizza il prompt
    text_data = clap_model.tokenizer([prompt])
    
    # Converti in tensori con batch dimension
    input_ids = text_data['input_ids'].clone().detach().unsqueeze(0).to(device)
    attention_mask = text_data['attention_mask'].clone().detach().unsqueeze(0).to(device)
    
    # Usa encode_text del modello interno
    with torch.no_grad():
        text_embed = clap_model.model.encode_text(
            {'input_ids': input_ids, 'attention_mask': attention_mask}, 
            device=device
        )
    
    return text_embed


def compute_prompt_adherence(folder_path, clap_model, json_data):
    """
    Calcola Prompt Adherence usando i prompt dal JSON.
    Restituisce lo score medio e la lista di file saltati.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    scores = []
    skipped_files = []
    
    print(f"--> Calcolo Prompt Adherence per {folder_path}...")

    for f in tqdm(files):
        try:
            sound_id = f.replace("sound_", "").replace(".wav", "")
            
            # Controlla se l'ID esiste nel JSON
            if sound_id not in json_data:
                skipped_files.append((f, "ID non trovato nel JSON"))
                continue
            
            # Controlla se il prompt esiste
            if 'conditioning' not in json_data[sound_id] or 'prompt' not in json_data[sound_id]['conditioning']:
                skipped_files.append((f, "Prompt mancante nel JSON"))
                continue
            
            prompt = json_data[sound_id]['conditioning']['prompt']
            
            # Calcola embedding del testo dal prompt
            text_embed = get_text_embedding(clap_model, prompt, DEVICE)
            
            audio_path = os.path.join(folder_path, f)
            
            # Ottieni embedding audio da file
            with torch.no_grad():
                audio_embed = clap_model.get_audio_embedding_from_filelist(x=[audio_path])
            
            if isinstance(audio_embed, np.ndarray): 
                audio_embed = torch.from_numpy(audio_embed).to(DEVICE)
            elif isinstance(audio_embed, torch.Tensor): 
                audio_embed = audio_embed.to(DEVICE)
            if audio_embed.ndim == 1: 
                audio_embed = audio_embed.unsqueeze(0)

            # Normalizza e calcola similarity
            audio_embed = F.normalize(audio_embed, p=2, dim=1)
            text_embed = F.normalize(text_embed, p=2, dim=1)
            scores.append(torch.sum(audio_embed * text_embed).item())
                
        except Exception as e:
            skipped_files.append((f, str(e)))
            continue
            
    return np.mean(scores) if len(scores) > 0 else 0.0, skipped_files


# --- MAIN ---

def main():
    print("Caricamento JSON prompt...")
    with open(INPUT_JSON, 'r') as f:
        json_data = json.load(f)
        
    print("Inizializzazione CLAP...")
    CLAP = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, device=DEVICE)
    CLAP.load_ckpt()
    CLAP.eval()

    print("\n" + "="*60)
    print(f"{'CONFIGURATION':<30} | {'ADHERENCE (Higher=Better)':<25}")
    print("="*60)

    all_skipped = {}

    for folder in FOLDERS_TO_EVALUATE:
        if not os.path.exists(folder):
            print(f"Cartella non trovata: {folder}")
            continue
            
        label = os.path.basename(folder).upper().replace("_", " ")
        
        # Calcolo Adherence
        adherence_score, skipped = compute_prompt_adherence(folder, CLAP, json_data)
        
        if skipped:
            all_skipped[folder] = skipped
        
        print(f"{label:<30} | {adherence_score:<25.4f}")

    print("="*60)
    
    # Stampa file saltati
    if all_skipped:
        print("\n" + "="*60)
        print("FILE SALTATI:")
        print("="*60)
        for folder, skipped_list in all_skipped.items():
            print(f"\n📁 {folder} ({len(skipped_list)} file saltati):")
            for filename, reason in skipped_list:
                print(f"   ⚠️ {filename}: {reason}")
        print("="*60)


if __name__ == "__main__":
    main()
