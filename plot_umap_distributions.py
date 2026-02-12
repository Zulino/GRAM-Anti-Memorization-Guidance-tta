import argparse
import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from tqdm import tqdm

# Importiamo CLAP come nei tuoi script precedenti
# Assicurati che stable_audio_tools sia installato e accessibile
try:
    from stable_audio_tools.inference import amg_generation
except ImportError:
    print("ERRORE: stable_audio_tools non trovato. Assicurati di essere nell'environment corretto.")
    exit(1)

# --- CONFIGURAZIONE VISUALIZZAZIONE ---
# Colori per le configurazioni (blind-friendly)
COLORS = {
    'real': '#333333',      # Nero per il background reale
    'no_amg': '#48e5c2',    # Ciano
    'baseline': '#d62828',  # Arancione
    'gram': '#f3d3bd'       # Verde
}
MARKERS = {
    'real': '.',
    'no_amg': 'o',
    'baseline': '^',
    'gram': 's'
}
ALPHA_REAL = 0.4
ALPHA_GEN = 0.8
NEIGHBORS_PER_SAMPLE = 10  # Quanti vicini reali prendere per ogni campione generato (per costruire il contesto)

# Configurazione per selezione audio da cluster
NUM_CLUSTERS = 60
AUDIO_PER_CLUSTER = 10  # Primi 10 audio per ogni cluster

def load_clap_model(device):
    print(f"[INFO] Loading CLAP model on {device}...")
    model = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, device=device)
    model.load_ckpt()
    model.eval()
    return model

def compute_embeddings_from_folder(model, folder_path, device):
    """Calcola gli embedding per tutti i file wav in una cartella."""
    embeddings = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    files.sort()
    
    print(f"[INFO] Computing embeddings for {folder_path} ({len(files)} files)...")
    
    for fname in tqdm(files):
        path = os.path.join(folder_path, fname)
        try:
            wav, sr = torchaudio.load(path)
            # Preprocessing standard CLAP (48kHz)
            if sr != 48000:
                resampler = T.Resample(sr, 48000)
                wav = resampler(wav)
            
            # Mix to mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # Peak normalize
            peak = wav.abs().max().clamp_min(1e-6)
            wav = (wav / peak).clamp(-1, 1)
            
            wav = wav.to(device)
            
            with torch.no_grad():
                emb = model.get_audio_embedding_from_data(x=wav, use_tensor=True)
                # Normalizza subito per cosine similarity corretta
                emb = torch.nn.functional.normalize(emb, dim=1)
                embeddings.append(emb.cpu().numpy())
                
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
            
    return np.vstack(embeddings) if embeddings else None


def compute_embeddings_from_clusters(model, base_path, device, num_clusters=NUM_CLUSTERS, audio_per_cluster=AUDIO_PER_CLUSTER):
    """
    Calcola gli embedding dai primi N audio di ogni cluster.
    
    Args:
        model: Il modello CLAP
        base_path: Path base che contiene le cartelle dei cluster (es. baseline&noAmg_full_clusters_6k/no_amg/)
        device: Device per computazione
        num_clusters: Numero di cluster da processare (default 60)
        audio_per_cluster: Numero di audio da selezionare per ogni cluster (default 10)
    
    Returns:
        np.array con tutti gli embedding
    """
    embeddings = []
    
    # Trova tutte le cartelle dei cluster (dovrebbero essere numerate 0-59)
    all_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    # Ordina numericamente se possibile
    try:
        all_dirs = sorted(all_dirs, key=lambda x: int(x))
    except ValueError:
        all_dirs = sorted(all_dirs)
    
    # Prendi solo i primi num_clusters
    cluster_dirs = all_dirs[:num_clusters]
    
    print(f"[INFO] Processing {len(cluster_dirs)} clusters from {base_path}...")
    print(f"[INFO] Selecting first {audio_per_cluster} audio from each cluster...")
    
    total_audio = 0
    for cluster_name in tqdm(cluster_dirs, desc="Clusters"):
        cluster_path = os.path.join(base_path, cluster_name)
        
        # Trova tutti i file wav nel cluster
        files = [f for f in os.listdir(cluster_path) if f.endswith('.wav')]
        files.sort()
        
        # Prendi solo i primi audio_per_cluster
        selected_files = files[:audio_per_cluster]
        
        for fname in selected_files:
            path = os.path.join(cluster_path, fname)
            try:
                wav, sr = torchaudio.load(path)
                # Preprocessing standard CLAP (48kHz)
                if sr != 48000:
                    resampler = T.Resample(sr, 48000)
                    wav = resampler(wav)
                
                # Mix to mono
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                
                # Peak normalize
                peak = wav.abs().max().clamp_min(1e-6)
                wav = (wav / peak).clamp(-1, 1)
                
                wav = wav.to(device)
                
                with torch.no_grad():
                    emb = model.get_audio_embedding_from_data(x=wav, use_tensor=True)
                    # Normalizza subito per cosine similarity corretta
                    emb = torch.nn.functional.normalize(emb, dim=1)
                    embeddings.append(emb.cpu().numpy())
                    total_audio += 1
                    
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")
    
    print(f"[INFO] Total audio processed from {base_path}: {total_audio}")
    return np.vstack(embeddings) if embeddings else None

def load_real_embeddings(json_path):
    """Carica gli embedding reali dal JSON. 
       Supporta formati dict {id: [emb]} o list of dicts.
    """
    print(f"[INFO] Loading real embeddings from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    embeddings = []
    # Logica flessibile per diversi formati JSON
    if isinstance(data, dict):
        # Caso { "id": [vector], ... }
        # o caso nidificato { "data": { "id": ... } }
        target_dict = data.get("data", data) # Prova a prendere "data" se esiste
        for k, v in target_dict.items():
            # Cerca una lista/array nel valore
            if isinstance(v, list):
                embeddings.append(v)
            elif isinstance(v, dict) and "embedding" in v:
                embeddings.append(v["embedding"])
    elif isinstance(data, list):
        # Caso [ { "embedding": ... }, ... ]
        for item in data:
            if "embedding" in item:
                embeddings.append(item["embedding"])
            elif isinstance(item, list):
                embeddings.append(item)
                
    if not embeddings:
        raise ValueError("Could not parse embeddings from JSON.")
        
    embs_np = np.array(embeddings)
    # Normalizzazione L2 anche per i reali (fondamentale per UMAP cosine)
    norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
    return embs_np / (norms + 1e-8)

def select_relevant_real_samples(gen_embs, real_embs, k=NEIGHBORS_PER_SAMPLE):
    """
    Auto-Zoom Logic: Seleziona solo i campioni reali che sono 'vicini' 
    a quelli generati per creare un plot denso ma focalizzato.
    """
    print("[INFO] Selecting relevant real samples context...")
    
    # Calcolo similarità coseno (dot product perché normalizzati)
    # gen: [300, 512], real: [N, 512]
    sim_matrix = np.dot(gen_embs, real_embs.T) # -> [300, N]
    
    # Per ogni generato, prendi gli indici dei top K reali
    top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
    
    # Unione unica di tutti gli indici trovati
    unique_indices = np.unique(top_k_indices)
    
    print(f"[INFO] Context built: {len(unique_indices)} unique real samples selected out of {len(real_embs)}.")
    return real_embs[unique_indices]


def plot_embedding(embedding_2d, idx_real, idx_no_amg, idx_gram, idx_baseline, title, output_path):
    """
    Funzione riutilizzabile per plottare gli embedding 2D.
    
    Args:
        embedding_2d: Array numpy con le coordinate 2D
        idx_real, idx_no_amg, idx_gram, idx_baseline: Slice per ogni categoria
        title: Titolo del plot
        output_path: Path base per salvare il file
    """
    plt.figure(figsize=(10, 8))
    
    # Plot Real (Background)
    plt.scatter(
        embedding_2d[idx_real, 0], 
        embedding_2d[idx_real, 1], 
        c=COLORS['real'], 
        marker=MARKERS['real'], 
        label='Training Data (Context)', 
        alpha=ALPHA_REAL, 
        s=30,
        zorder=1
    )
    
    # Plot No AMG
    plt.scatter(
        embedding_2d[idx_no_amg, 0], 
        embedding_2d[idx_no_amg, 1], 
        c=COLORS['no_amg'], 
        marker=MARKERS['no_amg'], 
        label='No AMG', 
        alpha=ALPHA_GEN, 
        edgecolors='white', 
        linewidth=0.5, 
        s=60, 
        zorder=2
    )
    
    # Plot Baseline
    plt.scatter(
        embedding_2d[idx_baseline, 0], 
        embedding_2d[idx_baseline, 1], 
        c=COLORS['baseline'], 
        marker=MARKERS['baseline'], 
        label='Baseline AMG', 
        alpha=ALPHA_GEN, 
        edgecolors='white', 
        linewidth=0.5, 
        s=60, 
        zorder=3
    )

    # Plot GRAM
    plt.scatter(
        embedding_2d[idx_gram, 0], 
        embedding_2d[idx_gram, 1], 
        c=COLORS['gram'], 
        marker=MARKERS['gram'], 
        label='GRAM (Ours)', 
        alpha=ALPHA_GEN, 
        edgecolors='white', 
        linewidth=0.5, 
        s=60, 
        zorder=4
    )
    
    plt.title(title, fontsize=14)
    plt.legend(loc='best', frameon=True)
    plt.axis('off')
    plt.tight_layout()
    
    # Salva PDF
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to {output_path}")

    # Salva anche PNG
    output_png = output_path.replace('.pdf', '.png')
    if output_png == output_path:
        output_png = output_path + ".png"
        
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to {output_png}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate UMAP distribution plot for AMG analysis.")
    parser.add_argument("--no-amg", type=str, required=True, help="Path to 'No AMG' base folder containing cluster subfolders (e.g., baseline&noAmg_full_clusters_6k/no_amg/)")
    parser.add_argument("--gram", type=str, required=True, help="Path to 'GRAM' base folder containing cluster subfolders")
    parser.add_argument("--baseline", type=str, required=True, help="Path to 'Baseline' base folder containing cluster subfolders")
    parser.add_argument("--real-json", type=str, required=True, help="Path to JSON containing real embeddings")
    parser.add_argument("--output", type=str, default="umap_distribution.pdf", help="Output filename (PDF)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for UMAP reproducibility")
    parser.add_argument("--num-clusters", type=int, default=NUM_CLUSTERS, help="Number of clusters to process (default: 60)")
    parser.add_argument("--audio-per-cluster", type=int, default=AUDIO_PER_CLUSTER, help="Number of audio to select per cluster (default: 10)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Carica Modello
    clap_model = load_clap_model(device)
    
    # 2. Calcola Embedding Generati dai cluster
    # Ogni path contiene 60 cartelle (cluster) con 100 audio ciascuno
    # Selezioniamo i primi 10 audio per ogni cluster
    print(f"\n[INFO] Processing generated audio: {args.num_clusters} clusters x {args.audio_per_cluster} audio = {args.num_clusters * args.audio_per_cluster} audio per config")
    print(f"[INFO] Total generated audio expected: 3 x {args.num_clusters * args.audio_per_cluster} = {3 * args.num_clusters * args.audio_per_cluster}")
    
    emb_no_amg = compute_embeddings_from_clusters(clap_model, args.no_amg, device, args.num_clusters, args.audio_per_cluster)
    emb_gram = compute_embeddings_from_clusters(clap_model, args.gram, device, args.num_clusters, args.audio_per_cluster)
    emb_baseline = compute_embeddings_from_clusters(clap_model, args.baseline, device, args.num_clusters, args.audio_per_cluster)
    
    if emb_no_amg is None or emb_gram is None or emb_baseline is None:
        print("[ERROR] Failed to load generated audios.")
        return

    print(f"\n[INFO] Embedding shapes: no_amg={emb_no_amg.shape}, gram={emb_gram.shape}, baseline={emb_baseline.shape}")

    # 3. Carica Reali
    emb_real_full = load_real_embeddings(args.real_json)
    
    # 4. Selezione del Sottospazio (Auto-Context)
    # Combiniamo tutti i generati per trovare i vicini nel dataset reale
    all_gen = np.vstack([emb_no_amg, emb_gram, emb_baseline])
    emb_real_subset = select_relevant_real_samples(all_gen, emb_real_full)
    
    # Creiamo il dataset combinato: [Real_Subset, No_AMG, GRAM, Baseline]
    X_combined = np.vstack([emb_real_subset, emb_no_amg, emb_gram, emb_baseline])
    
    # Separazione indici per plotting
    n_real = len(emb_real_subset)
    n_no_amg = len(emb_no_amg)
    n_gram = len(emb_gram)
    
    idx_real = slice(0, n_real)
    idx_no_amg = slice(n_real, n_real + n_no_amg)
    idx_gram = slice(n_real + n_no_amg, n_real + n_no_amg + n_gram)
    idx_baseline = slice(n_real + n_no_amg + n_gram, None)
    
    # 5. UMAP Projection
    print("[INFO] Running UMAP...")
    reducer_umap = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.0,
        spread=0.1, 
        metric='cosine', 
        random_state=args.seed
    )
    embedding_umap = reducer_umap.fit_transform(X_combined)
    
    # 6. t-SNE Projection
    print("[INFO] Running t-SNE...")
    reducer_tsne = TSNE(
        n_components=2,
        perplexity=100,
        metric='cosine',
        random_state=args.seed,
        n_iter=2000,
        init='pca'
    )
    embedding_tsne = reducer_tsne.fit_transform(X_combined)
    
    # 7. Plotting UMAP
    print("[INFO] Plotting UMAP...")
    plot_embedding(embedding_umap, idx_real, idx_no_amg, idx_gram, idx_baseline, 
                   'Latent Space Distribution (UMAP)', args.output)
    
    # 8. Plotting t-SNE
    print("[INFO] Plotting t-SNE...")
    # Genera nome file per t-SNE
    tsne_output = args.output.replace('.pdf', '_tsne.pdf').replace('.png', '_tsne.png')
    if tsne_output == args.output:
        tsne_output = args.output + '_tsne'
    plot_embedding(embedding_tsne, idx_real, idx_no_amg, idx_gram, idx_baseline,
                   'Latent Space Distribution (t-SNE)', tsne_output)

if __name__ == "__main__":
    main()