import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects # Import necessario per gli effetti testo
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap # pip install umap-learn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
from tqdm import tqdm

# Import dai tuoi moduli esistenti
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference import amg_generation
from stable_audio_tools.inference.amg_generation import my_generate_diffusion_cond

# --- CONFIGURAZIONE ---
OUTPUT_DIR = "./batch_output_baseline_no_sphere_50"
NUM_GENERATIONS = 50 # O quanti ne vuoi fare
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
EMBEDDINGS_FILE = 'embeddings_new.json'

# TARGET PRINCIPALE (Prompt corrente)
TARGET_ID = "5131"

# CLUSTER 49 (Altri membri da evidenziare)
CLUSTER_IDS = [
    "5117", "5119", "5121", "5123", "5128", 
    "5130", "5131", "5134", "5139", "5144", "5145"
]

# Parametri Generazione
SEED_START = -1 # Random
CFG_SCALE = 7
STEPS = 100
DURATION = 1.7143310657596371 
PROMPT = "\"ATTACK loop 140 bpm-00.wav\" till \"ATTACK loop 140 bpm-31.wav\"\r\nare all part of the \"ATTACK LOOP 6\" sample package and belong together\r\nas they are all variations on the same 1 measure 4/4 140 bpm drumloop. The loop has a techno-trance\r\nfeel. The first four loops (00 till 03) contain some variations of the\r\npure drumloop, where 00 is the most minimal and 03 the fullest. All\r\nother variations add other sound effects, some of them being sounds\r\nwith a certain pitch, mostly C. These loop are suitable for your trance\r\nand techno productions. They were created using the Waldorf Attack VSTi within Cubase SX. Mastering (EQ, Stereo Enhancer, Multi-Band expand/compress/limit, dither, fades at start and/or end) done within Wavelab.\r\n"

# Parametri AMG / GRAM
C_GRAM = 0.0
GRAM_SCALE = 0.6 
LAMBDA_MIN = 0.7
LAMBDA_MAX = 0.8

# --- SETUP AMBIENTE ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.backends.cuda.enable_math_sdp(True) # Decommenta per determinismo

# --- CARICAMENTO DATI TRAINING ---
print("Caricamento embeddings di training...")
with open(EMBEDDINGS_FILE, 'r') as f:
    train_data = json.load(f)

train_ids = sorted(list(train_data.keys()))
train_embeddings = torch.stack([
    torch.tensor(train_data[sound_id]['embedding'], dtype=torch.float32)
    for sound_id in train_ids
], dim=0).to(DEVICE)

# Gestione Target e Cluster Indices
target_train_idx = None
if TARGET_ID in train_ids:
    target_train_idx = train_ids.index(TARGET_ID)

cluster_train_indices = []
for cid in CLUSTER_IDS:
    if cid in train_ids:
        cluster_train_indices.append(train_ids.index(cid))
    else:
        print(f"Warning: Cluster ID {cid} non trovato nel dataset.")

# Caricamento Embedding Target per calcoli similarità
if TARGET_ID in train_data:
    target_embedding = torch.tensor(train_data[TARGET_ID]['embedding'], device=DEVICE, dtype=torch.float32)
    target_embedding_norm = F.normalize(target_embedding.unsqueeze(0), p=2, dim=1)
else:
    target_embedding_norm = None

train_embeddings_norm = F.normalize(train_embeddings, p=2, dim=1)

# --- CARICAMENTO MODELLO ---
print("Caricamento modelli...")
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
downsampling_ratio = model.pretransform.downsampling_ratio if hasattr(model, 'pretransform') else 1
model = model.to(DEVICE)

CLAP = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, device=DEVICE)
CLAP.load_ckpt()

# Prompt Embedding
text_embed = CLAP.get_text_embedding([PROMPT], use_tensor=True)
if isinstance(text_embed, torch.Tensor): text_embed = text_embed.to(DEVICE)
else: text_embed = torch.tensor(text_embed).to(DEVICE)
text_embed_norm = F.normalize(text_embed, p=2, dim=1)

requested_samples = int(DURATION * sample_rate)
generation_sample_size = int(np.ceil(requested_samples / downsampling_ratio) * downsampling_ratio)

# --- LOOP DI GENERAZIONE ---
generated_embeddings_list = []
generated_files = []

print(f"Avvio generazione di {NUM_GENERATIONS} file...")

for i in tqdm(range(NUM_GENERATIONS)):
    current_seed = SEED_START + i if SEED_START != -1 else -1
    
    conditioning = [{"prompt": PROMPT, "seconds_start": 0, "seconds_total": DURATION}]
    negative_conditioning = [{"prompt": "", "seconds_start": 0, "seconds_total": DURATION}]
    
    output = my_generate_diffusion_cond(
        model,
        steps=STEPS,
        cfg_scale=CFG_SCALE,
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        sample_size=generation_sample_size,
        sample_rate=sample_rate,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="my-dpmpp-3m-sde",
        device=DEVICE,
        c1=6, c2=6, c3=1000, 
        c_gram=C_GRAM,
        gram_neighborhood_scale=GRAM_SCALE,
        lambda_min=LAMBDA_MIN,
        lambda_max=LAMBDA_MAX,
        seed=current_seed
    )

    output = rearrange(output, "b d n -> d (b n)")
    output = output[:, :requested_samples]
    
    output_float = output.to(torch.float32)
    peak = output_float.abs().max().clamp_min(1e-6)
    output_normalized = (output_float / peak).clamp(-1, 1)
    
    filename = os.path.join(OUTPUT_DIR, f"gen_{i:03d}.wav")
    output_int16 = output_normalized.mul(32767).to(torch.int16).cpu()
    torchaudio.save(filename, output_int16, sample_rate)
    generated_files.append(filename)
    
    with torch.no_grad():
        mono_audio = output_normalized.mean(dim=0, keepdim=True).to(DEVICE)
        emb = CLAP.get_audio_embedding_from_data(x=mono_audio, use_tensor=True)[0]
        generated_embeddings_list.append(emb)

gen_embeddings = torch.stack(generated_embeddings_list).to(DEVICE)
gen_embeddings_norm = F.normalize(gen_embeddings, p=2, dim=1)

# --- ANALISI STATISTICA ---
print("\n" + "="*40)
print("       RISULTATI ANALISI       ")
print("="*40)

cosine_sim_matrix = torch.mm(gen_embeddings_norm, train_embeddings_norm.T)
global_avg_sim = cosine_sim_matrix.mean().item()
max_sim_per_gen = cosine_sim_matrix.max(dim=1).values
global_avg_max_sim = max_sim_per_gen.mean().item()

if target_embedding_norm is not None:
    target_sims = torch.mm(gen_embeddings_norm, target_embedding_norm.T)
    avg_target_sim = target_sims.mean().item()
    max_target_sim = target_sims.max().item()
else:
    avg_target_sim = 0.0
    max_target_sim = 0.0

prompt_sims = torch.mm(gen_embeddings_norm, text_embed_norm.T)
avg_prompt_adherence = prompt_sims.mean().item()

gen_sim_matrix = torch.mm(gen_embeddings_norm, gen_embeddings_norm.T)
mask_diag = torch.eye(gen_sim_matrix.shape[0], device=DEVICE).bool()
off_diag_sims = gen_sim_matrix[~mask_diag]
intra_list_diversity = 1.0 - off_diag_sims.mean().item() if len(off_diag_sims) > 0 else 0.0
total_variance = torch.var(gen_embeddings, dim=0).sum().item()

print(f"Generazioni Totali: {NUM_GENERATIONS}")
print("-" * 30)
print(f"Global Avg Similarity: {global_avg_sim:.4f}")
print(f"Avg Nearest Neighbor Sim: {global_avg_max_sim:.4f}")
print(f"Avg Target Sim ({TARGET_ID}): {avg_target_sim:.4f}")
print(f"Div: {intra_list_diversity:.4f} | Var: {total_variance:.4f}")
print("="*40)


# --- FUNZIONE DI PLOTTING AGGIORNATA ---
def plot_embedding_space(X_2d, method_name, filename, 
                         global_avg_sim, avg_target_sim, avg_prompt_adherence, 
                         intra_list_diversity, total_variance, max_target_sim, global_avg_max_sim,
                         train_offset, best_gen_indices, 
                         target_idx=None, cluster_indices=None): # NUOVO ARGOMENTO
    
    plt.figure(figsize=(12, 11))
    
    X_train = X_2d[:train_offset]
    X_gen = X_2d[train_offset:]
    
    # --- 1. Plot Training Data (Background) ---
    # Creiamo una maschera per escludere Target e Cluster dal grigio
    mask_bg = np.ones(len(X_train), dtype=bool)
    
    if target_idx is not None:
        mask_bg[target_idx] = False
    
    if cluster_indices is not None:
        mask_bg[cluster_indices] = False
        
    plt.scatter(X_train[mask_bg, 0], X_train[mask_bg, 1], c='lightgray', label='Training Neighbors', alpha=0.4, s=30, edgecolors='grey', linewidth=0.3)

    # --- 2. Plot Generated Data ---
    plt.scatter(
        X_gen[:, 0], X_gen[:, 1], 
        c='red', label='Generated Audio (AMG)', 
        alpha=0.8, s=40, edgecolors='black', linewidth=0.5
    )

    # --- 3. Plot Cluster 49 (STELLE CIANO) ---
    # Escludiamo il target principale da questo gruppo per non disegnarci sopra due volte
    if cluster_indices is not None:
        # Filtriamo il target_idx se presente nella lista cluster
        cluster_only_indices = [idx for idx in cluster_indices if idx != target_idx]
        
        if len(cluster_only_indices) > 0:
            cluster_coords = X_train[cluster_only_indices]
            plt.scatter(
                cluster_coords[:, 0], cluster_coords[:, 1],
                c='cyan', marker='*', s=200, # Stelle Ciano
                edgecolors='black', linewidth=0.8,
                label='Cluster 49 Members',
                zorder=9
            )

    # --- 4. Annotazione Migliori Generazioni ---
    if best_gen_indices is not None:
        for i, gen_idx in enumerate(best_gen_indices):
            if gen_idx < len(X_gen):
                x, y = X_gen[gen_idx]
                plt.annotate(
                    str(gen_idx), (x, y),
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='darkblue',
                    path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')]
                )

    # --- 5. Plot Target (STELLA ORO) ---
    if target_idx is not None:
        target_coords = X_train[target_idx]
        plt.scatter(
            target_coords[0], target_coords[1],
            c='gold', marker='*', s=450, # Stella Oro Grande
            edgecolors='black', linewidth=1.5,
            label=f'Target ({TARGET_ID})',
            zorder=10
        )
        plt.annotate(
            TARGET_ID, (target_coords[0], target_coords[1]),
            xytext=(10, -15), textcoords='offset points',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
        )

    title_str = (
        f"{method_name} Analysis\n"
        f"Glob Sim: {global_avg_sim:.3f} | Tgt Sim: {avg_target_sim:.3f} | Prompt Adh: {avg_prompt_adherence:.3f}\n"
        f"Div: {intra_list_diversity:.3f} | Var: {total_variance:.1f} | Max Tgt: {max_target_sim:.3f} | NN Sim: {global_avg_max_sim:.3f}"
    )
    
    plt.title(title_str, fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    full_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(full_path, dpi=300)
    print(f"Grafico salvato in: {full_path}")
    plt.close()

# --- PREPARAZIONE DATI PER GRAFICI ---
print("\nPreparazione Dati Grafici...")

# Classifica Migliori
best_gen_indices = None
if target_embedding_norm is not None:
    num_total_gens = target_sims.flatten().shape[0]
    k_best = min(10, num_total_gens)
    vals, indices = torch.topk(target_sims.flatten(), k=k_best, largest=False)
    best_gen_indices = indices.cpu().numpy()

# Selezione Vicini per T-SNE/PCA
max_neighbors = min(train_embeddings.shape[0], 511)
k_neighbors = int(1 + round(GRAM_SCALE * (max_neighbors - 1)))
vals, indices = torch.topk(cosine_sim_matrix, k=k_neighbors, dim=1, largest=True)
relevant_train_indices = torch.unique(indices.flatten()).cpu().numpy()

# --- INCLUSIONE FORZATA CLUSTER E TARGET ---
# Aggiungiamo tutti gli indici del cluster e del target ai dati da plottare
indices_to_add = []
if target_train_idx is not None: indices_to_add.append(target_train_idx)
indices_to_add.extend(cluster_train_indices)

relevant_train_indices = np.union1d(relevant_train_indices, np.array(indices_to_add))
relevant_train_indices = np.sort(relevant_train_indices)

# Mapping degli indici globali -> indici locali nel subset X_train
target_subset_idx = np.searchsorted(relevant_train_indices, target_train_idx) if target_train_idx is not None else None

cluster_subset_indices = []
for cid_idx in cluster_train_indices:
    local_idx = np.searchsorted(relevant_train_indices, cid_idx)
    cluster_subset_indices.append(local_idx)

relevant_train_embeds = train_embeddings[relevant_train_indices].cpu().numpy()
gen_embeds_np = gen_embeddings.cpu().numpy()

X_combined = np.concatenate([relevant_train_embeds, gen_embeds_np], axis=0)
train_offset = len(relevant_train_embeds)
n_samples_total = X_combined.shape[0]

print(f"Punti totali da plottare: {n_samples_total} (Training: {train_offset}, Gen: {NUM_GENERATIONS})")

# --- 1. ESECUZIONE T-SNE ---
print("Calcolo t-SNE...")
perp_val = min(30, n_samples_total - 1) if n_samples_total > 1 else 1
tsne = TSNE(n_components=2, random_state=42, perplexity=perp_val, n_iter=1000)
X_tsne = tsne.fit_transform(X_combined)

plot_embedding_space(
    X_tsne, "t-SNE", "analysis_tsne.png",
    global_avg_sim, avg_target_sim, avg_prompt_adherence, 
    intra_list_diversity, total_variance, max_target_sim, global_avg_max_sim,
    train_offset, best_gen_indices, 
    target_idx=target_subset_idx, cluster_indices=cluster_subset_indices # PASSAGGIO INDICI
)

# --- 2. ESECUZIONE PCA ---
if n_samples_total >= 2:
    print("Calcolo PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    expl_var = pca.explained_variance_ratio_.sum() * 100
    plot_embedding_space(
        X_pca, f"PCA (Var: {expl_var:.1f}%)", "analysis_pca.png",
        global_avg_sim, avg_target_sim, avg_prompt_adherence, 
        intra_list_diversity, total_variance, max_target_sim, global_avg_max_sim,
        train_offset, best_gen_indices,
        target_idx=target_subset_idx, cluster_indices=cluster_subset_indices
    )

# --- 3. ESECUZIONE UMAP ---
print("Calcolo UMAP...")
n_neigh = min(15, n_samples_total - 1) if n_samples_total > 1 else 1
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neigh, min_dist=0.1)
X_umap = reducer.fit_transform(X_combined)

plot_embedding_space(
    X_umap, "UMAP", "analysis_umap.png",
    global_avg_sim, avg_target_sim, avg_prompt_adherence, 
    intra_list_diversity, total_variance, max_target_sim, global_avg_max_sim,
    train_offset, best_gen_indices,
    target_idx=target_subset_idx, cluster_indices=cluster_subset_indices
)

print("\nAnalisi completa e grafici salvati.")