import torch
import os
import json
import re
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap 
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from datetime import datetime

# Import CLAP for audio embeddings
from stable_audio_tools.inference import amg_generation

# --- CONFIGURATION ---
EMBEDDINGS_FILE = 'embeddings_new.json'
CLUSTERS_FILE = 'clusters.json'  # Cluster ID -> list of member IDs
CLUSTER_REPRESENTATIVES_FILE = 'cluster_representatives.csv'  # Cluster ID -> representative ID
DATASET_DIR = 'soundDataset'  # Directory with original audio files for on-the-fly embedding extraction
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
MAX_TRAIN_EMBEDDINGS = 512  # Maximum number of training embeddings to visualize


def extract_dataset_embeddings(clap_model, train_data, device, dataset_dir=DATASET_DIR):
    """
    Extract embeddings from audio files using the specified CLAP model.
    This is used when a different CLAP model is selected (not clap-laion-audio).
    
    Args:
        clap_model: Loaded CLAP model instance
        train_data: Dict from embeddings_new.json (for IDs and conditioning)
        device: Device to use
        dataset_dir: Path to soundDataset directory
    
    Returns:
        dict: sound_id -> embedding tensor
    """
    print(f"\nExtracting embeddings from {dataset_dir} using current CLAP model...")
    print(f"This may take a while for {len(train_data)} files...")
    
    embeddings_dict = {}
    missing_files = []
    
    sound_ids = sorted(list(train_data.keys()))
    
    for sound_id in tqdm(sound_ids, desc="Extracting embeddings"):
        audio_path = os.path.join(dataset_dir, f"sound_{sound_id}.wav")
        
        if not os.path.exists(audio_path):
            missing_files.append(sound_id)
            # Use zero embedding as placeholder
            embeddings_dict[sound_id] = torch.zeros(512, device=device)
            continue
        
        try:
            # Load and preprocess audio
            audio, sr = torchaudio.load(audio_path)
            
            # Resample to 48000 if needed
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000)
                audio = resampler(audio)
            
            # Normalize
            audio = audio.to(torch.float32)
            peak = audio.abs().max().clamp_min(1e-6)
            audio = (audio / peak).clamp(-1, 1)
            
            # Get mono
            mono_audio = audio.mean(dim=0, keepdim=True).to(device)
            
            with torch.no_grad():
                emb = clap_model.get_audio_embedding_from_data(x=mono_audio, use_tensor=True)[0]
                embeddings_dict[sound_id] = emb.to(device)
                
        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")
            embeddings_dict[sound_id] = torch.zeros(512, device=device)
            missing_files.append(sound_id)
    
    if missing_files:
        print(f"  Warning: {len(missing_files)} files missing or failed to process")
    
    print(f"  Extracted {len(embeddings_dict)} embeddings")
    return embeddings_dict


def load_clusters_json(json_path):
    """Load cluster membership from clusters.json.
    Returns a dict: cluster_id -> list of member IDs
    """
    if not os.path.exists(json_path):
        print(f"Warning: Clusters file {json_path} not found")
        return {}
    
    with open(json_path, 'r') as f:
        clusters = json.load(f)
    
    # Ensure all keys and values are strings
    return {str(k): [str(v) for v in vals] for k, vals in clusters.items()}


def load_cluster_representatives(csv_path):
    """Load cluster representatives from CSV file.
    Returns a dict: representative_id -> cluster_id
    """
    rep_to_cluster = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Cluster representatives file {csv_path} not found")
        return rep_to_cluster
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster_id = str(row['cluster_id'])
            representative_id = str(row['representative_id'])
            rep_to_cluster[representative_id] = cluster_id
    
    return rep_to_cluster


def get_cluster_members_for_target(target_id, clusters, rep_to_cluster):
    """Get all cluster members for a target ID.
    Returns list of member IDs (excluding the target itself).
    """
    # Check if target is a representative
    cluster_id = rep_to_cluster.get(target_id)
    
    if cluster_id is None:
        # Target might be a member, search all clusters
        for cid, members in clusters.items():
            if target_id in members:
                cluster_id = cid
                break
    
    if cluster_id is None:
        return []
    
    # Get all members of this cluster, excluding target
    members = clusters.get(cluster_id, [])
    return [m for m in members if m != target_id]

# --- HELPER FUNCTIONS ---
def extract_id_from_filename(filename):
    """Extract the sound ID from filename like 'sound_6237_1.wav' -> '6237'"""
    match = re.match(r'sound_(\d+)_\d+\.wav', filename)
    if match:
        return match.group(1)
    return None


def compute_log_rge_for_sample(
    gen_embedding: torch.Tensor,
    candidate_embeddings_norm: torch.Tensor,
    cosine_sim_row: torch.Tensor,
    k_neighbors: int = 100,
    epsilon: float = 1e-6,
):
    """Compute Log-RGE for one generated sample using its k nearest candidate neighbors.

    The vectors are assumed to be L2-normalized already.
    The score is log(det(G_aug)) - log(det(G_base)), where G_base is the Gram matrix
    of the neighbor set and G_aug is the Gram matrix after appending the generated sample.
    """

    num_candidates, embedding_dim = candidate_embeddings_norm.shape
    if num_candidates == 0 or embedding_dim <= 1:
        return None

    max_k = min(k_neighbors, num_candidates, embedding_dim - 1)
    if max_k < 1:
        return None

    topk_indices = torch.topk(cosine_sim_row, k=max_k, dim=0, largest=True).indices
    neighbor_embeddings = candidate_embeddings_norm[topk_indices]

    def stable_logdet_from_rows(embeddings: torch.Tensor):
        embeddings_64 = embeddings.to(torch.float64)
        gram_matrix = embeddings_64 @ embeddings_64.T
        gram_matrix = gram_matrix + torch.eye(
            gram_matrix.shape[0], device=gram_matrix.device, dtype=torch.float64
        ) * epsilon
        sign, log_det = torch.linalg.slogdet(gram_matrix)
        if sign <= 0:
            return None
        return log_det

    base_log_det = stable_logdet_from_rows(neighbor_embeddings)
    if base_log_det is None:
        return None

    augmented_embeddings = torch.cat([neighbor_embeddings, gen_embedding.unsqueeze(0)], dim=0)
    aug_log_det = stable_logdet_from_rows(augmented_embeddings)
    if aug_log_det is None:
        return None

    return (aug_log_det - base_log_det).to(gen_embedding.dtype)

def plot_combined_analysis(X_tsne, X_pca, X_umap, subdir_name,
                           metrics, train_offset, best_gen_indices,
                           target_idx=None, cluster_indices=None,
                           cluster_ids=None, target_id=None, 
                           output_path=None, pca_var=None):
    """Create a combined figure with t-SNE, PCA, and UMAP subplots.
    
    Args:
        cluster_indices: list of indices in the subset for cluster members
        cluster_ids: list of IDs corresponding to cluster_indices (for labels)
        target_id: the ID of the target (for label)
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    datasets = [
        (X_tsne, "t-SNE", axes[0]),
        (X_pca, f"PCA (Var: {pca_var:.1f}%)" if pca_var else "PCA", axes[1]),
        (X_umap, "UMAP", axes[2])
    ]
    
    for X_2d, method_name, ax in datasets:
        X_train = X_2d[:train_offset]
        X_gen = X_2d[train_offset:]
        
        # 1. Plot Training Data (Background)
        mask_bg = np.ones(len(X_train), dtype=bool)
        
        if target_idx is not None and target_idx < len(mask_bg):
            mask_bg[target_idx] = False
        
        if cluster_indices is not None:
            for idx in cluster_indices:
                if idx < len(mask_bg):
                    mask_bg[idx] = False
        
        ax.scatter(X_train[mask_bg, 0], X_train[mask_bg, 1], 
                   c='lightgray', label='Training Neighbors', 
                   alpha=0.4, s=30, edgecolors='grey', linewidth=0.3)

        # 2. Plot Generated Data
        ax.scatter(X_gen[:, 0], X_gen[:, 1], 
                   c='red', label='Generated Audio', 
                   alpha=0.8, s=40, edgecolors='black', linewidth=0.5)

        # 3. Plot Cluster Members (CYAN STARS with ID labels)
        if cluster_indices is not None and len(cluster_indices) > 0:
            # Exclude target from cluster display
            cluster_only_indices = []
            cluster_only_ids = []
            for i, idx in enumerate(cluster_indices):
                if idx != target_idx and idx < len(X_train):
                    cluster_only_indices.append(idx)
                    if cluster_ids is not None and i < len(cluster_ids):
                        cluster_only_ids.append(cluster_ids[i])
            
            if len(cluster_only_indices) > 0:
                cluster_coords = X_train[cluster_only_indices]
                ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                           c='cyan', marker='*', s=200,
                           edgecolors='black', linewidth=0.8,
                           label='Cluster Members', zorder=9)
                
                # Add ID labels next to each cluster member star
                for i, (cx, cy) in enumerate(cluster_coords):
                    if i < len(cluster_only_ids):
                        member_id = cluster_only_ids[i]
                        ax.annotate(str(member_id), (cx, cy),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, fontweight='bold', color='darkblue',
                                    bbox=dict(boxstyle="round,pad=0.2", fc="cyan", alpha=0.6),
                                    path_effects=[PathEffects.withStroke(linewidth=1, foreground='white')])

        # 4. Annotate Best Generations
        if best_gen_indices is not None:
            for i, gen_idx in enumerate(best_gen_indices):
                if gen_idx < len(X_gen):
                    x, y = X_gen[gen_idx]
                    ax.annotate(str(gen_idx), (x, y),
                                xytext=(0, 5), textcoords='offset points',
                                fontsize=9, fontweight='bold', color='darkblue',
                                path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])

        # 5. Plot Target (GOLD STAR with ID label)
        if target_idx is not None and target_idx < len(X_train):
            target_coords = X_train[target_idx]
            ax.scatter(target_coords[0], target_coords[1],
                       c='gold', marker='*', s=450,
                       edgecolors='black', linewidth=1.5,
                       label=f'Target ({target_id})', zorder=10)
            ax.annotate(str(target_id), (target_coords[0], target_coords[1]),
                        xytext=(10, -15), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

        ax.set_title(method_name, fontsize=12)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Add metrics as super title
    title_str = (
        f"{subdir_name} Analysis\n"
        f"Glob Sim: {metrics['global_avg_sim']:.3f} | Tgt Sim: {metrics['avg_target_sim']:.3f} | "
        f"Prompt Adh: {metrics['avg_prompt_adherence']:.3f}\n"
        f"Div: {metrics['intra_list_diversity']:.3f} | Var: {metrics['total_variance']:.1f} | "
        f"Max Tgt: {metrics['max_target_sim']:.3f} | NN Sim: {metrics['global_avg_max_sim']:.3f}"
    )
    fig.suptitle(title_str, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {output_path}")
    
    plt.close()


def process_subdir(subdir_path, subdir_name, train_data, train_ids, 
                   train_embeddings, train_embeddings_norm, CLAP, device,
                   clusters=None, rep_to_cluster=None, viz_scale=1.0, skip_plots=False):
    """Process a single subdirectory and return metrics + embeddings
    
    Args:
        clusters: dict from load_clusters_json() - cluster_id -> list of member IDs
        rep_to_cluster: dict from load_cluster_representatives() - rep_id -> cluster_id
        viz_scale: float 0-1, controls how many training embeddings to display
                   (1.0 = 512 embeddings, 0.5 = 256, etc.)
        skip_plots: bool, if True skip generating visualization plots
    """
    
    # Get all wav files - support both patterns:
    # - sound_ID_N.wav (random prompts)
    # - gen_N.wav (cluster mode)
    wav_files = [f for f in os.listdir(subdir_path) 
                 if f.endswith('.wav') and (re.match(r'sound_\d+_\d+\.wav', f) or 
                                             re.match(r'gen_\d+\.wav', f))]
    
    if not wav_files:
        print(f"  No valid wav files found in {subdir_name}, skipping...")
        return None
    
    print(f"\nProcessing {subdir_name} ({len(wav_files)} files)...")
    
    # Try to load cluster_info.json if it exists (cluster mode)
    cluster_info_path = os.path.join(subdir_path, "cluster_info.json")
    target_id = None
    is_cluster_mode = False
    
    if os.path.exists(cluster_info_path):
        with open(cluster_info_path, 'r') as f:
            cluster_info = json.load(f)
        target_id = str(cluster_info.get('representative_id'))
        is_cluster_mode = True
        print(f"  Cluster mode: Target ID = {target_id} (cluster {cluster_info.get('cluster_id')})")
    else:
        # Random mode: group files by target ID from filename
        files_by_target = {}
        for wav_file in wav_files:
            sound_id = extract_id_from_filename(wav_file)
            if sound_id and sound_id in train_data:
                if sound_id not in files_by_target:
                    files_by_target[sound_id] = []
                files_by_target[sound_id].append(wav_file)
        
        if not files_by_target:
            print(f"  No matching IDs found in {subdir_name}, skipping...")
            return None
        
        # Get the primary target ID (the one with most files)
        target_id = max(files_by_target.keys(), key=lambda x: len(files_by_target[x]))
    
    if target_id is None or target_id not in train_data:
        print(f"  Target ID {target_id} not found in training data, skipping...")
        return None
    
    # For cluster mode, all files are for this target; for random mode, filter
    if is_cluster_mode:
        target_files = wav_files
    else:
        target_files = files_by_target.get(target_id, wav_files)
    
    print(f"  Target ID: {target_id} ({len(target_files)} generations)")
    
    # Get target embedding
    target_train_idx = train_ids.index(target_id) if target_id in train_ids else None
    target_embedding = torch.tensor(train_data[target_id]['embedding'], device=device, dtype=torch.float32)
    target_embedding_norm = F.normalize(target_embedding.unsqueeze(0), p=2, dim=1)
    
    # Get prompt for this target
    prompt = train_data[target_id]['conditioning']['prompt']
    
    # Get text embedding for prompt
    text_embed = CLAP.get_text_embedding([prompt], use_tensor=True)
    if isinstance(text_embed, torch.Tensor):
        text_embed = text_embed.to(device)
    else:
        text_embed = torch.tensor(text_embed).to(device)
    text_embed_norm = F.normalize(text_embed, p=2, dim=1)

    # Get cluster members from clusters.json (used both for Log-RGE and visualization).
    cluster_member_ids = []
    if clusters is not None and rep_to_cluster is not None:
        cluster_member_ids = get_cluster_members_for_target(target_id, clusters, rep_to_cluster)
    print(f"  Found {len(cluster_member_ids)} cluster members from clusters.json")

    # Candidate pool for Log-RGE: current cluster only, including the representative.
    cluster_candidate_train_indices = []
    if target_train_idx is not None:
        cluster_candidate_train_indices.append(target_train_idx)
    for member_id in cluster_member_ids:
        if member_id in train_ids:
            cluster_candidate_train_indices.append(train_ids.index(member_id))
    cluster_candidate_train_indices = sorted(set(cluster_candidate_train_indices))
    cluster_candidate_embeddings_norm = (
        train_embeddings_norm[cluster_candidate_train_indices]
        if cluster_candidate_train_indices
        else None
    )
    
    # Process all files for this target
    generated_embeddings_list = []
    generated_files = []
    
    for wav_file in tqdm(sorted(target_files), desc=f"  {subdir_name}"):
        wav_path = os.path.join(subdir_path, wav_file)
        
        try:
            audio, sr = torchaudio.load(wav_path)
        except Exception as e:
            print(f"    Error loading {wav_file}: {e}")
            continue
        
        # Resample to 48000 if needed (CLAP expects 48000)
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            audio = resampler(audio)
        
        # Normalize audio
        audio = audio.to(torch.float32)
        peak = audio.abs().max().clamp_min(1e-6)
        audio = (audio / peak).clamp(-1, 1)
        
        # Get mono audio
        mono_audio = audio.mean(dim=0, keepdim=True).to(device)
        
        with torch.no_grad():
            emb = CLAP.get_audio_embedding_from_data(x=mono_audio, use_tensor=True)[0]
            generated_embeddings_list.append(emb)
            generated_files.append(wav_file)
    
    if not generated_embeddings_list:
        print(f"  No embeddings generated for {subdir_name}, skipping...")
        return None
    
    # Stack embeddings
    gen_embeddings = torch.stack(generated_embeddings_list).to(device)
    gen_embeddings_norm = F.normalize(gen_embeddings, p=2, dim=1)
    
    # --- CALCULATE METRICS ---
    cosine_sim_matrix = torch.mm(gen_embeddings_norm, train_embeddings_norm.T)
    
    # Local normalized Log-RGE using only the current cluster as candidate pool.
    cluster_log_rge_list = []
    if cluster_candidate_embeddings_norm is not None:
        cluster_cosine_sim_matrix = torch.mm(gen_embeddings_norm, cluster_candidate_embeddings_norm.T)
        for gen_idx in range(gen_embeddings_norm.shape[0]):
            log_rge = compute_log_rge_for_sample(
                gen_embedding=gen_embeddings_norm[gen_idx],
                candidate_embeddings_norm=cluster_candidate_embeddings_norm,
                cosine_sim_row=cluster_cosine_sim_matrix[gen_idx],
                k_neighbors=100,
            )
            if log_rge is not None:
                cluster_log_rge_list.append(float(log_rge.item()))
    avg_cluster_log_rge = float(np.mean(cluster_log_rge_list)) if cluster_log_rge_list else float('nan')
    
    global_avg_sim = cosine_sim_matrix.mean().item()
    max_sim_per_gen = cosine_sim_matrix.max(dim=1).values
    global_avg_max_sim = max_sim_per_gen.mean().item()

    # ratio between first and second NN-sim (guard when there are <2 training embeddings)
    if train_embeddings.shape[0] >= 2:
        top2_vals = torch.topk(cosine_sim_matrix, k=2, dim=1, largest=True).values
        top1_sim = top2_vals[:, 0]
        top2_sim = top2_vals[:, 1]
        nn_ratio_per_gen = top1_sim / torch.clamp(top2_sim, min=1e-8)
        avg_nn_ratio = nn_ratio_per_gen.mean().item()
    else:
        avg_nn_ratio = float('nan')
    
    # Target similarity
    target_sims = torch.mm(gen_embeddings_norm, target_embedding_norm.T)
    avg_target_sim = target_sims.mean().item()
    max_target_sim = target_sims.max().item()
    
    # Prompt adherence
    prompt_sims = torch.mm(gen_embeddings_norm, text_embed_norm.T)
    avg_prompt_adherence = prompt_sims.mean().item()
    
    # Diversity
    gen_sim_matrix = torch.mm(gen_embeddings_norm, gen_embeddings_norm.T)
    mask_diag = torch.eye(gen_sim_matrix.shape[0], device=device).bool()
    off_diag_sims = gen_sim_matrix[~mask_diag]
    intra_list_diversity = 1.0 - off_diag_sims.mean().item() if len(off_diag_sims) > 0 else 0.0
    total_variance = torch.var(gen_embeddings, dim=0).mean().item()
    
    
    metrics = {
        'global_avg_sim': global_avg_sim,
        'global_avg_max_sim': global_avg_max_sim,
        'avg_nn_ratio': avg_nn_ratio,
        'avg_cluster_log_rge': avg_cluster_log_rge,
        'avg_target_sim': avg_target_sim,
        'max_target_sim': max_target_sim,
        'avg_prompt_adherence': avg_prompt_adherence,
        'intra_list_diversity': intra_list_diversity,
        'total_variance': total_variance
    }
    
    # Get best generations (lowest similarity to target)
    num_total_gens = target_sims.flatten().shape[0]
    k_best = min(10, num_total_gens)
    vals, indices = torch.topk(target_sims.flatten(), k=k_best, largest=False)
    best_gen_indices = indices.cpu().numpy()
    
    # Select neighbors for visualization based on viz_scale
    # viz_scale=1.0 -> 512 embeddings, viz_scale=0.5 -> 256, etc.
    num_neighbors = max(1, int(viz_scale * MAX_TRAIN_EMBEDDINGS))
    
    # Get most similar training embeddings to generated ones
    vals, indices = torch.topk(cosine_sim_matrix, k=min(num_neighbors, train_embeddings.shape[0]), dim=1, largest=True)
    relevant_train_indices = torch.unique(indices.flatten()).cpu().numpy()
    
    # Force include target
    if target_train_idx is not None:
        relevant_train_indices = np.union1d(relevant_train_indices, np.array([target_train_idx]))
    
    # Force include all cluster members in visualization
    cluster_member_train_indices = []
    for member_id in cluster_member_ids:
        if member_id in train_ids:
            idx = train_ids.index(member_id)
            cluster_member_train_indices.append(idx)
            relevant_train_indices = np.union1d(relevant_train_indices, np.array([idx]))
    
    relevant_train_indices = np.sort(relevant_train_indices)
    
    # Map to local indices
    target_subset_idx = np.searchsorted(relevant_train_indices, target_train_idx) if target_train_idx is not None else None
    
    # Map cluster member indices to local subset indices
    cluster_subset_indices = []
    cluster_subset_ids = []
    for member_id in cluster_member_ids:
        if member_id in train_ids:
            global_idx = train_ids.index(member_id)
            local_idx = np.searchsorted(relevant_train_indices, global_idx)
            if local_idx < len(relevant_train_indices) and relevant_train_indices[local_idx] == global_idx:
                cluster_subset_indices.append(local_idx)
                cluster_subset_ids.append(member_id)
    
    # Only run visualization if not skipping plots
    if not skip_plots:
        # Prepare data for plotting
        relevant_train_embeds = train_embeddings[relevant_train_indices].cpu().numpy()
        gen_embeds_np = gen_embeddings.cpu().numpy()
        
        X_combined = np.concatenate([relevant_train_embeds, gen_embeds_np], axis=0)
        train_offset = len(relevant_train_embeds)
        n_samples_total = X_combined.shape[0]
        
        # Run dimensionality reduction
        print(f"  Running dimensionality reduction on {n_samples_total} points...")
        
        # t-SNE
        perp_val = min(30, n_samples_total - 1) if n_samples_total > 1 else 1
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp_val, n_iter=1000)
        X_tsne = tsne.fit_transform(X_combined)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_combined)
        pca_var = pca.explained_variance_ratio_.sum() * 100
        
        # UMAP
        n_neigh = min(15, n_samples_total - 1) if n_samples_total > 1 else 1
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neigh, min_dist=0.1)
        X_umap = reducer.fit_transform(X_combined)
        
        # Create combined plot
        output_path = os.path.join(subdir_path, "analysis_combined.png")
        plot_combined_analysis(
            X_tsne, X_pca, X_umap, subdir_name,
            metrics, train_offset, best_gen_indices,
            target_idx=target_subset_idx, 
            cluster_indices=cluster_subset_indices,
            cluster_ids=cluster_subset_ids,
            target_id=target_id, output_path=output_path, pca_var=pca_var
        )
    else:
        print(f"  Skipping plot generation...")
    
    # Save JSON results
    results_data = {
        "experiment_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subdir": subdir_name,
            "target_id": target_id,
            "prompt": prompt,
            "num_generations": len(generated_files),
            "cluster_members": cluster_member_ids
        },
        "metrics": {
            "global_avg_similarity": float(global_avg_sim),
            "avg_nearest_neighbor_similarity": float(global_avg_max_sim),
            "avg_target_similarity": float(avg_target_sim),
            "avg_nn_ratio": float(avg_nn_ratio),
            "avg_cluster_log_rge": float(avg_cluster_log_rge),
            "max_target_similarity": float(max_target_sim),
            "avg_prompt_adherence": float(avg_prompt_adherence),
            "intra_list_diversity": float(intra_list_diversity),
            "total_variance": float(total_variance)
        },
        "generated_files": generated_files
    }
    
    json_path = os.path.join(subdir_path, "analysis_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"  JSON saved to: {json_path}")
    
    return metrics


def parse_cluster_range(range_str):
    """
    Parse cluster range string.
    
    Args:
        range_str: String like "0-10" or "5" or None
    
    Returns:
        set of cluster numbers (as strings), or None if no filtering
    """
    if range_str is None:
        return None
    
    range_str = range_str.strip()
    
    # Check for range format "0-10"
    if '-' in range_str:
        parts = range_str.split('-')
        if len(parts) == 2:
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                return set(str(i) for i in range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid cluster range '{range_str}', ignoring filter")
                return None
    else:
        # Single cluster
        try:
            cluster_id = int(range_str.strip())
            return {str(cluster_id)}
        except ValueError:
            print(f"Warning: Invalid cluster ID '{range_str}', ignoring filter")
            return None
    
    return None


def find_all_leaf_dirs_with_wav(base_dir):
    """Recursively find all directories that contain wav files."""
    leaf_dirs = []
    
    for root, dirs, files in os.walk(base_dir):
        wav_files = [f for f in files if f.endswith('.wav')]
        if wav_files:
            leaf_dirs.append(root)
    
    return sorted(leaf_dirs)


def detect_nested_structure(input_dir, leaf_dirs, cluster_filter_set=None):
    """
    Detect if there's a nested structure (config -> cluster folders).
    
    Args:
        input_dir: Base directory
        leaf_dirs: List of leaf directories with wav files
        cluster_filter_set: Optional set of cluster IDs (as strings) to filter
    
    Returns:
        dict or None: If nested structure detected, returns:
            {
                'config_name': {
                    'path': '/path/to/config',
                    'clusters': ['cluster_folder_name', ...]
                }
            }
        Returns None if flat structure (no nesting).
    """
    # Get depth of each leaf dir relative to input_dir
    depths = {}
    for leaf in leaf_dirs:
        rel_path = os.path.relpath(leaf, input_dir)
        parts = rel_path.split(os.sep)
        depth = len(parts)
        if depth not in depths:
            depths[depth] = []
        depths[depth].append(leaf)
    
    # If all leaves are at depth 2, we have a nested structure (config/cluster)
    if len(depths) == 1 and 2 in depths:
        # Group by parent (config folder)
        configs = {}
        for leaf in depths[2]:
            rel_path = os.path.relpath(leaf, input_dir)
            parts = rel_path.split(os.sep)
            config_name = parts[0]
            cluster_name = parts[1]
            
            # Apply cluster filter if specified
            if cluster_filter_set is not None:
                # Try to extract cluster number from folder name
                # Folder might be just "0", "1", etc. or "cluster_0", etc.
                cluster_num = None
                if cluster_name.isdigit():
                    cluster_num = cluster_name
                elif cluster_name.startswith('cluster_'):
                    try:
                        cluster_num = cluster_name.split('_')[1]
                    except IndexError:
                        pass
                
                # Skip if cluster doesn't match filter
                if cluster_num is None or cluster_num not in cluster_filter_set:
                    continue
            
            if config_name not in configs:
                configs[config_name] = {
                    'path': os.path.join(input_dir, config_name),
                    'clusters': []
                }
            configs[config_name]['clusters'].append(cluster_name)
        
        # Verify all configs have at least 1 cluster
        if configs and all(len(c['clusters']) >= 1 for c in configs.values()):
            print(f"\nDetected nested structure with {len(configs)} configurations:")
            for config_name, config_data in configs.items():
                print(f"  - {config_name}: {len(config_data['clusters'])} clusters")
            return configs
    
    return None


def compute_aggregated_metrics(results_by_cluster):
    """
    Compute mean and std metrics across all clusters for a configuration.
    
    Args:
        results_by_cluster: dict of cluster_name -> metrics dict
    
    Returns:
        dict with aggregated statistics
    """
    if not results_by_cluster:
        return None
    
    # Collect all metric values
    metric_keys = [
        'global_avg_sim', 'global_avg_max_sim', 'avg_target_sim', 'avg_nn_ratio', 'avg_cluster_log_rge',
        'max_target_sim', 'avg_prompt_adherence', 'intra_list_diversity', 
        'total_variance'
    ]
    
    aggregated = {
        'num_clusters': len(results_by_cluster),
        'clusters_processed': list(results_by_cluster.keys()),
        'mean_metrics': {},
        'std_metrics': {},
        'min_metrics': {},
        'max_metrics': {},
        'per_cluster_metrics': results_by_cluster
    }
    
    for key in metric_keys:
        values = [m[key] for m in results_by_cluster.values() if key in m]
        if values:
            aggregated['mean_metrics'][key] = float(np.mean(values))
            aggregated['std_metrics'][key] = float(np.std(values))
            aggregated['min_metrics'][key] = float(np.min(values))
            aggregated['max_metrics'][key] = float(np.max(values))
    
    return aggregated


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Evaluate generated audio in subdirectories with visualization'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Base directory containing generated audio subdirectories'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=DEVICE,
        help=f'Device to use (default: {DEVICE})'
    )
    parser.add_argument(
        '--viz-scale',
        type=float,
        default=1.0,
        help='Scale for number of training embeddings to visualize (0-1). '
             '1.0 = 512 embeddings, 0.5 = 256, etc. (default: 1.0)'
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip generating visualization plots (faster processing)'
    )
    parser.add_argument(
        '--cluster-range',
        type=str,
        default=None,
        help='Range of clusters to analyze (e.g., "0-10" for clusters 0 to 10, or "5" for cluster 5 only)'
    )
    parser.add_argument(
        '--clap-model',
        type=str,
        default='clap-laion-music-base',
        choices=['clap-laion-audio', 'clap-laion-music', 'clap-laion-music-base'],
        help='CLAP model to use: clap-laion-audio (general audio, HTSAT-tiny), clap-laion-music (music, HTSAT-tiny), or clap-laion-music-base (music, HTSAT-base) (default: clap-laion-music-base)'
    )
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    device = args.device
    viz_scale = max(0.0, min(1.0, args.viz_scale))  # Clamp to 0-1
    cluster_filter_set = parse_cluster_range(args.cluster_range)
    clap_model = args.clap_model
    
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory")
        return
    
    print(f"Using device: {device}")
    print(f"Input directory: {input_dir}")
    print(f"Visualization scale: {viz_scale:.2f} ({int(viz_scale * MAX_TRAIN_EMBEDDINGS)} max training embeddings)")
    print(f"CLAP model: {clap_model}")
    if args.skip_plots:
        print("Plot generation: DISABLED")
    if cluster_filter_set is not None:
        cluster_list = sorted([int(c) for c in cluster_filter_set])
        print(f"Cluster filter: {min(cluster_list)}-{max(cluster_list)} ({len(cluster_list)} clusters)")
    
    # --- LOAD TRAINING DATA (conditioning info) ---
    print("Loading training data...")
    with open(EMBEDDINGS_FILE, 'r') as f:
        train_data = json.load(f)
    
    train_ids = sorted(list(train_data.keys()))
    
    # NOTE: Embeddings will be loaded/extracted AFTER CLAP model is loaded
    # to ensure we use the correct model for embedding extraction
    train_embeddings = None
    train_embeddings_norm = None
    
    # --- LOAD CLUSTER DATA ---
    clusters = load_clusters_json(CLUSTERS_FILE)
    rep_to_cluster = load_cluster_representatives(CLUSTER_REPRESENTATIVES_FILE)
    print(f"Loaded {len(clusters)} clusters with {sum(len(v) for v in clusters.values())} total members")
    print(f"Loaded {len(rep_to_cluster)} cluster representatives")
    
    # --- LOAD CLAP MODEL ---
    print(f"Loading CLAP model: {clap_model}...")
    # Clear GPU cache before loading new model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Map fadtk model names to laion_clap parameters
    if clap_model == 'clap-laion-music':
        # Music model: HTSAT-tiny with music_audioset checkpoint (from laion_clap defaults)
        CLAP = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny", device=device)
        CLAP.load_ckpt(model_id=1)  # model_id=1 loads music_audioset checkpoint
        print("  -> Loaded CLAP music model (HTSAT-tiny, music_audioset checkpoint)")
    elif clap_model == 'clap-laion-audio':
        # Audio model: HTSAT-tiny with LAION-Audio-630K checkpoint
        CLAP = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, device=device)
        CLAP.load_ckpt()  # model_id=0 loads general audio checkpoint
        print("  -> Loaded CLAP audio model (HTSAT-tiny, LAION-Audio-630K checkpoint)")
    elif clap_model == 'clap-laion-music-base':
        # Music model: HTSAT-base with custom music_audioset checkpoint
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), 
            'model_cache/clap_checkpoints/music_audioset_epoch_15_esc_90.14.pt'
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"CLAP checkpoint not found at {checkpoint_path}")
        CLAP = amg_generation.laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
        CLAP.load_ckpt(checkpoint_path)  # Load custom checkpoint
        print(f"  -> Loaded CLAP music model (HTSAT-base, custom checkpoint from {checkpoint_path})")
    else:
        raise ValueError(f"Unknown CLAP model: {clap_model}")
    
    # Set model to eval mode and disable gradients
    CLAP.eval()
    for param in CLAP.parameters():
        param.requires_grad = False

    # --- LOAD/EXTRACT TRAINING EMBEDDINGS ---
    # If using clap-laion-audio (same as amg_generation.py), use pre-computed embeddings from JSON
    # Otherwise, extract embeddings on-the-fly from soundDataset using the current CLAP model
    if clap_model == 'clap-laion-audio':
        print("Using pre-computed embeddings from JSON (compatible with amg_generation.py)...")
        train_embeddings = torch.stack([
            torch.tensor(train_data[sound_id]['embedding'], dtype=torch.float32)
            for sound_id in train_ids
        ], dim=0).to(device)
    else:
        print(f"CLAP model '{clap_model}' differs from amg_generation.py - extracting fresh embeddings...")
        extracted_embeddings = extract_dataset_embeddings(CLAP, train_data, device, DATASET_DIR)
        train_embeddings = torch.stack([
            extracted_embeddings[sound_id] if isinstance(extracted_embeddings[sound_id], torch.Tensor) 
            else torch.tensor(extracted_embeddings[sound_id], dtype=torch.float32)
            for sound_id in train_ids
        ], dim=0).to(device)
        
        # Also update train_data with extracted embeddings for use in process_subdir
        for sound_id in train_ids:
            emb = extracted_embeddings[sound_id]
            if isinstance(emb, torch.Tensor):
                train_data[sound_id]['embedding'] = emb.cpu().tolist()
            else:
                train_data[sound_id]['embedding'] = emb
    
    train_embeddings_norm = F.normalize(train_embeddings, p=2, dim=1)
    print(f"Train embeddings shape: {train_embeddings.shape}")

    # --- FIND ALL DIRECTORIES WITH WAV FILES ---
    all_leaf_dirs = find_all_leaf_dirs_with_wav(input_dir)
    
    # --- FILTER BY CLUSTER RANGE IF SPECIFIED ---
    if cluster_filter_set is not None:
        leaf_dirs = []
        for leaf in all_leaf_dirs:
            rel_path = os.path.relpath(leaf, input_dir)
            parts = rel_path.split(os.sep)
            # Check if this is a cluster directory
            if len(parts) >= 2:
                cluster_name = parts[-1]  # Last part is cluster folder
                cluster_num = None
                if cluster_name.isdigit():
                    cluster_num = cluster_name
                elif cluster_name.startswith('cluster_'):
                    try:
                        cluster_num = cluster_name.split('_')[1]
                    except IndexError:
                        pass
                
                if cluster_num in cluster_filter_set:
                    leaf_dirs.append(leaf)
            else:
                # Not a nested structure, include all
                leaf_dirs.append(leaf)
        print(f"Found {len(all_leaf_dirs)} total directories, processing {len(leaf_dirs)} after cluster filter")
    else:
        leaf_dirs = all_leaf_dirs
        print(f"Found {len(leaf_dirs)} directories with wav files to process")
    
    # --- DETECT NESTED STRUCTURE ---
    nested_structure = detect_nested_structure(input_dir, leaf_dirs, cluster_filter_set)
    
    all_results = {}
    config_results = {}  # For aggregated results per configuration
    
    for subdir_path in leaf_dirs:
        # Get relative path from input_dir for display name
        rel_path = os.path.relpath(subdir_path, input_dir)
        subdir_name = rel_path.replace(os.sep, '/')
        
        metrics = process_subdir(
            subdir_path, subdir_name, train_data, train_ids,
            train_embeddings, train_embeddings_norm, CLAP, device,
            clusters=clusters, rep_to_cluster=rep_to_cluster, 
            viz_scale=viz_scale, skip_plots=args.skip_plots
        )
        
        if metrics:
            all_results[subdir_name] = metrics
            
            # If nested structure, also collect by config
            if nested_structure:
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    config_name = parts[0]
                    cluster_name = parts[1]
                    
                    if config_name not in config_results:
                        config_results[config_name] = {}
                    config_results[config_name][cluster_name] = metrics
    
    # --- PRINT SUMMARY ---
    print("\n" + "="*100)
    print("                                    SUMMARY RESULTS                                    ")
    print("="*100)
    print(f"{'Subdirectory':<50} {'Glob Sim':>10} {'Tgt Sim':>10} {'NN Ratio':>10} {'Log-RGE':>12} {'Prompt Adh':>12} {'Diversity':>10}")
    print("-"*100)
    
    for subdir, metrics in sorted(all_results.items()):
        display_name = subdir if len(subdir) <= 48 else '...' + subdir[-45:]
        print(f"{display_name:<50} {metrics['global_avg_sim']:>10.4f} {metrics['avg_target_sim']:>10.4f} "
              f"{metrics['avg_nn_ratio']:>10.4f} {metrics['avg_cluster_log_rge']:>12.4f} "
              f"{metrics['avg_prompt_adherence']:>12.4f} {metrics['intra_list_diversity']:>10.4f}")
    
    print("="*100)
    
    # Calculate overall means across all subdirectories
    overall_means = {}
    if all_results:
        metric_keys = ['global_avg_sim', 'global_avg_max_sim', 'avg_target_sim', 
                   'max_target_sim', 'avg_prompt_adherence', 'intra_list_diversity', 
                   'total_variance', 'avg_nn_ratio', 'avg_cluster_log_rge']
        for key in metric_keys:
            values = [m[key] for m in all_results.values() if key in m]
            if values:
                overall_means[key] = float(np.mean(values))
                overall_means[f'{key}_std'] = float(np.std(values))
    
    # Print overall means
    if overall_means:
        print(f"\nOVERALL MEANS (across {len(all_results)} subdirectories):")
        print(f"  Global Avg Similarity:     {overall_means.get('global_avg_sim', 0):.4f} ± {overall_means.get('global_avg_sim_std', 0):.4f}")
        print(f"  Avg NN Similarity:         {overall_means.get('global_avg_max_sim', 0):.4f} ± {overall_means.get('global_avg_max_sim_std', 0):.4f}")
        print(f"  Avg NN Ratio:             {overall_means.get('avg_nn_ratio', 0):.4f} ± {overall_means.get('avg_nn_ratio_std', 0):.4f}")
        print(f"  Avg Log-RGE (cluster):      {overall_means.get('avg_cluster_log_rge', 0):.4f} ± {overall_means.get('avg_cluster_log_rge_std', 0):.4f}")
        print(f"  Avg Target Similarity:     {overall_means.get('avg_target_sim', 0):.4f} ± {overall_means.get('avg_target_sim_std', 0):.4f}")
        print(f"  Avg Prompt Adherence:      {overall_means.get('avg_prompt_adherence', 0):.4f} ± {overall_means.get('avg_prompt_adherence_std', 0):.4f}")
        print(f"  Intra-list Diversity:      {overall_means.get('intra_list_diversity', 0):.4f} ± {overall_means.get('intra_list_diversity_std', 0):.4f}")
    
    # Compute aggregated results by configuration BEFORE saving summary
    aggregated_by_config = {}
    if nested_structure and config_results:
        for config_name, cluster_metrics in config_results.items():
            aggregated = compute_aggregated_metrics(cluster_metrics)
            if aggregated:
                aggregated_by_config[config_name] = {
                    'num_clusters': aggregated['num_clusters'],
                    'mean_metrics': aggregated['mean_metrics'],
                    'std_metrics': aggregated['std_metrics'],
                    'per_cluster_results': cluster_metrics  # Include individual cluster results
                }
    
    # Save summary with overall means AND per-config aggregated results
    summary_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_subdirectories': len(all_results),
        'overall_means': overall_means,
        'per_subdirectory_results': all_results
    }
    
    # Add aggregated by config if available
    if aggregated_by_config:
        summary_data['aggregated_by_config'] = aggregated_by_config
    
    summary_path = os.path.join(input_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"\nSummary saved to: {summary_path}")
    
    # --- AGGREGATED RESULTS PER CONFIGURATION (if nested structure) ---
    if nested_structure and config_results:
        print("\n" + "="*100)
        print("                          AGGREGATED RESULTS BY CONFIGURATION                          ")
        print("="*100)
        
        aggregated_all = {}
        
        for config_name, cluster_metrics in sorted(config_results.items()):
            aggregated = compute_aggregated_metrics(cluster_metrics)
            if aggregated:
                aggregated_all[config_name] = aggregated
                
                mean = aggregated['mean_metrics']
                std = aggregated['std_metrics']
                
                print(f"\n{config_name} ({aggregated['num_clusters']} clusters):")
                print(f"  Global Avg Similarity:     {mean.get('global_avg_sim', 0):.4f} ± {std.get('global_avg_sim', 0):.4f}")
                print(f"  Avg Target Similarity:     {mean.get('avg_target_sim', 0):.4f} ± {std.get('avg_target_sim', 0):.4f}")
                print(f"  Avg NN Ratio:             {mean.get('avg_nn_ratio', 0):.4f} ± {std.get('avg_nn_ratio', 0):.4f}")
                print(f"  Avg Log-RGE (cluster):      {mean.get('avg_cluster_log_rge', 0):.4f} ± {std.get('avg_cluster_log_rge', 0):.4f}")
                print(f"  Max Target Similarity:     {mean.get('max_target_sim', 0):.4f} ± {std.get('max_target_sim', 0):.4f}")
                print(f"  Avg Prompt Adherence:      {mean.get('avg_prompt_adherence', 0):.4f} ± {std.get('avg_prompt_adherence', 0):.4f}")
                print(f"  Intra-list Diversity:      {mean.get('intra_list_diversity', 0):.4f} ± {std.get('intra_list_diversity', 0):.4f}")
                print(f"  Avg NN Similarity:         {mean.get('global_avg_max_sim', 0):.4f} ± {std.get('global_avg_max_sim', 0):.4f}")
                
                # Save per-config aggregated results
                config_summary_path = os.path.join(input_dir, config_name, "aggregated_results.json")
                with open(config_summary_path, 'w') as f:
                    json.dump(aggregated, f, indent=4)
                print(f"  Saved to: {config_summary_path}")
        []
        # Save global aggregated summary
        global_aggregated_path = os.path.join(input_dir, "aggregated_by_config.json")
        
        # Create a cleaner summary for easy comparison
        comparison_summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_configurations': len(aggregated_all),
            'configurations': {}
        }
        
        for config_name, agg in aggregated_all.items():
            comparison_summary['configurations'][config_name] = {
                'num_clusters': agg['num_clusters'],
                'mean_metrics': agg['mean_metrics'],
                'std_metrics': agg['std_metrics']
            }
        
        with open(global_aggregated_path, 'w') as f:
            json.dump(comparison_summary, f, indent=4)
        print(f"\nGlobal aggregated comparison saved to: {global_aggregated_path}")
        
        # Print comparison table
        print("\n" + "="*120)
        print("                                    CONFIGURATION COMPARISON                                    ")
        print("="*120)
        print(f"{'Configuration':<35} {'Glob Sim':>12} {'Tgt Sim':>12} {'NN Ratio':>12} {'Log-RGE':>12} {'Max Tgt':>12} {'Prompt Adh':>12} {'Diversity':>12} {'NN Sim':>12}")
        print("-"*120)
        
        for config_name, agg in sorted(aggregated_all.items()):
            mean = agg['mean_metrics']
            print(f"{config_name:<35} "
                  f"{mean.get('global_avg_sim', 0):>12.4f} "
                  f"{mean.get('avg_target_sim', 0):>12.4f} "
                  f"{mean.get('avg_nn_ratio', 0):>12.4f} "
                  f"{mean.get('avg_cluster_log_rge', 0):>12.4f} "
                  f"{mean.get('max_target_sim', 0):>12.4f} "
                  f"{mean.get('avg_prompt_adherence', 0):>12.4f} "
                  f"{mean.get('intra_list_diversity', 0):>12.4f} "
                  f"{mean.get('global_avg_max_sim', 0):>12.4f}")
        
        print("="*120)
    
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
