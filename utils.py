import numpy as np
import torch
import torch.nn.functional as F
def split_audio(wav: np.ndarray, sr: int, window_sec: float) -> np.ndarray:
    """
    Split a 1D audio signal into windows of window_sec seconds.
    Zero-pad the last window if shorter.

    Args:
        wav: numpy array of shape (n_samples,)
        sr: sample rate (samples per second)
        window_sec: length of each window in seconds

    Returns:
        windows: numpy array of shape (n_windows, window_samples)
    """
    window_len = int(window_sec * sr)
    n_samples = wav.shape[1]
    # compute number of windows
    n_windows = int(np.ceil(n_samples / window_len))
    padded_len = n_windows * window_len
    # pad wav with zeros
    pad_width = padded_len - n_samples
    wav_padded = np.pad(wav, ((0, 0), (0, pad_width)), mode='constant')
    # reshape
    windows = wav_padded.reshape(n_windows, window_len)
    return windows


def compute_embeddings(windows: np.ndarray, loss_fn: None, device: torch.device, feature: str = 'acoustic') -> torch.Tensor:
    """
    Compute embeddings for each window of audio.

    Args:
        windows: numpy array of shape (n_windows, window_len)
        loss_fn: CDPAM instance with get_embedding method
        device: torch device

    Returns:
        embeddings: torch.Tensor of shape (n_windows, embedding_dim)
    """
    embeddings = []
    for win in windows:
        # convert to torch tensor, add batch dimension if needed
        wav_tensor = torch.from_numpy(win).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = loss_fn.get_embedding(wav_tensor, feature)
        embeddings.append(emb.squeeze(0).cpu())
    return torch.stack(embeddings, dim=0)


def compare_embeddings(ref_emb: torch.Tensor, test_emb: torch.Tensor, metric: str = 'cosine', agg: str = 'mean') -> torch.Tensor:
    """
    For each test embedding, compute similarity (or distance) to all reference embeddings
    and return the maximum similarity per test slice.

    Args:
        ref_emb: tensor of shape (n_ref, d)
        test_emb: tensor of shape (n_test, d)
        metric: 'cosine' or 'euclid'

    Returns:
        max_scores: tensor of shape (n_test,)
    """
    if metric == 'cosine':
        # normalize
        ref_norm = F.normalize(ref_emb, dim=1)
        test_norm = F.normalize(test_emb, dim=1)
        # compute cosine similarities matrix: (n_test, n_ref)
        sims = torch.matmul(test_norm, ref_norm.T)
        # for each test, get max similarity
        # max_scores, _ = sims.max(dim=1)
    elif metric == 'euclid':
        # compute pairwise distances
        # expand dims
        a = test_emb.unsqueeze(1)  # (n_test,1,d)
        b = ref_emb.unsqueeze(0)   # (1,n_ref,d)
        dists = torch.norm(a - b, dim=2)  # (n_test, n_ref)
        # convert to similarity as negative distance or use min distance
        # here we take negative distance
        # neg_dists = -dists
        # max_scores, _ = dists.min(dim=1)
    else:
        raise ValueError(f"Unknown metric {metric}, choose 'cosine' or 'euclid'.")
    
    if agg == 'max':
        scores, _ = sims.max(dim=1)
    elif agg == 'mean':
        scores = sims.mean(dim=1)
    elif agg == 'none':
        scores = sims
    else:
        raise ValueError(f"Unknown aggregation {agg}")
    return scores