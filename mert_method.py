from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.transforms as T
import soundfile as sf
from typing import Optional, Union
import numpy as np
from utils import compare_embeddings

# loading our model weights and processor
model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)

def load_audio(file_path, device: Union[str, torch.device]):
    # read and preprocess audio
    audio, sr = sf.read(file_path, dtype='float32')
    audio = torch.from_numpy(audio)
    # ensure mono
    if audio.ndim > 1:
        audio = audio.mean(dim=1)
    target_sr = processor.sampling_rate
    # resample if needed
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        audio = resampler(audio)
    # normalize and prepare inputs
    inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt")
    # move tensors to device
    dev = torch.device(device) if isinstance(device, str) else device
    return {k: v.to(dev) for k, v in inputs.items()}


def extract_timbre(embeddings):
    # embeddings: [13, T, 768]
    # average layers 1-4, keep time-axis
    timbre = embeddings[1:5].mean(dim=0)   # [T, 768]
    return timbre


def extract_structure(embeddings):
    # layers 9-12 capture global/structural features
    structure = embeddings[9:13].mean(dim=0)  # [T, 768]
    return structure


def pool_time(x: torch.Tensor, out_len: Optional[int]) -> torch.Tensor:
    """
    Reduce temporal resolution by averaging to a fixed number of time bins.

    Args:
        x: Tensor of shape [T, D]
        out_len: Target number of time bins. If None, <=0, or >= T, returns x unchanged.

    Returns:
        Tensor of shape [out_len, D] (or [T, D] if no reduction applied)
    """
    if out_len is None or out_len <= 0 or x.shape[0] <= out_len:
        return x
    # reshape to [N=1, C=D, L=T] for adaptive_avg_pool1d, then back to [T, D]
    x_bcL = x.transpose(0, 1).unsqueeze(0)
    y = F.adaptive_avg_pool1d(x_bcL, out_len)
    return y.squeeze(0).transpose(0, 1)


def get_mert_file_embedding(
    file_path: str,
    feature: str = 'both',
    window_sec: Optional[float] = None,
    default_window_sec: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """Compute a single embedding vector for an audio file using MERT.

    Behavior of window_sec:
    - None: use default windowing (default_window_sec seconds per time bin via pooling), then average across time to one vector.
    - >0: pool to ~duration/window_sec bins, then average across time to one vector.
    - <0: treat whole file as one window, i.e., directly average all temporal embeddings to one vector.

    Returns a 1-D numpy float32 vector of size 1536 ('both') or 768 ('timbre'/'structure').
    """
    # resolve device
    dev = torch.device(device) if isinstance(device, str) else (device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(dev)
    model.eval()

    # Helper: robust audio loader with fallbacks
    def _load_audio_np(path: str, target_sr: int) -> tuple[np.ndarray, int]:
        # Try torchaudio first
        try:
            import torchaudio as ta
            wav, sr0 = ta.load(path)  # [C, T]
            if wav.dim() > 1:
                wav = wav.mean(dim=0)
            wav_np = wav.cpu().numpy()
            if sr0 != target_sr:
                resampler = T.Resample(sr0, target_sr)
                wav_t = resampler(wav)
                wav_np = wav_t.cpu().numpy()
                sr0 = target_sr
            return wav_np.astype('float32'), sr0
        except Exception:
            pass
        # Then librosa (falls back to audioread/ffmpeg if needed)
        try:
            import librosa
            y, _ = librosa.load(path, sr=target_sr, mono=True)
            return y.astype('float32'), target_sr
        except Exception:
            pass
        # Finally, soundfile
        audio_np, sr1 = sf.read(path, dtype='float32', always_2d=False)
        if isinstance(audio_np, np.ndarray) and audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr1 != target_sr:
            resampler = T.Resample(sr1, target_sr)
            audio_t = torch.from_numpy(audio_np)
            audio_t = resampler(audio_t)
            audio_np = audio_t.cpu().numpy().astype('float32')
            sr1 = target_sr
        return audio_np, sr1

    # Load audio to get duration and prepare inputs
    target_sr = processor.sampling_rate
    audio, sr = _load_audio_np(file_path, target_sr)
    # Sanitize and ensure minimum length to satisfy conv feature extractor
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if audio.dtype != np.float32:
        audio = audio.astype('float32')
    # Estimate minimum safe input length using HuBERT first conv kernel
    try:
        first_conv = model.feature_extractor.conv_layers[0].conv
        k0 = int(first_conv.kernel_size[0]) if hasattr(first_conv, 'kernel_size') else 10
    except Exception:
        k0 = 10
    min_samples = max(1024, int(0.1 * target_sr), k0)
    if audio.shape[0] < min_samples:
        pad_width = min_samples - audio.shape[0]
        audio = np.pad(audio, (0, pad_width), mode='constant', constant_values=0.0).astype('float32')
        sr = target_sr
    duration = float(len(audio)) / float(sr) if sr > 0 else 0.0
    inputs = processor(torch.from_numpy(audio), sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    # Chunked forward to avoid OOM; accumulate global mean over time
    max_chunk_sec = 20.0
    max_chunk_samples = max(int(max_chunk_sec * sr), 1)
    total_T = 0
    sum_vec: Optional[torch.Tensor] = None

    use_amp = (dev.type == 'cuda')

    # Ensure a safe minimum per-chunk length (pad tails if shorter)
    min_forward_sec = 1.0
    min_forward_samples = max(int(min_forward_sec * sr), k0)

    for start in range(0, audio.shape[0], max_chunk_samples):
        chunk = audio[start:start + max_chunk_samples]
        if chunk.shape[0] == 0:
            continue
        # Pad too-short tail chunks to meet minimum conv requirement
        if chunk.shape[0] < min_forward_samples:
            pad_width = min_forward_samples - chunk.shape[0]
            chunk = np.pad(chunk, (0, pad_width), mode='constant', constant_values=0.0).astype('float32')
        inputs = processor(torch.from_numpy(chunk), sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(**inputs, output_hidden_states=True)
            else:
                out = model(**inputs, output_hidden_states=True)
        hid = torch.stack(out.hidden_states, dim=0).squeeze(1)  # [13, T, 768]
        # Select feature representation -> [T, D]
        if feature == 'both':
            tim = extract_timbre(hid)
            stru = extract_structure(hid)
            feat = torch.cat((tim, stru), dim=-1)  # [T, 1536]
        elif feature == 'timbre':
            feat = extract_timbre(hid)  # [T, 768]
        elif feature == 'structure':
            feat = extract_structure(hid)  # [T, 768]
        else:
            raise ValueError(f"Unknown MERT feature: {feature}")

        T_i = int(feat.shape[0])
        if T_i == 0:
            continue
        # accumulate sum over time to compute global mean
        s = feat.sum(dim=0)
        if sum_vec is None:
            sum_vec = s.detach()
        else:
            sum_vec = sum_vec + s.detach()
        total_T += T_i

        # free chunk tensors
        del out, hid, feat, s
        torch.cuda.empty_cache() if dev.type == 'cuda' else None

    if total_T == 0 or sum_vec is None:
        # fallback zero vector
        out_dim = 1536 if feature == 'both' else 768
        return np.zeros((out_dim,), dtype=np.float32)

    vec = (sum_vec / float(total_T)).detach().cpu().numpy().astype(np.float32)
    return vec


def mert_method(ref_path,
                test_path,
                metric,
                agg,
                use_gpu,
                feature: str = 'both',
                num_segments: Optional[int] = None,
                device: Optional[Union[str, torch.device]] = None):
    """
    Compute similarity scores using MERT embeddings.

    Args:
        ref_path: path to reference audio
        test_path: list of paths to test audios
        metric: 'cosine' or 'euclid'
        agg: aggregation method
        use_gpu: whether to use GPU if available (ignored if device is provided)
        feature: 'both' | 'timbre' | 'structure'
        num_segments: optional temporal pooling segments
        device: explicit device (e.g., 'cpu', 'cuda:0')
    """
    # set up device
    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
    else:
        dev = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    model.to(dev)
    model.eval()
    scores = []

    # load and preprocess
    audio_ref = load_audio(ref_path, dev)
    for path in test_path:
        audio_test = load_audio(path, dev)

        # forward pass
        with torch.no_grad():
            out_ref = model(**audio_ref, output_hidden_states=True)
            out_test = model(**audio_test, output_hidden_states=True)

        # stack hidden states: [13, T, 768]
        emb_ref = torch.stack(out_ref.hidden_states, dim=0).squeeze(1)
        emb_test = torch.stack(out_test.hidden_states, dim=0).squeeze(1)

        # select feature representation
        if feature == 'both':
            tim_ref = extract_timbre(emb_ref)
            str_ref = extract_structure(emb_ref)
            tim_test = extract_timbre(emb_test)
            str_test = extract_structure(emb_test)
            # concat along feature axis: [T, 1536]
            ref_emb = torch.cat((tim_ref, str_ref), dim=-1)
            test_emb = torch.cat((tim_test, str_test), dim=-1)
        elif feature == 'timbre':
            ref_emb = extract_timbre(emb_ref)
            test_emb = extract_timbre(emb_test)
        elif feature == 'structure':
            ref_emb = extract_structure(emb_ref)
            test_emb = extract_structure(emb_test)
        else:
            raise ValueError(f"Unknown feature type {feature}, choose 'both', 'timbre', or 'structure'.")

        # Optional temporal downsampling to reduce number of time embeddings
        ref_emb = pool_time(ref_emb, num_segments)
        test_emb = pool_time(test_emb, num_segments)

    # compute similarity or distance
    score = compare_embeddings(ref_emb, test_emb, metric=metric, agg=agg)
    scores.append(score)
    return scores
