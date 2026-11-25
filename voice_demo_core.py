"""Inferensi cosine dan graph untuk demo; tidak menjalankan training notebook."""
from pathlib import Path
import base64
import json
import tempfile
import threading

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

METHODS = ["MFCC + Cosine", "ECAPA Pretrained", "ECAPA Fine-Tuned"]
EMBEDDING_FILES = {
    METHODS[0]: "embeddings_mfcc.npz",
    METHODS[1]: "embeddings_ecapa_pretrained.npz",
    METHODS[2]: "embeddings_ecapa_finetuned.npz",
}
GALLERY_FILES = dict(zip(METHODS, ["speaker_gallery_mfcc.npz", "speaker_gallery_pretrained.npz", "speaker_gallery_finetuned.npz"]))
INFERENCE_LOCK = threading.RLock()


def unit(vector):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    norm = np.linalg.norm(vector)
    if not len(vector) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Vektor suara tidak valid atau tidak memiliki energi.")
    return vector / norm


def load_bundle(folder):
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    if config.get("export_complete") is False:
        raise ValueError("Export belum selesai. Jalankan sampai bagian Referensi Anggota dan Threshold Khusus Demo.")
    manifest = pd.read_csv(folder / "split_manifest.csv")
    matrices = {}
    for method, name in EMBEDDING_FILES.items():
        with np.load(folder / name, allow_pickle=False) as data:
            paths = data["filepaths"].astype(str).tolist()
            values = np.asarray(data["embeddings"], dtype=float)
        if len(paths) != len(values) or not np.isfinite(values).all():
            raise ValueError(f"Embedding tidak valid: {method}.")
        matrices[method] = {path: unit(vector) for path, vector in zip(paths, values)}
    reference = pd.read_csv(folder / "reference_scores.csv")
    return config, manifest, matrices, reference


def read_audio(data, filename, sample_rate=16000, trim_db=15):
    import librosa
    if not data or len(data) > 25 * 1024 * 1024:
        raise ValueError("Pilih file audio yang tidak kosong, maksimal 25 MB.")
    extension = Path(filename).suffix.lower()
    if extension not in {".wav", ".m4a", ".ogg", ".mp3", ".flac"}:
        raise ValueError("Gunakan WAV, M4A, OGG, MP3, atau FLAC.")
    with tempfile.TemporaryDirectory(prefix="voice-demo-") as temp:
        file = Path(temp) / ("input" + extension)
        file.write_bytes(data)
        audio, original_sr = librosa.load(file, sr=None, mono=False, duration=31)
    if audio.ndim > 1:
        if audio.shape[0] > 2:
            raise ValueError("Gunakan audio mono atau stereo.")
        audio = audio.mean(axis=0)
    if len(audio) > 30 * original_sr:
        raise ValueError("Gunakan cuplikan maksimal 30 detik.")
    audio = audio.astype(np.float32)
    if original_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sample_rate)
    audio = audio.astype(np.float32)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio kosong atau tidak terbaca dengan benar.")
    audio = audio - audio.mean()
    audio, _ = librosa.effects.trim(audio, top_db=trim_db)
    if len(audio) < int(sample_rate * 0.5) or np.mean(audio ** 2) < 1e-12:
        raise ValueError("Suara terlalu singkat atau senyap. Gunakan ujaran jelas minimal 0,5 detik.")
    return audio


def load_encoder(folder, fine_tuned=False):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    folder = Path(folder).resolve()
    local = folder / "pretrained_ecapa"
    if not (local / "hyperparams.yaml").is_file():
        raise FileNotFoundError("Checkpoint pretrained belum lengkap. Jalankan kembali sel Export Model terbaru.")
    model = EncoderClassifier.from_hparams(
        source=str(local), savedir=str(folder / ".model_cache"),
        overrides={"pretrained_path": str(local)}, run_opts={"device": "cpu"})
    if fine_tuned:
        model.mods.load_state_dict(torch.load(folder / "ecapa_finetuned.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model


def feature(audio, method, config, encoder=None):
    if method == METHODS[0]:
        import librosa
        mfcc = librosa.feature.mfcc(y=audio, sr=config["sample_rate"], n_mfcc=config["n_mfcc"])
        transform = config.get("mfcc_transform")
        if config.get("bundle_format", 2) >= 3 and not transform:
            raise ValueError("Paket MFCC baru membutuhkan scaler; jalankan kembali export terbaru.")
        if transform and transform.get("drop_c0"):
            mfcc = mfcc[1:]
        vector = np.r_[mfcc.mean(axis=1), mfcc.std(axis=1)].astype(float)
        if transform:
            mean, scale = np.asarray(transform.get("mean", []), dtype=float), np.asarray(transform.get("scale", []), dtype=float)
            if (mean.shape != vector.shape or scale.shape != vector.shape or
                    not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
                raise ValueError("Scaler MFCC tidak valid; gunakan paket export terbaru yang lengkap.")
            vector = (vector - mean) / scale
        return unit(vector)
    if method not in METHODS or encoder is None:
        raise ValueError("Pilih metode yang tersedia dan muat encoder terlebih dahulu.")
    import torch
    with INFERENCE_LOCK, torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        return unit(encoder.encode_batch(tensor).reshape(-1).cpu().numpy())


def load_gallery(folder, method, config):
    with np.load(Path(folder) / GALLERY_FILES[method], allow_pickle=False) as data:
        ids, matrix = data["speaker_ids"].astype(str).tolist(), data["embeddings"]
    names = config["speaker_identification_demo"]["display_names"]
    if matrix.ndim != 2 or len(ids) != len(matrix) or len(set(ids)) != len(ids) or set(ids) != set(names):
        raise ValueError("Referensi speaker tidak cocok dengan konfigurasi export.")
    return {speaker: unit(vector) for speaker, vector in zip(ids, matrix)}


def rank_speakers(vector, gallery, threshold):
    if not gallery or not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Referensi atau threshold speaker tidak valid.")
    query = unit(vector)
    rows = sorted([(speaker, float(np.clip(query @ unit(template), -1, 1)))
                   for speaker, template in gallery.items()], key=lambda item: (-item[1], item[0]))
    ranking = pd.DataFrame(rows, columns=["Speaker", "Cosine"])
    ranking.insert(0, "Peringkat", range(1, len(rows) + 1))
    ranking["Cosine × 100 (%)"] = ranking.Cosine * 100
    ranking["Lolos threshold"] = ranking.Cosine >= threshold
    return rows[0][0] if rows[0][1] >= threshold else None, ranking


def decode_live_packet(packet, target_sr=16000, window_seconds=5):
    """Decode jendela PCM browser; batas ukuran diperiksa sebelum alokasi audio."""
    import librosa
    if not isinstance(packet, dict) or packet.get("active") is not True:
        raise ValueError("Mikrofon belum aktif.")
    sr, encoded = packet.get("sample_rate"), packet.get("pcm")
    if not isinstance(sr, int) or isinstance(sr, bool) or not 8000 <= sr <= 192000:
        raise ValueError("Sampling rate mikrofon tidak didukung.")
    if not isinstance(encoded, str) or not 0 < len(encoded) <= int(192000 * (window_seconds + .5) * 2 * 4 / 3 + 8):
        raise ValueError("Paket audio kosong atau terlalu besar.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("Paket audio mikrofon tidak terbaca.") from error
    if len(raw) % 2 or not 0 < len(raw) <= int(sr * (window_seconds + .5)) * 2:
        raise ValueError("Panjang audio mikrofon tidak valid.")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(audio[-int(target_sr * window_seconds):], dtype=np.float32)


def prepare_live_audio(audio, config):
    import librosa
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio mikrofon tidak valid.")
    audio = audio - audio.mean()
    audio, _ = librosa.effects.trim(audio, top_db=config["trim_top_db"])
    if len(audio) < int(.25 * config["sample_rate"]) or np.mean(audio ** 2) < 1e-12:
        raise ValueError("Belum ada ujaran yang cukup jelas.")
    return np.ascontiguousarray(audio, dtype=np.float32)


def contains_speech(audio, vad_model, config):
    import torch
    from silero_vad import get_speech_timestamps
    settings = config["speaker_identification_demo"]["speech_gate"]
    recent = np.ascontiguousarray(audio[-config["sample_rate"]:], dtype=np.float32)
    with INFERENCE_LOCK:
        return bool(get_speech_timestamps(torch.from_numpy(recent), vad_model,
            sampling_rate=config["sample_rate"], threshold=settings["threshold"],
            min_speech_duration_ms=settings["min_speech_ms"], min_silence_duration_ms=100))


def noisy(audio, snr_db, seed=42):
    power = float(np.mean(np.asarray(audio, dtype=float) ** 2))
    if power <= 1e-12 or not np.isfinite(power):
        raise ValueError("Audio senyap tidak dapat diberi noise terkontrol.")
    noise = np.random.default_rng(seed).normal(size=len(audio))
    noise *= np.sqrt(power / (10 ** (snr_db / 10) * np.mean(noise ** 2)))
    return (audio + noise).astype(np.float32)


def build_graph(vectors, manifest, k=3):
    paths = list(vectors)
    if len(paths) < 2 or not 1 <= k < len(paths):
        raise ValueError("Koleksi minimal dua rekaman dan k harus lebih kecil dari jumlah node.")
    matrix = np.stack([unit(vectors[path]) for path in paths])
    similarities = np.clip(matrix @ matrix.T, -1, 1)
    graph = nx.Graph(name="k-NN utama")
    metadata = manifest.set_index("filepath").to_dict("index")
    ids = {path: f"U{i:02d}" for i, path in enumerate(sorted(p for p in paths if p not in {"QA", "QB"}), 1)}
    for path in paths:
        row = metadata.get(path, {})
        node_id = path if path in {"QA", "QB"} else ids[path]
        speaker = str(row.get("speaker", "Audio A" if path == "QA" else "Audio B"))
        label = "/".join(Path(path).parts[-3:]) if path not in {"QA", "QB"} else speaker
        graph.add_node(path, node_id=node_id, speaker=speaker, label=label)
    complete = nx.Graph()
    complete.add_nodes_from(graph.nodes(data=True))
    for i, a in enumerate(paths):
        for j in range(i+1, len(paths)):
            b = paths[j]
            similarity = float(similarities[i, j])
            complete.add_edge(a, b, weight=1-similarity, similarity=similarity,
                              layout_weight=max(0.001, (similarity+1)/2), mst_bridge=False)
        neighbors = sorted((j for j in range(len(paths)) if j != i), key=lambda j: (-similarities[i, j], paths[j]))[:k]
        for j in neighbors:
            b = paths[j]
            similarity = float(similarities[i, j])
            graph.add_edge(a, b, weight=1-similarity, similarity=similarity,
                           layout_weight=max(0.001, (similarity+1)/2), mst_bridge=False)
    with_mst = graph.copy()
    with_mst.name = "k-NN + MST (pembanding)"
    for a, b, attrs in nx.minimum_spanning_edges(complete, data=True):
        if not with_mst.has_edge(a, b):
            with_mst.add_edge(a, b, **dict(attrs, mst_bridge=True))
    return graph, with_mst


def path_result(graph, source, target):
    if source not in graph or target not in graph:
        raise ValueError("Pilih source dan target dari koleksi.")
    try:
        distance, path = nx.single_source_dijkstra(graph, source, target, weight="weight")
    except nx.NetworkXNoPath:
        return {"path": [], "distance": None, "hops": None, "mst_edges": None}
    return {"path": path, "distance": float(distance), "hops": len(path)-1,
            "mst_edges": sum(graph[a][b]["mst_bridge"] for a, b in zip(path, path[1:]))}


def path_table(graph, path):
    rows, total = [], 0.0
    for i, (a, b) in enumerate(zip(path, path[1:]), 1):
        edge = graph[a][b]
        total += edge["weight"]
        rows.append({"Langkah": i, "Dari": graph.nodes[a]["node_id"], "Ke": graph.nodes[b]["node_id"],
                     "Cosine": edge["similarity"], "Jarak": edge["weight"], "Total kumulatif": total,
                     "Edge": "MST tambahan" if edge["mst_bridge"] else "k-NN"})
    return pd.DataFrame(rows)


def draw_path(graph, source, target, result):
    visible = {source, target, *result["path"]}
    visible.update(graph.neighbors(source))
    visible.update(graph.neighbors(target))
    view = graph.subgraph(visible)
    positions = nx.spring_layout(view, seed=42, weight="layout_weight")
    speakers = sorted({graph.nodes[node]["speaker"] for node in graph})
    palette = {speaker: plt.cm.Set2(i/max(1, len(speakers)-1)) for i, speaker in enumerate(speakers)}
    fig, ax = plt.subplots(figsize=(9, 6))
    for bridge, color, style in [(False, "#bec9ce", "solid"), (True, "#b16a12", "dashed")]:
        edges = [(a, b) for a, b, data in view.edges(data=True) if data["mst_bridge"] == bridge]
        nx.draw_networkx_edges(view, positions, edgelist=edges, edge_color=color, style=style, alpha=.7, ax=ax)
    nodes = list(view)
    nx.draw_networkx_nodes(view, positions, nodelist=nodes,
                           node_color=[palette[graph.nodes[node]["speaker"]] for node in nodes], node_size=500, ax=ax)
    edges = list(zip(result["path"], result["path"][1:]))
    for bridge, style in [(False, "solid"), (True, "dashed")]:
        selected = [(a, b) for a, b in edges if graph[a][b]["mst_bridge"] == bridge]
        nx.draw_networkx_edges(view, positions, edgelist=selected, edge_color="#bf2546", width=3.5, style=style, ax=ax)
    nx.draw_networkx_edge_labels(view, positions, edge_labels={(a,b): f"{graph[a][b]['weight']:.3f}" for a,b in edges}, font_size=8, ax=ax)
    for node, shape in [(source, "s"), (target, "D")]:
        nx.draw_networkx_nodes(view, positions, nodelist=[node], node_shape=shape, node_size=800,
                               node_color=[palette[graph.nodes[node]["speaker"]]], edgecolors="#162d3a", linewidths=2, ax=ax)
    nx.draw_networkx_labels(view, positions, labels={node: graph.nodes[node]["node_id"]+"\n"+graph.nodes[node]["speaker"] for node in nodes}, font_size=8, ax=ax)
    ax.set_title(graph.name)
    if not result["path"]:
        ax.text(.5, .02, "Source dan target berada pada komponen terpisah", ha="center", transform=ax.transAxes, color="#a6203f")
    ax.margins(.25)
    ax.axis("off")
    fig.tight_layout()
    return fig


def distribution_plot(reference, method, score, threshold):
    subset = reference[reference.model == method]
    fig, ax = plt.subplots(figsize=(9, 4))
    values = subset.score.to_numpy()
    lo, hi = min(float(values.min()), score, threshold), max(float(values.max()), score, threshold)
    pad = max(.01, (hi-lo)*.08)
    bins = np.linspace(max(-1, lo-pad), min(1, hi+pad), 18)
    for label, title, color in [(0, "Impostor / speaker berbeda", "#d58b3a"), (1, "Genuine / speaker sama", "#126f80")]:
        ax.hist(subset.loc[subset.label == label, "score"], bins=bins, alpha=.65, color=color, label=title)
    ax.axvline(score, color="#bf2546", linewidth=2, label="Pasangan input")
    ax.axvline(threshold, color="#162d3a", linestyle="--", label="Threshold validation")
    ax.set(xlabel="Cosine similarity", ylabel="Jumlah pasangan referensi")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
