"""Copie test_cube_2 -> test_cube_2_480 et convertit observation.images.wrist de 1280x720 en 640x480
(recadrage central 4:3 puis réduction, comme le fait la webcam en mode 640x480).
Recalcule les stats wrist (par épisode + globales) et aligne la feature sur test_cube_3 pour permettre la fusion."""
import json
import shutil
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import (
    DEFAULT_QUANTILES,
    aggregate_feature_stats,
    auto_downsample_height_width,
    get_feature_stats,
)
from lerobot.datasets.video_utils import _get_codec_options

ROOT = Path.home() / ".cache/huggingface/lerobot/cemrenurkeles"
SRC, DST, REF = ROOT / "test_cube_2", ROOT / "test_cube_2_480", ROOT / "test_cube_3"
KEY = "observation.images.wrist"
OUT_W, OUT_H, FPS = 640, 480, 30

if DST.exists():
    raise SystemExit(f"{DST} existe déjà, rien n'est modifié.")
shutil.copytree(SRC, DST)
print(f"copie {SRC.name} -> {DST.name}")

ep_files = sorted((DST / "meta/episodes").glob("*/*.parquet"))
episodes = pd.concat(pd.read_parquet(f) for f in ep_files).sort_values("episode_index")


def transform(rgb):
    h, w = rgb.shape[:2]
    crop_w = h * 4 // 3
    x0 = (w - crop_w) // 2
    return cv2.resize(rgb[:, x0 : x0 + crop_w], (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


ep_stats = {}
for video in sorted((DST / "videos" / KEY).glob("*/*.mp4")):
    chunk_idx = int(video.parent.name.split("-")[1])
    file_idx = int(video.stem.split("-")[1])
    eps = episodes[
        (episodes[f"videos/{KEY}/chunk_index"] == chunk_idx) & (episodes[f"videos/{KEY}/file_index"] == file_idx)
    ]
    buffers = {int(e.episode_index): [] for e in eps.itertuples()}
    bounds = [(int(e.episode_index), e[eps.columns.get_loc(f"videos/{KEY}/from_timestamp") + 1],
               e[eps.columns.get_loc(f"videos/{KEY}/to_timestamp") + 1]) for e in eps.itertuples()]

    tmp = video.with_suffix(".tmp.mp4")
    n = 0
    with av.open(str(video)) as src, av.open(str(tmp), "w") as dst:
        out = dst.add_stream("libsvtav1", FPS, options=_get_codec_options("libsvtav1"))
        out.pix_fmt, out.width, out.height = "yuv420p", OUT_W, OUT_H
        out.time_base = Fraction(1, FPS)
        for frame in src.decode(video=0):
            idx = round(frame.time * FPS)
            assert idx == n, f"{video.name}: trame {n} au temps {frame.time} (non contiguë)"
            img = transform(frame.to_ndarray(format="rgb24"))
            vf = av.VideoFrame.from_ndarray(img, format="rgb24")
            vf.pts, vf.time_base = n, Fraction(1, FPS)
            for pkt in out.encode(vf):
                dst.mux(pkt)
            t = n / FPS
            for ep, t0, t1 in bounds:
                if t0 - 1e-4 <= t < t1 - 1e-4:
                    buffers[ep].append(auto_downsample_height_width(img.transpose(2, 0, 1)))
                    break
            n += 1
        for pkt in out.encode():
            dst.mux(pkt)
    tmp.replace(video)

    for ep, frames in buffers.items():
        expected = int(episodes.loc[episodes.episode_index == ep, "length"].iloc[0])
        assert len(frames) == expected, f"épisode {ep}: {len(frames)} trames au lieu de {expected}"
        st = get_feature_stats(np.stack(frames), axis=(0, 2, 3), keepdims=True, quantile_list=DEFAULT_QUANTILES)
        ep_stats[ep] = {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in st.items()}
    print(f"{video.relative_to(DST)} : {n} trames -> {OUT_W}x{OUT_H}, épisodes {sorted(buffers)}")

assert sorted(ep_stats) == sorted(episodes.episode_index), "stats manquantes pour certains épisodes"

# stats par épisode dans meta/episodes
for f in ep_files:
    df = pd.read_parquet(f)
    for stat in ep_stats[int(df.episode_index.iloc[0])]:
        col = f"stats/{KEY}/{stat}"
        df[col] = [ep_stats[int(ep)][stat].tolist() for ep in df.episode_index]
    df.to_parquet(f, index=False)

# stats globales
stats_path = DST / "meta/stats.json"
stats = json.loads(stats_path.read_text())
agg = aggregate_feature_stats([ep_stats[ep] for ep in sorted(ep_stats)])
stats[KEY] = {k: np.asarray(v).tolist() for k, v in agg.items()}
stats_path.write_text(json.dumps(stats, indent=4))

# feature identique à test_cube_3
info_path = DST / "meta/info.json"
info = json.loads(info_path.read_text())
info["features"][KEY] = json.loads((REF / "meta/info.json").read_text())["features"][KEY]
info_path.write_text(json.dumps(info, indent=4))
print("stats et info.json mis à jour")
