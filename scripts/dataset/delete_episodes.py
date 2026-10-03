#!/usr/bin/env python3
"""Supprime des épisodes d'un dataset LeRobot local, sans risque pour l'original, et peut mettre à jour le Hub.

Contrairement à `lerobot-edit-dataset --operation.type delete_episodes` en place :
- le résultat est écrit à côté puis vérifié, l'original n'est remplacé qu'à la fin (sauvegarde horodatée gardée) ;
- les références meta/episodes incohérentes laissées par un merge sont corrigées avant suppression ;
- avec --push, les fichiers devenus obsolètes sur le Hub sont retirés, en ligne == local.

Usage : python scripts/dataset/delete_episodes.py cemrenurkeles/test_cube_all 14 15 16 [--push]
Les numéros d'épisodes commencent à 0.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def fix_meta_refs(root: Path) -> int:
    """Fait pointer chaque ligne de meta/episodes vers le fichier qui la contient réellement."""
    fixed = 0
    for f in sorted(root.glob("meta/episodes/*/*.parquet")):
        chunk, file = int(f.parent.name.split("-")[1]), int(f.stem.split("-")[1])
        df = pd.read_parquet(f)
        bad = (df["meta/episodes/chunk_index"] != chunk) | (df["meta/episodes/file_index"] != file)
        if bad.any():
            df["meta/episodes/chunk_index"], df["meta/episodes/file_index"] = chunk, file
            df.to_parquet(f, index=False)
            fixed += int(bad.sum())
    return fixed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id")
    parser.add_argument("episodes", type=int, nargs="+")
    parser.add_argument("--push", action="store_true", help="mettre à jour le dataset sur Hugging Face")
    args = parser.parse_args()

    root = HF_LEROBOT_HOME / args.repo_id
    tmp = root.with_name(root.name + "_tmp_delete")
    if not (root / "meta/info.json").exists():
        raise SystemExit(f"Dataset introuvable : {root}")
    if tmp.exists():
        raise SystemExit(f"{tmp} existe (essai précédent interrompu ?) : le supprimer avant de relancer.")

    if n := fix_meta_refs(root):
        print(f"{n} référence(s) meta/episodes corrigée(s)")

    src = LeRobotDataset(args.repo_id, root=root, video_backend="pyav")
    to_delete = sorted(set(args.episodes))
    invalid = [e for e in to_delete if not 0 <= e < src.num_episodes]
    if invalid:
        raise SystemExit(f"Épisodes inexistants : {invalid} (le dataset a {src.num_episodes} épisodes, de 0 à {src.num_episodes - 1})")
    kept = [e for e in range(src.num_episodes) if e not in to_delete]
    print(f"{args.repo_id} : {src.num_episodes} épisodes, suppression de {to_delete} -> {len(kept)} épisodes")

    delete_episodes(src, to_delete, output_dir=tmp, repo_id=args.repo_id)
    fix_meta_refs(tmp)

    # Vérification : chaque épisode gardé a la même longueur et les mêmes actions qu'avant
    new = LeRobotDataset(args.repo_id, root=tmp, video_backend="pyav")
    assert new.num_episodes == len(kept), f"{new.num_episodes} épisodes au lieu de {len(kept)}"
    for new_idx, old_idx in enumerate(kept):
        n_len, o_len = int(new.meta.episodes["length"][new_idx]), int(src.meta.episodes["length"][old_idx])
        assert n_len == o_len, f"épisode {new_idx} (ancien {old_idx}) : {n_len} trames au lieu de {o_len}"
        n0, o0 = int(new.meta.episodes["dataset_from_index"][new_idx]), int(src.meta.episodes["dataset_from_index"][old_idx])
        for i in (0, n_len // 2, n_len - 1):
            assert np.allclose(new.hf_dataset[n0 + i]["action"], src.hf_dataset[o0 + i]["action"]), (
                f"épisode {new_idx} (ancien {old_idx}) : actions différentes"
            )
    new[0], new[len(new) - 1]  # décodage vidéo au début et à la fin
    print("vérification OK : longueurs et actions identiques pour tous les épisodes gardés")

    backup = root.with_name(f"{root.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}")
    root.rename(backup)
    tmp.rename(root)
    print(f"remplacé ; ancienne version gardée dans {backup}")
    print("nouvelle numérotation :", ", ".join(f"{o}->{n}" for n, o in enumerate(kept) if o != n) or "inchangée")

    if args.push:
        from huggingface_hub import HfApi

        LeRobotDataset(args.repo_id, root=root, video_backend="pyav").push_to_hub()
        api = HfApi()
        local = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        stale = [f for f in api.list_repo_files(args.repo_id, repo_type="dataset")
                 if f not in local and f not in (".gitattributes", "README.md")]
        for f in stale:
            api.delete_file(f, args.repo_id, repo_type="dataset", commit_message=f"Remove stale {f}")
        print(f"Hub mis à jour ({len(stale)} fichier(s) obsolète(s) retiré(s)) : https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
