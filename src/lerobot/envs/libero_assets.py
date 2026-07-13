#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import libero.libero as libero_module
import robosuite
from libero.libero import benchmark, get_libero_path


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except PermissionError:
        return False


_LIBERO_PATH_FALLBACKS = {
    "benchmark_root": ".",
    "bddl_files": "bddl_files",
    "init_states": "init_files",
    "assets": "assets",
}


def get_libero_resource_path(query_key: str) -> Path:
    """Return a valid LIBERO resource path, falling back to the installed package."""
    configured = Path(get_libero_path(query_key))
    if _path_exists(configured):
        return configured

    fallback_rel = _LIBERO_PATH_FALLBACKS.get(query_key)
    if fallback_rel is None:
        return configured

    package_root = Path(benchmark.__file__).resolve().parents[1]
    fallback = package_root / fallback_rel
    if _path_exists(fallback):
        return fallback
    return configured


def validate_libero_assets() -> Path:
    """Fail early with an actionable message when LIBERO assets are incomplete."""
    # `libero.libero.get_assets_path()` attempts a Hugging Face download when
    # package-local assets are absent. Avoid that side effect here; this check
    # should only report what is available locally.
    cache_assets_path = Path.home() / ".cache" / "libero" / "assets"
    package_assets_path = Path(benchmark.__file__).resolve().parents[1] / "assets"
    candidate_paths = [
        *(Path(path) for path in os.environ.get("LIBERO_ASSETS_PATH", "").split(os.pathsep) if path),
        get_libero_resource_path("assets"),
        package_assets_path,
        cache_assets_path,
    ]
    required_rel_paths = [
        Path("turbosquid_objects") / "wine_rack" / "wine_rack.xml",
        Path("stable_scanned_objects") / "akita_black_bowl" / "akita_black_bowl.xml",
        Path("stable_hope_objects") / "cream_cheese" / "cream_cheese.xml",
    ]

    seen: set[Path] = set()
    missing_by_candidate: list[tuple[Path, list[Path]]] = []
    inaccessible_candidates: list[Path] = []
    for assets_path in candidate_paths:
        assets_path = assets_path.expanduser()
        if assets_path in seen:
            continue
        seen.add(assets_path)
        missing = []
        inaccessible = False
        for rel_path in required_rel_paths:
            required_path = assets_path / rel_path
            try:
                exists = required_path.exists()
            except PermissionError:
                inaccessible = True
                break
            if not exists:
                missing.append(required_path)
        if inaccessible:
            inaccessible_candidates.append(assets_path)
            continue
        if not missing:
            libero_module._assets_path_cache = str(assets_path)
            return assets_path
        if _path_exists(assets_path):
            missing_by_candidate.append((assets_path, missing))

    fallback_assets_path = candidate_paths[0] if candidate_paths else cache_assets_path
    assets_path, missing = (
        missing_by_candidate[0]
        if missing_by_candidate
        else (
            fallback_assets_path,
            [fallback_assets_path / rel for rel in required_rel_paths],
        )
    )
    inaccessible_text = ""
    if inaccessible_candidates:
        inaccessible_text = "\nInaccessible LIBERO asset candidates:\n" + "\n".join(
            f"  - {path}" for path in inaccessible_candidates
        )
    missing_text = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(
        "LIBERO assets are missing or incomplete. Missing required files:\n"
        f"{missing_text}\n"
        f"{inaccessible_text}\n"
        "Restore the local LIBERO simulator assets and point LIBERO_ASSETS_PATH to them, for example:\n"
        f"  export LIBERO_ASSETS_PATH={assets_path}\n"
        "Training data remains local and is not affected by this check."
    )


def rewrite_libero_demo_xml_paths(xml_str: str) -> str:
    """Point asset paths in a recorded LIBERO model XML at this installation."""
    assets_path = validate_libero_assets().resolve()
    robosuite_assets_path = Path(robosuite.__file__).resolve().parent / "models" / "assets"
    root = ET.fromstring(xml_str)
    asset_node = root.find("asset")
    if asset_node is None:
        return xml_str

    path_roots = {
        "/chiliocosm/assets/": assets_path,
        "/libero/libero/assets/": assets_path,
        "/robosuite/models/assets/": robosuite_assets_path,
    }
    for element in [*asset_node.findall("mesh"), *asset_node.findall("texture")]:
        old_path = element.get("file")
        if old_path is None:
            continue
        for marker, local_root in path_roots.items():
            if marker in old_path:
                element.set("file", str(local_root / old_path.split(marker, 1)[1]))
                break
    return ET.tostring(root, encoding="unicode")
