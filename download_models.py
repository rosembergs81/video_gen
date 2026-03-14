#!/usr/bin/env python3
"""
download_models.py
──────────────────
Descarga modelos de HuggingFace al caché local.

Modos de uso:
  python download_models.py               # Interactivo: pregunta cuáles descargar
  python download_models.py --all          # Descarga TODOS los modelos (~100 GB)
  python download_models.py --minimal      # Solo CogVideoX-5B T2V (~20 GB)
  python download_models.py --select 1 2   # Descarga modelos por número

Espacio en disco estimado:
  • CogVideoX-5B         ~20 GB (T2V + V2V usan el mismo repo)
  • CogVideoX-5B-I2V     ~20 GB
  • LTX-Video            ~10 GB (T2V + I2V usan el mismo repo)
  ─────────────────────────────
  Total (todos)           ~50 GB en safetensors
  Recomendado disco       ≥ 120 GB (modelos + outputs + caché)
"""

import os
import sys
import argparse
import shutil

# ─── Modelo registry (sin importar torch para no requerir GPU) ────────────────
MODEL_REPOS = {
    1: {
        "repo": "THUDM/CogVideoX-5b",
        "label": "CogVideoX-5B (Texto→Video + Video→Video)",
        "size_est": "~20 GB",
    },
    2: {
        "repo": "THUDM/CogVideoX-5b-I2V",
        "label": "CogVideoX-5B-I2V (Imagen→Video)",
        "size_est": "~20 GB",
    },
    3: {
        "repo": "Lightricks/LTX-Video",
        "label": "LTX-Video (T2V + I2V rápido)",
        "size_est": "~10 GB",
    },
}

# Archivos que NUNCA necesitamos (ahorran ~30-50% de espacio)
IGNORE_PATTERNS = [
    "*.msgpack",
    "*.bin",         # redundante si existen .safetensors
    "*.h5",
    "*.ckpt",
    "*.ot",
    "*.md",          # READMEs del repo
    ".gitattributes",
    "*.png",         # imágenes de ejemplo
    "*.jpg",
    "*.gif",
    "*training*",    # scripts de entrenamiento
    "*train*",
]


def get_disk_free_gb(path: str = "/") -> float:
    """Returns free disk space in GB."""
    try:
        total, used, free = shutil.disk_usage(path)
        return free / (1024 ** 3)
    except Exception:
        return -1


def download_repo(repo_id: str, token: str | None):
    """Download a single repo with aggressive filtering."""
    from huggingface_hub import snapshot_download
    
    snapshot_download(
        repo_id=repo_id,
        token=token,
        ignore_patterns=IGNORE_PATTERNS,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Descarga modelos de HuggingFace para Video Generator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--all", action="store_true",
                        help="Descarga TODOS los modelos (~50 GB)")
    parser.add_argument("--minimal", action="store_true",
                        help="Solo CogVideoX-5B T2V (~20 GB)")
    parser.add_argument("--select", nargs="+", type=int, metavar="N",
                        help="Descarga modelos por número (ej: --select 1 3)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    print("")
    print("══════════════════════════════════════════════")
    print(" 📥 Video Generator — Descarga de Modelos")
    print("══════════════════════════════════════════════")

    # Show disk space
    free_gb = get_disk_free_gb("/workspace" if os.path.exists("/workspace") else "/")
    if free_gb > 0:
        status = "✅" if free_gb > 60 else "⚠️" if free_gb > 25 else "❌"
        print(f"  {status} Espacio libre en disco: {free_gb:.1f} GB")
        if free_gb < 25:
            print("  ❌ ADVERTENCIA: Menos de 25 GB libres. Usa --minimal o libera espacio.")
    
    if token:
        print(f"  ✅ HF_TOKEN detectado (longitud={len(token)})")
    else:
        print("  ⚠️  HF_TOKEN no definido (algunos modelos gated podrían fallar)")

    print("")
    print("  Modelos disponibles:")
    for num, info in MODEL_REPOS.items():
        print(f"    [{num}] {info['label']}  ({info['size_est']})")
    print("")

    # Determine which models to download
    if args.minimal:
        selected = [1]
        print("  📦 Modo minimal: solo CogVideoX-5B")
    elif args.all:
        selected = list(MODEL_REPOS.keys())
        print("  📦 Modo completo: descargando TODOS los modelos")
    elif args.select:
        selected = [n for n in args.select if n in MODEL_REPOS]
        if not selected:
            print("  ❌ Ningún número válido. Usa 1, 2, o 3.")
            sys.exit(1)
        names = ", ".join(MODEL_REPOS[n]["label"] for n in selected)
        print(f"  📦 Seleccionados: {names}")
    else:
        # Interactive mode
        print("  Ingresa los números separados por espacio (ej: 1 3)")
        print("  O presiona Enter para descargar solo el modelo principal [1]:")
        try:
            choice = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = ""
        
        if not choice:
            selected = [1]
        else:
            try:
                selected = [int(x) for x in choice.split() if int(x) in MODEL_REPOS]
            except ValueError:
                selected = [1]
        
        if not selected:
            selected = [1]
        names = ", ".join(MODEL_REPOS[n]["label"] for n in selected)
        print(f"  📦 Descargando: {names}")

    print("")

    # Download
    success = 0
    failed = 0
    for num in selected:
        info = MODEL_REPOS[num]
        print(f"  ➜ [{num}/{len(selected)}] Descargando: {info['repo']} ({info['size_est']})...")
        try:
            download_repo(info["repo"], token)
            print(f"    ✅ {info['repo']} — OK")
            success += 1
        except Exception as e:
            print(f"    ❌ Error: {e}")
            failed += 1

    # Summary
    print("")
    print("══════════════════════════════════════════════")
    if failed == 0:
        print(f" 🎉 {success} modelo(s) descargado(s) exitosamente.")
    else:
        print(f" ⚠️ {success} exitoso(s), {failed} fallido(s).")
    
    free_after = get_disk_free_gb("/workspace" if os.path.exists("/workspace") else "/")
    if free_after > 0:
        print(f" 💾 Espacio libre restante: {free_after:.1f} GB")
    print("══════════════════════════════════════════════")


if __name__ == "__main__":
    main()
