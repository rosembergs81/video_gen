"""
modules/cache_manager.py
────────────────────────
Implementa un caché en disco (LRU manual) para segmentos de video generados.
Evita el recálculo (ahorrando tiempo y GPU) cuando el usuario re-evalúa
un "Story Mode" sin modificar los prompts iniciales y sus semillas.
"""

import os
import json
import hashlib
import shutil
import time
from pathlib import Path
from PIL import Image

CACHE_DIR = Path(".cache/segments")
MAX_CACHE_SIZE_MB = 2000  # 2 GB max cache size

class SegmentCache:
    def __init__(self, cache_dir: Path | str = CACHE_DIR, max_mb: int = MAX_CACHE_SIZE_MB):
        self.cache_dir = Path(cache_dir)
        self.max_bytes = max_mb * 1024 * 1024
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._enforce_limits()

    def _compute_hash(self, model_key: str, prompt: str, negative: str, 
                      num_frames: int, guidance: float, steps: int, 
                      width: int, height: int, seed: int, 
                      loras: list, v2v_strength: float = 0.0) -> str:
        """Generates a unique SHA-256 hash for the generation parameters."""
        data = {
            "model": model_key,
            "prompt": prompt.strip().lower(),
            "negative": (negative or "").strip().lower(),
            "frames": num_frames,
            "guidance": guidance,
            "steps": steps,
            "width": width,
            "height": height,
            "seed": seed,
            "loras": sorted(loras),  # order doesn't matter for hash
            "v2v_strength": v2v_strength
        }
        # Serialize stably
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_cached_segment(self, **kwargs) -> list | None:
        """
        Attempts to load frames from cache if they exist.
        Takes the same kwargs as _compute_hash.
        """
        hash_key = self._compute_hash(**kwargs)
        target_dir = self.cache_dir / hash_key
        
        if not target_dir.exists():
            return None

        # Touch directory to update access time (for LRU)
        os.utime(target_dir, None)

        try:
            # Load frames
            frames = []
            files = sorted(target_dir.glob("frame_*.jpg"))
            if not files:
                return None
                
            for file in files:
                img = Image.open(file).convert("RGB")
                img.load() # force load so file can be closed
                frames.append(img)
            return frames
        except Exception as e:
            print(f"[CacheManager] Error reading cache {hash_key}: {e}")
            return None

    def save_cached_segment(self, frames: list, **kwargs):
        """
        Saves a list of PIL Images to the cache directory.
        """
        hash_key = self._compute_hash(**kwargs)
        target_dir = self.cache_dir / hash_key
        
        # Don't overwrite if it exists and is complete
        if target_dir.exists():
            return
            
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                frame.save(target_dir / f"frame_{i:04d}.jpg", "JPEG", quality=90)
            
            self._enforce_limits()
        except Exception as e:
            print(f"[CacheManager] Error saving cache {hash_key}: {e}")
            # Cleanup on failure
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)

    def _enforce_limits(self):
        """Purge oldest caches if we exceed max_bytes."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        if total_size <= self.max_bytes:
            return

        # Get all subdirectories sorted by last modified (oldest first)
        dirs = [d for d in self.cache_dir.iterdir() if d.is_dir()]
        dirs.sort(key=lambda x: x.stat().st_mtime)

        while total_size > self.max_bytes and dirs:
            oldest = dirs.pop(0)
            try:
                # Calculate size of this directory
                dir_size = sum(f.stat().st_size for f in oldest.rglob('*') if f.is_file())
                shutil.rmtree(oldest)
                total_size -= dir_size
                print(f"[CacheManager] Caché LRU: eliminado segmento antiguo {oldest.name}")
            except Exception as e:
                print(f"[CacheManager] Error al limpiar caché {oldest.name}: {e}")

segment_cache = SegmentCache()
