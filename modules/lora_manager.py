"""
modules/lora_manager.py
────────────────────────
Maneja el registro, adición y eliminación de LoRAs personalizados en la UI.
"""
import os
import json
from pathlib import Path
import gradio as gr

LORAS_DIR   = Path("loras")
LORAS_DIR.mkdir(exist_ok=True)
LORAS_INDEX = LORAS_DIR / "index.json"


def load_lora_index() -> list:
    if LORAS_INDEX.exists():
        try:
            return json.loads(LORAS_INDEX.read_text())
        except Exception:
            return []
    return []


def save_lora_index(idx: list):
    LORAS_INDEX.write_text(json.dumps(idx, indent=2, ensure_ascii=False))


def get_lora_choices(idx: list = None) -> list:
    if idx is None:
        idx = load_lora_index()
    return [f"{l['name']}  [{l['category']}]" for l in idx]


def add_lora(name, source, category, scale, weight_name, frame_start, frame_end, curve, thumbnail_url=None):
    name = name.strip(); source = source.strip()
    if not name or not source:
        return "⚠️ Nombre y fuente son obligatorios.", gr.update(), gr.update()
    
    idx = load_lora_index()
    if any(l["name"] == name for l in idx):
        return f"⚠️ Ya existe '{name}'.", gr.update(), gr.update()
    
    idx.append({
        "name": name, "source": source, "category": category,
        "scale": round(float(scale), 2),
        "weight_name": weight_name.strip() or None,
        "frame_start": int(frame_start),
        "frame_end":   int(frame_end),
        "curve":       curve,
        "thumbnail":   thumbnail_url.strip() if thumbnail_url else None,
    })
    save_lora_index(idx)
    choices = get_lora_choices(idx)
    return f"✅ LoRA '{name}' añadida.", gr.update(choices=choices, value=[]), gr.update(choices=choices)


def remove_loras(selected):
    if not selected:
        return "⚠️ Selecciona al menos una.", gr.update(), gr.update()
    
    idx = load_lora_index()
    idx = [l for l in idx if f"{l['name']}  [{l['category']}]" not in selected]
    save_lora_index(idx)
    choices = get_lora_choices(idx)
    return f"🗑️ {len(selected)} eliminada(s).", gr.update(choices=choices, value=[]), gr.update(choices=choices, value=[])


def refresh_loras():
    choices = get_lora_choices()
    return gr.update(choices=choices), gr.update(choices=choices)
