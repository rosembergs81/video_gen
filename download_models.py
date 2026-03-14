import os
from huggingface_hub import snapshot_download
from modules.pipeline_utils import MODELS

def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    print("==============================================")
    print(" 📥 Iniciando descarga de modelos en caché...")
    print("==============================================")
    
    # Recopilar repos únicos para no descargar múltiples veces el mismo (ej. CogVideoX-5b se usa para T2V y V2V)
    unique_repos = set(cfg["repo"] for cfg in MODELS.values())
    
    for repo in unique_repos:
        print(f"➜ Descargando: {repo}...")
        try:
            # Ignoramos pesos en formatos redundantes para ahorrar espacio (ej. msgpack, h5 o pytorch_model.bin si usamos safetensors)
            snapshot_download(
                repo_id=repo, 
                token=token, 
                ignore_patterns=["*.msgpack", "*.bin", "*.h5", "*.ckpt"]
            )
            print(f"  ✅ {repo} precargado exitosamente.")
        except Exception as e:
            print(f"  ❌ Error descargando {repo}: {e}")
            
    print("==============================================")
    print(" 🎉 Todas las descargas solicitadas han finalizado.")
    print("==============================================")

if __name__ == "__main__":
    main()
