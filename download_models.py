import os
import urllib.request

def download_file(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path}...")
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"Downloaded {local_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    else:
        print(f"Already exists {local_path}")

base_dir = "./checkpoints"

download_file("https://huggingface.co/QuarkAudio/QuarkAudio-UniSE/resolve/main/epoch%3D20-step%3D109367.ckpt", f"{base_dir}/epoch=20-step=109367.ckpt")

download_file("https://huggingface.co/SparkAudio/Spark-TTS-0.5B/resolve/main/BiCodec/config.yaml", f"{base_dir}/BiCodec/config.yaml")

download_file("https://huggingface.co/SparkAudio/Spark-TTS-0.5B/resolve/main/BiCodec/model.safetensors", f"{base_dir}/BiCodec/model.safetensors")

download_file("https://huggingface.co/SparkAudio/Spark-TTS-0.5B/resolve/main/config.yaml", f"{base_dir}/config.yaml")

download_file("https://huggingface.co/SparkAudio/Spark-TTS-0.5B/resolve/main/wav2vec2-large-xlsr-53/config.json", f"{base_dir}/wav2vec2-large-xlsr-53/config.json")

download_file("https://huggingface.co/SparkAudio/Spark-TTS-0.5B/resolve/main/wav2vec2-large-xlsr-53/preprocessor_config.json", f"{base_dir}/wav2vec2-large-xlsr-53/preprocessor_config.json")

download_file("https://huggingface.co/SparkAudio/Spark-TTS-0.5B/resolve/main/wav2vec2-large-xlsr-53/pytorch_model.bin", f"{base_dir}/wav2vec2-large-xlsr-53/pytorch_model.bin")

print("All done!")
