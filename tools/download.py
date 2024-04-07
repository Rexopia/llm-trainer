import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

SAVE_PATH = "/home/ruiji/workspace/datas/snapshot_download"

snapshot_download(
    repo_id="google/gemma-7b",
    repo_type="model",
    local_dir=f"/home/ruiji/workspace/models/snapshot_download/gemma-7b",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns="model-*",
    max_workers=1
)

# snapshot_download(
#     repo_id="Tele-AI/TeleChat-PTD",
#     repo_type="dataset",
#     local_dir="/nfs/datas/snapshot_download/TeleChat-PTD",
#     local_dir_use_symlinks=False,
#     resume_download=True,
# )