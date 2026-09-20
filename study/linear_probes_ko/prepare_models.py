"""Explicitly download one pinned model on a new instance; never persist a token."""
import argparse
import getpass
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["truth", "instruct"])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    spec = json.loads((here / "models.json").read_text())[args.model]
    token = os.environ.get("HF_TOKEN") or getpass.getpass("HF read token (not saved): ")
    path = snapshot_download(
        spec["repo_id"], revision=spec["revision"], token=token,
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.tiktoken"],
        max_workers=4,
    )
    del token
    target = here / "model_paths.local.json"
    paths = json.loads(target.read_text()) if target.exists() else {}
    paths[spec["repo_id"]] = path
    target.write_text(json.dumps(paths, indent=2))
    print("Ready:", spec["repo_id"])


if __name__ == "__main__":
    main()
