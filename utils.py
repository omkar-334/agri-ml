import contextlib
import os
import sys
import zipfile
from typing import Literal

from huggingface_hub import HfApi, HfFolder, hf_hub_download, upload_file


def upload(paths, repo="omkar334/agri", repo_type: Literal["model", "space", "dataset"] = "model", token=None):
    token = token or HfFolder.get_token()  # Automatically uses your token if logged in

    # api = HfApi()

    for file_path in paths:
        filename = os.path.basename(file_path)
        print(f"Uploading {filename}...")
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=repo,
            repo_type=repo_type,
            token=token,
        )
    print("✅ Upload complete.")


def download(files, repo="omkar334/agri", local_dir="./models"):
    os.makedirs(local_dir, exist_ok=True)

    # Download each model
    for filename in files:
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=local_dir,
        )


def create_zip(output_dir, zip_filename="output.zip"):
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, output_dir))

    print(f"Folder '{output_dir}' has been zipped as '{zip_filename}'")


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = fnull
            sys.stderr = fnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    model_files = [
        "maianet_nirmal.pth",
        "soyatrans_nirmal.pth",
        "tswinf_nirmalsankana.pth",
        "maianet_pungliya.pth",
        "soyatrans_pungliya.pth",
        "tswinf_pungliyavithika.pth",
        "maianet_mendeley.pth",
        "soyatrans_mendeley.pth",
        "tswinf_mendeley.pth",
    ]
    download(model_files, repo="omkar334/agri", local_dir="./models")

    paths = ["coreplant_pungliya.ipynb", "coreplant_nirmal.ipynb", "coreplant_mendeley.ipynb"]
    upload(paths, repo="omkar334/agri", repo_type="model")
