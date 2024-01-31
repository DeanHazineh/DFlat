import zipfile
import os
import shutil

CKPT_LINK = "https://www.dropbox.com/scl/fi/83vpqe8zolh9tj6g84iwz/ckpt.zip?rlkey=1p6l69n3r8at626hx2m5om1zy&dl=1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_PATH = os.path.join(SCRIPT_DIR, "model_ckpts.zip")
UNZIP_PATH = os.path.join(SCRIPT_DIR, "temp_fold/")


def download_file(url, save_path):
    import requests

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return


def unzip_file(zip_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    return


def unpack_and_move():
    ckpt_fold_names = os.listdir(os.path.join(UNZIP_PATH, "ckpt/"))
    moveto = os.path.join(SCRIPT_DIR, "dflat", "metasurface", "ckpt")

    if not os.path.exists(moveto):
        os.makedirs(moveto)

    for fname in ckpt_fold_names:
        source_path = os.path.join(UNZIP_PATH, "ckpt", fname)
        target_path = os.path.join(moveto, fname)

        if os.path.exists(target_path):
            if os.path.isdir(target_path):
                for item in os.listdir(source_path):
                    shutil.move(
                        os.path.join(source_path, item), os.path.join(target_path, item)
                    )
            else:
                print(
                    f"Error: A file with the name '{fname}' already exists in the target directory."
                )
        else:
            shutil.move(source_path, target_path)

    return


def execute_data_management():
    print("Downloading Dflat checkpoints from dropbox...")
    download_file(CKPT_LINK, DEST_PATH)

    print("Unzipping data...")
    unzip_file(DEST_PATH, UNZIP_PATH)

    print("Moving data files to ckpt folders")
    unpack_and_move()

    print("Cleaning and deleting the initial zip file")
    os.remove(DEST_PATH)
    shutil.rmtree(UNZIP_PATH)

    print("Downloading model checkpoints finished")
    return
